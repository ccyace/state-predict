"""
用两个 .npz 计算 FID（pytorch-fid Inception-v3）。

支持的 .npz 内容（优先级从高到低）:
  1) 已有统计量: mu, sigma  —— 直接算 Frechet，最快
  2) 图像数组: images / imgs / x  —— 提特征后再算
     形状可为 (N,H,W,C) 或 (N,C,H,W)；uint8 或 float

示例:
  python scripts/compute_fid_npz.py \\
    --npz1 real_images_last10k_noise.npz \\
    --npz2 w4a8test/images.npz

  # 两边若已含 mu/sigma，会跳过 Inception，几乎秒出
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import adaptive_avg_pool2d
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

IMAGE_KEYS = ("images", "imgs", "x", "data", "arr_0")


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6) -> float:
    from scipy import linalg

    mu1, mu2 = np.atleast_1d(mu1), np.atleast_1d(mu2)
    sigma1, sigma2 = np.atleast_2d(sigma1), np.atleast_2d(sigma2)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f"Imaginary component {np.max(np.abs(covmean.imag))}")
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def _pick_images(npz: np.lib.npyio.NpzFile) -> np.ndarray:
    for key in IMAGE_KEYS:
        if key in npz.files:
            return npz[key]
    raise KeyError(
        f".npz 中未找到图像键 {IMAGE_KEYS}，现有键: {list(npz.files)}"
    )


def load_stats_or_images(
    path: Path,
    prefer_stats: bool,
) -> tuple[str, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    返回 (mode, mu, sigma, images)
    mode: "stats" | "images"
    """
    print(f"加载 {path} ...", flush=True)
    with np.load(path, allow_pickle=False) as npz:
        keys = list(npz.files)
        print(f"  keys={keys}", flush=True)
        has_stats = "mu" in npz.files and "sigma" in npz.files
        if prefer_stats and has_stats:
            mu, sigma = npz["mu"], npz["sigma"]
            print(f"  使用预计算统计量: mu={mu.shape}, sigma={sigma.shape}", flush=True)
            return "stats", mu, sigma, None
        try:
            images = _pick_images(npz)
        except KeyError:
            if has_stats:
                mu, sigma = npz["mu"], npz["sigma"]
                print(f"  回退到统计量: mu={mu.shape}, sigma={sigma.shape}", flush=True)
                return "stats", mu, sigma, None
            raise
        images = np.asarray(images)
        print(f"  使用图像数组: shape={images.shape}, dtype={images.dtype}", flush=True)
        return "images", None, None, images.copy()


def images_to_nchw_float(images: np.ndarray, max_images: int | None) -> torch.Tensor:
    x = images
    if max_images is not None:
        x = x[:max_images]
    if x.ndim != 4:
        raise ValueError(f"期望 4D 图像数组，得到 shape={x.shape}")

    # (N,H,W,C) -> (N,C,H,W)
    if x.shape[-1] in (1, 3):
        x = np.transpose(x, (0, 3, 1, 2))
    elif x.shape[1] not in (1, 3):
        raise ValueError(f"无法判断通道维: shape={x.shape}")

    t = torch.from_numpy(np.ascontiguousarray(x))
    if t.dtype == torch.uint8:
        t = t.float() / 255.0
    else:
        t = t.float()
        if t.max() > 1.5:
            t = t / 255.0
    return t


@torch.no_grad()
def activation_stats(
    images: torch.Tensor,
    model: torch.nn.Module,
    batch_size: int,
    device: torch.device,
    dims: int,
    desc: str,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    n = images.shape[0]
    bs = min(batch_size, n)
    loader = DataLoader(
        TensorDataset(images),
        batch_size=bs,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    acts = np.empty((n, dims), dtype=np.float64)
    start = 0
    for (batch,) in tqdm(loader, desc=desc, mininterval=1.0):
        batch = batch.to(device, non_blocking=True)
        pred = model(batch)[0]
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        pred = pred.squeeze(-1).squeeze(-1).cpu().numpy()
        acts[start : start + pred.shape[0]] = pred
        start += pred.shape[0]
    return np.mean(acts, axis=0), np.cov(acts, rowvar=False)


def stats_from_npz(
    path: Path,
    prefer_stats: bool,
    model: torch.nn.Module | None,
    batch_size: int,
    device: torch.device,
    dims: int,
    max_images: int | None,
    desc: str,
) -> tuple[np.ndarray, np.ndarray]:
    mode, mu, sigma, images = load_stats_or_images(path, prefer_stats=prefer_stats)
    if mode == "stats":
        return mu, sigma
    assert images is not None and model is not None
    tensor = images_to_nchw_float(images, max_images)
    print(f"  {desc} tensor={tuple(tensor.shape)}", flush=True)
    return activation_stats(tensor, model, batch_size, device, dims, f"{desc} feats")


def parse_args():
    p = argparse.ArgumentParser(description="FID between two .npz files")
    p.add_argument("--npz1", type=str, required=True, help="参考集 .npz（如真实图像）")
    p.add_argument("--npz2", type=str, required=True, help="生成集 .npz")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--dims", type=int, default=2048, choices=[64, 192, 768, 2048])
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--max_images", type=int, default=None, help="两侧最多各用前 N 张图")
    p.add_argument(
        "--force_images",
        action="store_true",
        help="即使 npz 含 mu/sigma，也强制从 images 重算特征",
    )
    p.add_argument(
        "--save_stats1",
        type=str,
        default=None,
        help="可选：把 npz1 算出的 mu/sigma 另存为 .npz",
    )
    p.add_argument(
        "--save_stats2",
        type=str,
        default=None,
        help="可选：把 npz2 算出的 mu/sigma 另存为 .npz",
    )
    return p.parse_args()


def main():
    args = parse_args()
    path1 = Path(args.npz1).resolve()
    path2 = Path(args.npz2).resolve()
    for p in (path1, path2):
        if not p.is_file():
            raise SystemExit(f"文件不存在: {p}")

    prefer_stats = not args.force_images
    # 先窥探是否两边都能直接用 stats
    need_model = False
    for p in (path1, path2):
        with np.load(p, allow_pickle=False) as npz:
            has_stats = "mu" in npz.files and "sigma" in npz.files
            has_imgs = any(k in npz.files for k in IMAGE_KEYS)
            if prefer_stats and has_stats:
                continue
            if has_imgs:
                need_model = True
            elif not has_stats:
                raise SystemExit(f"{p} 既无 mu/sigma 也无 images")

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = None
    if need_model:
        from pytorch_fid.inception import InceptionV3

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[args.dims]
        print(f"加载 InceptionV3 (dims={args.dims}) -> {device}", flush=True)
        model = InceptionV3([block_idx]).to(device)

    mu1, sigma1 = stats_from_npz(
        path1, prefer_stats, model, args.batch_size, device, args.dims, args.max_images, "npz1"
    )
    mu2, sigma2 = stats_from_npz(
        path2, prefer_stats, model, args.batch_size, device, args.dims, args.max_images, "npz2"
    )

    if args.save_stats1:
        out = Path(args.save_stats1).resolve()
        np.savez_compressed(out, mu=mu1, sigma=sigma1)
        print(f"已保存 npz1 统计量 -> {out}", flush=True)
    if args.save_stats2:
        out = Path(args.save_stats2).resolve()
        np.savez_compressed(out, mu=mu2, sigma=sigma2)
        print(f"已保存 npz2 统计量 -> {out}", flush=True)

    fid = frechet_distance(mu1, sigma1, mu2, sigma2)
    print(f"FID (npz vs npz): {fid}", flush=True)


if __name__ == "__main__":
    main()
