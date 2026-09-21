

from __future__ import annotations

import os
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import adaptive_avg_pool2d
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

IMAGE_KEYS = ("images", "imgs", "x", "data", "arr_0")

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_REAL_NPZ = PROJECT_ROOT / "new_real_images" / "cifar10_python.npz"
DEFAULT_GEN_NPZ = PROJECT_ROOT / "w4a8test" / "images.npz"


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


def load_stats_or_images(path: Path, prefer_stats: bool = True):
    """返回 (mode, mu, sigma, images)，mode 为 'stats' 或 'images'。"""
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在: {path}")
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
            images = np.asarray(_pick_images(npz)).copy()
        except KeyError:
            if has_stats:
                mu, sigma = npz["mu"], npz["sigma"]
                print(f"  回退到统计量: mu={mu.shape}, sigma={sigma.shape}", flush=True)
                return "stats", mu, sigma, None
            raise
        print(f"  使用图像数组: shape={images.shape}, dtype={images.dtype}", flush=True)
        return "images", None, None, images


def images_to_nchw_float(images: np.ndarray) -> torch.Tensor:
    x = images
    if x.ndim != 4:
        raise ValueError(f"期望 4D 图像数组，得到 shape={x.shape}")
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
    for (batch,) in tqdm(loader, desc="Inception feats", mininterval=1.0):
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
    model: torch.nn.Module | None,
    batch_size: int,
    device: torch.device,
    dims: int,
) -> tuple[np.ndarray, np.ndarray]:
    mode, mu, sigma, images = load_stats_or_images(path)
    if mode == "stats":
        return mu, sigma
    assert images is not None and model is not None
    tensor = images_to_nchw_float(images)
    print(f"  tensor={tuple(tensor.shape)}", flush=True)
    return activation_stats(tensor, model, batch_size, device, dims)


def calculate_fid(
    gen_npz: str | Path,
    real_npz: str | Path = DEFAULT_REAL_NPZ,
    batch_size: int = 64,
    dims: int = 2048,
) -> float:
    real_npz = Path(real_npz).expanduser().resolve()
    gen_path = Path(gen_npz).resolve()
    if not real_npz.is_file():
        raise FileNotFoundError(f"真实参考不存在: {real_npz}")
    if not gen_path.is_file() or gen_path.suffix.lower() != ".npz":
        raise FileNotFoundError(f"生成集必须是已存在的 .npz 文件: {gen_path}")

    need_model = False
    for p in (real_npz, gen_path):
        with np.load(p, allow_pickle=False) as npz:
            has_stats = "mu" in npz.files and "sigma" in npz.files
            has_imgs = any(k in npz.files for k in IMAGE_KEYS)
            if has_stats:
                continue
            if has_imgs:
                need_model = True
            else:
                raise ValueError(f"{p} 既无 mu/sigma 也无 images")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    if need_model:
        from pytorch_fid.inception import InceptionV3

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        print(f"加载 InceptionV3 (dims={dims}) -> {device}", flush=True)
        model = InceptionV3([block_idx]).to(device)

    mu1, sigma1 = stats_from_npz(real_npz, model, batch_size, device, dims)
    mu2, sigma2 = stats_from_npz(gen_path, model, batch_size, device, dims)
    return frechet_distance(mu1, sigma1, mu2, sigma2)


calculate_fid_npz = calculate_fid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate FID from two NPZ files.")
    parser.add_argument("gen_npz", nargs="?", default=DEFAULT_GEN_NPZ)
    parser.add_argument("--real-npz", default=DEFAULT_REAL_NPZ)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dims", type=int, default=2048)
    args = parser.parse_args()
    fid_value = calculate_fid(
        args.gen_npz,
        real_npz=args.real_npz,
        batch_size=args.batch_size,
        dims=args.dims,
    )
    print(f"FID 值: {fid_value}")
