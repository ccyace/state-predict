"""
用两路图像 Tensor 计算 FID（pytorch-fid Inception-v3）。

从目录加载为 NCHW float 张量 [0,1]，再提特征求 Frechet 距离；
也可直接加载已保存的 .pt。

示例:
  python scripts/compute_fid_tensor.py \\
    --real C:/Users/ASUS/Desktop/ODE-scale/real_images \\
    --gen  C:/Users/ASUS/Desktop/ODE-scale/w4a8test_250

  # 首次加载后缓存为 .pt，下次更快
  python scripts/compute_fid_tensor.py --real real_images --gen w4a8test_250 --save_pt
  python scripts/compute_fid_tensor.py --real_pt real.pt --gen_pt gen.pt
"""

from __future__ import annotations

import argparse
import re
import os
from pathlib import Path

import numpy as np
import torch
try:
    from natsort import natsorted
except ImportError:
    def natsorted(values):
        def key(value):
            return [
                int(part) if part.isdigit() else part.lower()
                for part in re.split(r"(\d+)", str(value))
            ]
        return sorted(values, key=key)
from PIL import Image
from torch.nn.functional import adaptive_avg_pool2d
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def list_images(root: Path) -> list[Path]:
    files = [p for p in root.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    return natsorted(files, key=lambda p: p.name)


def load_images_as_tensor(img_dir: str | Path, desc: str) -> torch.Tensor:
    """读取目录下全部 RGB 图 -> (N, 3, H, W) float32, 范围 [0, 1]。"""
    root = Path(img_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"不是目录: {root}")
    files = list_images(root)
    if not files:
        raise RuntimeError(f"目录下没有图像: {root}")

    print(f"{desc}: 加载 {len(files)} 张 -> tensor ({root})", flush=True)
    tensors = []
    for path in tqdm(files, desc=desc, mininterval=1.0):
        with Image.open(path) as img:
            arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        # HWC -> CHW
        tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
    out = torch.stack(tensors, dim=0)
    print(f"{desc}: tensor shape={tuple(out.shape)}, dtype={out.dtype}", flush=True)
    return out


def load_tensor_pt(path: str | Path, desc: str) -> torch.Tensor:
    path = Path(path).resolve()
    print(f"{desc}: 从 .pt 加载 {path}", flush=True)
    x = torch.load(path, map_location="cpu")
    if isinstance(x, dict):
        for key in ("images", "x", "data", "tensor"):
            if key in x:
                x = x[key]
                break
        else:
            raise KeyError(f".pt 是 dict，未找到 images/x/data/tensor: {list(x.keys())}")
    if not torch.is_tensor(x):
        raise TypeError(f"期望 Tensor，得到 {type(x)}")
    if x.dtype != torch.float32:
        x = x.float()
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError(f"期望 (N,3,H,W)，得到 shape={tuple(x.shape)}")
    if x.max() > 1.5:
        x = x / 255.0
    print(f"{desc}: tensor shape={tuple(x.shape)}", flush=True)
    return x


@torch.no_grad()
def activation_stats(
    images: torch.Tensor,
    model: torch.nn.Module,
    batch_size: int,
    device: torch.device,
    dims: int,
    desc: str,
) -> tuple[np.ndarray, np.ndarray]:
    """对 (N,3,H,W)[0,1] 提 Inception 特征，返回 mu, sigma。"""
    model.eval()
    n = images.shape[0]
    if batch_size > n:
        batch_size = n

    loader = DataLoader(
        TensorDataset(images),
        batch_size=batch_size,
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

    mu = np.mean(acts, axis=0)
    sigma = np.cov(acts, rowvar=False)
    return mu, sigma


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
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean))


def fid_from_tensors(
    real: torch.Tensor,
    gen: torch.Tensor,
    batch_size: int = 64,
    device: str | None = None,
    dims: int = 2048,
) -> float:
    from pytorch_fid.inception import InceptionV3

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_t = torch.device(device)
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    print(f"加载 InceptionV3 (dims={dims}) -> {device_t}", flush=True)
    model = InceptionV3([block_idx]).to(device_t)

    print("提取真实图像特征...", flush=True)
    mu_r, sigma_r = activation_stats(real, model, batch_size, device_t, dims, "real feats")
    print("提取生成图像特征...", flush=True)
    mu_g, sigma_g = activation_stats(gen, model, batch_size, device_t, dims, "gen feats")
    return frechet_distance(mu_r, sigma_r, mu_g, sigma_g)


def parse_args():
    p = argparse.ArgumentParser(description="FID from image tensors (pytorch-fid)")
    p.add_argument("--real", type=str, default=None, help="真实图像目录")
    p.add_argument("--gen", type=str, default=None, help="生成图像目录")
    p.add_argument("--real_pt", type=str, default=None, help="真实图像 .pt (N,3,H,W)")
    p.add_argument("--gen_pt", type=str, default=None, help="生成图像 .pt (N,3,H,W)")
    p.add_argument("--save_pt", action="store_true", help="把加载的 tensor 存成 .pt")
    p.add_argument("--real_pt_out", type=str, default=None, help="真实 tensor 保存路径")
    p.add_argument("--gen_pt_out", type=str, default=None, help="生成 tensor 保存路径")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--dims", type=int, default=2048, choices=[64, 192, 768, 2048])
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--max_images", type=int, default=None, help="两边各最多用前 N 张")
    return p.parse_args()


def main():
    args = parse_args()

    if args.real_pt:
        real = load_tensor_pt(args.real_pt, "real")
    elif args.real:
        real = load_images_as_tensor(args.real, "real")
    else:
        raise SystemExit("请指定 --real 或 --real_pt")

    if args.gen_pt:
        gen = load_tensor_pt(args.gen_pt, "gen")
    elif args.gen:
        gen = load_images_as_tensor(args.gen, "gen")
    else:
        raise SystemExit("请指定 --gen 或 --gen_pt")

    if args.max_images is not None:
        n = args.max_images
        real, gen = real[:n], gen[:n]
        print(f"截断为前 {n} 张: real={tuple(real.shape)}, gen={tuple(gen.shape)}", flush=True)

    if args.save_pt or args.real_pt_out or args.gen_pt_out:
        real_out = args.real_pt_out or (
            str(Path(args.real).resolve().with_name(Path(args.real).name + "_tensor.pt"))
            if args.real
            else None
        )
        gen_out = args.gen_pt_out or (
            str(Path(args.gen).resolve().with_name(Path(args.gen).name + "_tensor.pt"))
            if args.gen
            else None
        )
        if real_out and not args.real_pt:
            print(f"保存 real tensor -> {real_out}", flush=True)
            torch.save(real, real_out)
        if gen_out and not args.gen_pt:
            print(f"保存 gen tensor -> {gen_out}", flush=True)
            torch.save(gen, gen_out)

    fid = fid_from_tensors(
        real, gen, batch_size=args.batch_size, device=args.device, dims=args.dims
    )
    print(f"FID (tensor): {fid}", flush=True)


if __name__ == "__main__":
    main()
