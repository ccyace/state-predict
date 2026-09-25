"""
对 CIFAR-10 真实图做可控加噪，并导出为 .npz（含可选 Inception mu/sigma）。

支持:
  - 单次导出（--noise_std / --tail）
  - 扫参导出（--sweep_std）
  - 按生成集高频能量标定建议 σ*（--calibrate_gen）

示例:
  # 扫参写出多个参考 npz
  python scripts/make_noisy_real_npz.py \\
    --input real_images \\
    --out_dir real_noise_refs \\
    --tail 10000 \\
    --sweep_std 0,0.01,0.02,0.03,0.05,0.08

  # 用 W4A8/W8A8 生成图标定建议 noise_std（不改生成图）
  python scripts/make_noisy_real_npz.py \\
    --input real_images \\
    --sweep_std 0,0.01,0.02,0.03,0.05,0.08 \\
    --calibrate_gen w4a8test/images.npz w8a8test/images.npz \\
    --calibrate_only
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import torch
from natsort import natsorted
from PIL import Image
from torch.nn.functional import adaptive_avg_pool2d
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
IMAGE_KEYS = ("images", "imgs", "x", "data", "arr_0")


def list_images(root: Path) -> list[Path]:
    files = [p for p in root.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    return natsorted(files, key=lambda p: p.name)


def load_all_uint8(img_dir: Path) -> np.ndarray:
    files = list_images(img_dir)
    if not files:
        raise RuntimeError(f"目录下没有图像: {img_dir}")
    print(f"加载 {len(files)} 张图像: {img_dir}", flush=True)
    imgs = []
    for path in tqdm(files, desc="load", mininterval=1.0):
        with Image.open(path) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        imgs.append(arr)
    out = np.stack(imgs, axis=0)
    print(f"images shape={out.shape}, dtype={out.dtype}", flush=True)
    return out


def load_images_any(path: Path, max_images: int | None = None) -> np.ndarray:
    """从 .npz 或图像目录加载 (N,H,W,3) uint8。"""
    path = path.resolve()
    if path.is_file() and path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as npz:
            for key in IMAGE_KEYS:
                if key in npz.files:
                    imgs = np.asarray(npz[key])
                    break
            else:
                raise KeyError(f"{path} 无图像键，keys={list(npz.files)}")
    elif path.is_dir():
        img_dir = path / "img" if (path / "img").is_dir() else path
        imgs = load_all_uint8(img_dir)
    else:
        raise FileNotFoundError(path)

    if imgs.ndim != 4:
        raise ValueError(f"期望 4D 图像，得到 {imgs.shape} from {path}")
    if imgs.shape[-1] not in (1, 3) and imgs.shape[1] in (1, 3):
        imgs = np.transpose(imgs, (0, 2, 3, 1))
    if imgs.dtype != np.uint8:
        x = imgs.astype(np.float32)
        if x.max() <= 1.5:
            x = x * 255.0
        imgs = np.clip(x + 0.5, 0, 255).astype(np.uint8)
    if max_images is not None:
        imgs = imgs[:max_images]
    return imgs


def add_mild_noise(
    images: np.ndarray,
    start_idx: int,
    noise_std: float,
    seed: int,
    verbose: bool = True,
) -> np.ndarray:
    """
    对 images[start_idx:] 加高斯噪声。
    noise_std: 相对 [0,1] 的标准差；0 表示不加噪（原样拷贝）。
    """
    if noise_std <= 0:
        if verbose:
            print(f"noise_std={noise_std} -> 不加噪（干净参考）", flush=True)
        return images.copy()

    out = images.astype(np.float32) / 255.0
    n_tail = out.shape[0] - start_idx
    if n_tail <= 0:
        raise ValueError(f"start_idx={start_idx} 超出图像数 {out.shape[0]}")

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, noise_std, size=out[start_idx:].shape).astype(np.float32)
    out[start_idx:] = np.clip(out[start_idx:] + noise, 0.0, 1.0)

    if verbose:
        print(
            f"已对后 {n_tail} 张加噪: index [{start_idx}, {out.shape[0]}), "
            f"noise_std={noise_std:.4f} (~{noise_std * 255:.1f}/255), seed={seed}",
            flush=True,
        )
    return (out * 255.0 + 0.5).clip(0, 255).astype(np.uint8)


def laplacian_hf_energy(images_uint8: np.ndarray, max_images: int | None = 5000) -> float:
    """
    平均 Laplacian 能量（高频代理）。
    对灰度图做 4-邻域 Laplace：4*c - n - s - e - w，再取平方均值。
    """
    x = images_uint8
    if max_images is not None and x.shape[0] > max_images:
        # 均匀取样，避免只看开头
        idx = np.linspace(0, x.shape[0] - 1, max_images).astype(np.int64)
        x = x[idx]
    g = x.astype(np.float32).mean(axis=-1)  # (N,H,W)
    c = g[:, 1:-1, 1:-1]
    lap = 4.0 * c - g[:, :-2, 1:-1] - g[:, 2:, 1:-1] - g[:, 1:-1, :-2] - g[:, 1:-1, 2:]
    return float(np.mean(lap * lap))


@torch.no_grad()
def compute_fid_stats(
    images_uint8: np.ndarray,
    batch_size: int,
    device: torch.device,
    dims: int = 2048,
    model: torch.nn.Module | None = None,
) -> tuple[np.ndarray, np.ndarray, torch.nn.Module]:
    """pytorch-fid 同款 Inception 统计量 mu / sigma；可复用 model。"""
    from pytorch_fid.inception import InceptionV3

    if model is None:
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        print(f"计算 FID 统计量 (Inception dims={dims}) on {device} ...", flush=True)
        model = InceptionV3([block_idx]).to(device)
    model.eval()

    x = torch.from_numpy(images_uint8).permute(0, 3, 1, 2).contiguous().float() / 255.0
    n = x.shape[0]
    bs = min(batch_size, n)
    loader = DataLoader(
        TensorDataset(x),
        batch_size=bs,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    acts = np.empty((n, dims), dtype=np.float64)
    start = 0
    for (batch,) in tqdm(loader, desc="fid-stats", mininterval=1.0):
        batch = batch.to(device, non_blocking=True)
        pred = model(batch)[0]
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        pred = pred.squeeze(-1).squeeze(-1).cpu().numpy()
        acts[start : start + pred.shape[0]] = pred
        start += pred.shape[0]

    mu = np.mean(acts, axis=0)
    sigma = np.cov(acts, rowvar=False)
    print(f"mu shape={mu.shape}, sigma shape={sigma.shape}", flush=True)
    return mu, sigma, model


def parse_sweep_stds(text: str) -> list[float]:
    parts = [p.strip() for p in re.split(r"[,;\s]+", text) if p.strip()]
    if not parts:
        raise ValueError("空的 --sweep_std")
    return [float(p) for p in parts]


def fmt_std(s: float) -> str:
    t = f"{s:.4f}".rstrip("0").rstrip(".")
    return t.replace(".", "p")


def save_one(
    out_path: Path,
    images: np.ndarray,
    noise_std: float,
    start_idx: int,
    tail: int,
    seed: int,
    mu: np.ndarray | None,
    sigma: np.ndarray | None,
    hf: float | None = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_kw: dict = {
        "images": images,
        "noise_std": np.array(noise_std, dtype=np.float32),
        "noise_start_idx": np.array(start_idx, dtype=np.int32),
        "noise_tail": np.array(tail, dtype=np.int32),
        "seed": np.array(seed, dtype=np.int32),
    }
    if hf is not None:
        save_kw["hf_laplacian"] = np.array(hf, dtype=np.float64)
    if mu is not None and sigma is not None:
        save_kw["mu"] = mu
        save_kw["sigma"] = sigma
    print(f"写入 {out_path} ...", flush=True)
    np.savez_compressed(out_path, **save_kw)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"完成: {out_path} ({size_mb:.1f} MB)", flush=True)


def calibrate_sigma(
    clean: np.ndarray,
    start_idx: int,
    seed: int,
    sweep_stds: list[float],
    gen_paths: list[Path],
    hf_max_images: int,
) -> dict[str, float]:
    """对每个 gen 找使 |HF(ref(σ))-HF(gen)| 最小的 σ*。"""
    print("\n=== HF 标定（Laplacian 能量）===", flush=True)
    gen_hf: dict[str, float] = {}
    for gp in gen_paths:
        gi = load_images_any(gp, max_images=None)
        h = laplacian_hf_energy(gi, max_images=hf_max_images)
        gen_hf[str(gp)] = h
        print(f"  gen HF[{gp.name}] = {h:.6f}  (n={gi.shape[0]})", flush=True)

    rows = []
    print(f"\n{'noise_std':>10} {'ref_HF':>14}", end="", flush=True)
    for gp in gen_paths:
        print(f"  |d-{gp.name[:12]}|", end="", flush=True)
    print(flush=True)

    best: dict[str, tuple[float, float]] = {str(gp): (float("inf"), 0.0) for gp in gen_paths}
    for std in sweep_stds:
        noisy = add_mild_noise(clean, start_idx, std, seed, verbose=False)
        hf = laplacian_hf_energy(noisy, max_images=hf_max_images)
        line = f"{std:10.4f} {hf:14.6f}"
        for gp in gen_paths:
            d = abs(hf - gen_hf[str(gp)])
            line += f"  {d:14.6f}"
            if d < best[str(gp)][0]:
                best[str(gp)] = (d, std)
        print(line, flush=True)
        rows.append((std, hf))

    suggested: dict[str, float] = {}
    print("\n建议 σ*（最小化 |HF_ref - HF_gen|）:", flush=True)
    vals = []
    for gp in gen_paths:
        d, std = best[str(gp)]
        suggested[str(gp)] = std
        vals.append(std)
        print(f"  {gp.name}: noise_std* = {std}  (|ΔHF|={d:.6f})", flush=True)
    if vals:
        mean_s = float(np.mean(vals))
        # 取网格上最接近均值的点
        nearest = min(sweep_stds, key=lambda s: abs(s - mean_s))
        suggested["__mean__"] = mean_s
        suggested["__grid_nearest_mean__"] = nearest
        print(f"  平均 σ* = {mean_s:.4f}  -> 网格最近点 = {nearest}", flush=True)
    return suggested


def parse_args():
    p = argparse.ArgumentParser(
        description="Mild-noise real images -> .npz (+sweep / HF calibrate)"
    )
    p.add_argument(
        "--input",
        type=str,
        default="real_images",
        help="真实图像目录",
    )
    p.add_argument(
        "--output",
        type=str,
        default="real_images_last10k_noise.npz",
        help="单次导出时的输出 .npz",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="扫参时输出目录（默认: <output 父目录>/real_noise_refs）",
    )
    p.add_argument("--tail", type=int, default=10000, help="对最后多少张加噪")
    p.add_argument(
        "--noise_std",
        type=float,
        default=0.03,
        help="单次导出的噪声 std（相对 [0,1]）；sweep 时忽略",
    )
    p.add_argument(
        "--sweep_std",
        type=str,
        default=None,
        help="逗号分隔噪声网格，如 0,0.01,0.02,0.03,0.05,0.08",
    )
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--no_stats", action="store_true", help="不计算 Inception mu/sigma")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--dims", type=int, default=2048, choices=[64, 192, 768, 2048])
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--calibrate_gen",
        type=str,
        nargs="+",
        default=None,
        help="生成集 .npz 或目录；用于 HF 标定建议 σ*",
    )
    p.add_argument(
        "--calibrate_only",
        action="store_true",
        help="只做 HF 标定，不写出参考 npz",
    )
    p.add_argument(
        "--hf_max_images",
        type=int,
        default=5000,
        help="HF 统计最多使用多少张图（均匀抽样）",
    )
    return p.parse_args()


def main():
    args = parse_args()
    img_dir = Path(args.input).resolve()
    clean = load_all_uint8(img_dir)
    n = clean.shape[0]
    if n < args.tail:
        raise SystemExit(f"图像数 {n} < tail={args.tail}")
    start_idx = n - args.tail

    sweep_stds = parse_sweep_stds(args.sweep_std) if args.sweep_std else None

    if args.calibrate_gen:
        if sweep_stds is None:
            sweep_stds = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08]
            print(f"标定默认网格 sweep_std={sweep_stds}", flush=True)
        gen_paths = [Path(p).resolve() for p in args.calibrate_gen]
        calibrate_sigma(
            clean,
            start_idx,
            args.seed,
            sweep_stds,
            gen_paths,
            args.hf_max_images,
        )
        if args.calibrate_only:
            return

    stds_to_write = sweep_stds if sweep_stds is not None else [args.noise_std]

    out_dir = None
    if sweep_stds is not None:
        if args.out_dir:
            out_dir = Path(args.out_dir).resolve()
        else:
            out_dir = Path(args.output).resolve().parent / "real_noise_refs"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"扫参输出目录: {out_dir}", flush=True)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = None

    for std in stds_to_write:
        noisy = add_mild_noise(clean, start_idx, std, args.seed, verbose=True)
        hf = laplacian_hf_energy(noisy, max_images=args.hf_max_images)
        print(f"  ref HF = {hf:.6f}", flush=True)

        mu = sigma = None
        if not args.no_stats:
            mu, sigma, model = compute_fid_stats(
                noisy, args.batch_size, device, args.dims, model=model
            )

        if out_dir is not None:
            out_path = out_dir / f"real_noise_std{fmt_std(std)}_tail{args.tail}.npz"
        else:
            out_path = Path(args.output).resolve()

        save_one(
            out_path,
            noisy,
            std,
            start_idx,
            args.tail,
            args.seed,
            mu,
            sigma,
            hf=hf,
        )
        if mu is not None:
            print(
                "可用 pytorch-fid / FID.py 作参考:\n"
                f"  python FID.py  # 或改路径指向 {out_path}",
                flush=True,
            )


if __name__ == "__main__":
    main()
