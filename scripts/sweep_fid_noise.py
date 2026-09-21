"""
在「带噪参考」网格上，对多个生成集（如 W4A8 / W8A8）扫 FID。

流程:
  1) 可选：现场生成参考 npz（--make_refs）
  2) 可选：HF 标定建议 σ*（--calibrate）
  3) 每个生成集只提一次 Inception 特征（缓存 fid_stats.npz）
  4) 对每个 (noise_std, gen) 用参考 mu/sigma 算 Frechet，写 CSV

示例（一键：扫参考 + HF 标定 + FID）:
  python scripts/sweep_fid_noise.py ^
    --real_dir C:/Users/ASUS/Desktop/ODE-scale/real_images ^
    --ref_dir C:/Users/ASUS/Desktop/ODE-scale/real_noise_refs ^
    --sweep_std 0,0.01,0.02,0.03,0.05,0.08 ^
    --tail 10000 ^
    --make_refs --calibrate ^
    --gens C:/Users/ASUS/Desktop/ODE-scale/w4a8test/images.npz ^
           C:/Users/ASUS/Desktop/ODE-scale/w8a8test
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = Path(__file__).resolve().parent
for p in (str(_ROOT), str(_SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from FID import frechet_distance  # noqa: E402
from make_noisy_real_npz import (  # noqa: E402
    add_mild_noise,
    calibrate_sigma,
    compute_fid_stats,
    fmt_std,
    laplacian_hf_energy,
    load_all_uint8,
    load_images_any,
    parse_sweep_stds,
    save_one,
)


def parse_args():
    p = argparse.ArgumentParser(description="Sweep FID over noisy real references")
    p.add_argument(
        "--real_dir",
        type=str,
        default=r"C:\Users\ASUS\Desktop\ODE-scale\real_images",
    )
    p.add_argument(
        "--ref_dir",
        type=str,
        default=r"C:\Users\ASUS\Desktop\ODE-scale\real_noise_refs",
    )
    p.add_argument(
        "--gens",
        type=str,
        nargs="+",
        default=[
            r"C:\Users\ASUS\Desktop\ODE-scale\w4a8test\images.npz",
            r"C:\Users\ASUS\Desktop\ODE-scale\w8a8test",
        ],
    )
    p.add_argument("--sweep_std", type=str, default="0,0.01,0.02,0.03,0.05,0.08")
    p.add_argument("--tail", type=int, default=10000)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--make_refs", action="store_true", help="缺少参考 npz 时现场生成")
    p.add_argument("--calibrate", action="store_true", help="打印 HF 标定建议 σ*")
    p.add_argument("--hf_max_images", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--dims", type=int, default=2048, choices=[64, 192, 768, 2048])
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--csv", type=str, default=None)
    p.add_argument(
        "--no_save_gen_stats",
        action="store_true",
        help="不把生成集 mu/sigma 缓存到 fid_stats.npz",
    )
    return p.parse_args()


def resolve_gen_path(path: Path) -> Path:
    path = path.resolve()
    if path.is_file():
        return path
    if path.is_dir():
        npz = path / "images.npz"
        return npz if npz.is_file() else path
    raise FileNotFoundError(path)


def gen_label(path: Path) -> str:
    if path.suffix.lower() == ".npz":
        return path.parent.name if path.stem == "images" else path.stem
    return path.name


def load_ref_stats(path: Path) -> tuple[float, np.ndarray, np.ndarray, float | None]:
    with np.load(path, allow_pickle=False) as npz:
        if "mu" not in npz.files or "sigma" not in npz.files:
            raise ValueError(f"参考缺少 mu/sigma: {path}")
        std = float(npz["noise_std"]) if "noise_std" in npz.files else float("nan")
        hf = float(npz["hf_laplacian"]) if "hf_laplacian" in npz.files else None
        return std, npz["mu"], npz["sigma"], hf


def stats_from_gen(
    path: Path,
    batch_size: int,
    device: torch.device,
    dims: int,
    model: torch.nn.Module | None,
    save_stats: bool,
) -> tuple[np.ndarray, np.ndarray, torch.nn.Module | None]:
    stats_path = path.parent / "fid_stats.npz" if path.is_file() else path / "fid_stats.npz"
    if stats_path.is_file():
        with np.load(stats_path, allow_pickle=False) as npz:
            if "mu" in npz.files and "sigma" in npz.files:
                print(f"  复用生成统计量: {stats_path}", flush=True)
                return npz["mu"], npz["sigma"], model

    images = load_images_any(path)
    print(f"  提取特征: {path} shape={images.shape}", flush=True)
    mu, sigma, model = compute_fid_stats(images, batch_size, device, dims, model=model)
    if save_stats:
        out = path.parent / "fid_stats.npz" if path.is_file() else path / "fid_stats.npz"
        np.savez_compressed(out, mu=mu, sigma=sigma)
        print(f"  已保存 {out}", flush=True)
    return mu, sigma, model


def ensure_refs(
    real_dir: Path,
    ref_dir: Path,
    stds: list[float],
    tail: int,
    seed: int,
    batch_size: int,
    device: torch.device,
    dims: int,
    hf_max_images: int,
) -> list[Path]:
    ref_dir.mkdir(parents=True, exist_ok=True)
    clean = load_all_uint8(real_dir)
    n = clean.shape[0]
    if n < tail:
        raise SystemExit(f"图像数 {n} < tail={tail}")
    start_idx = n - tail
    model = None
    paths = []
    for std in stds:
        out = ref_dir / f"real_noise_std{fmt_std(std)}_tail{tail}.npz"
        if out.is_file():
            with np.load(out, allow_pickle=False) as npz:
                ok = "mu" in npz.files and "sigma" in npz.files
            if ok:
                print(f"已存在参考: {out.name}", flush=True)
                paths.append(out)
                continue
            print(f"参考缺 mu/sigma，重算: {out.name}", flush=True)

        noisy = add_mild_noise(clean, start_idx, std, seed, verbose=True)
        hf = laplacian_hf_energy(noisy, max_images=hf_max_images)
        mu, sigma, model = compute_fid_stats(
            noisy, batch_size, device, dims, model=model
        )
        save_one(out, noisy, std, start_idx, tail, seed, mu, sigma, hf=hf)
        paths.append(out)
    return paths


def list_ref_npz(ref_dir: Path, tail: int | None = None) -> list[Path]:
    files = sorted(ref_dir.glob("real_noise_std*_tail*.npz"))
    if tail is not None:
        files = [f for f in files if f"tail{tail}" in f.name]
    return files


def main():
    args = parse_args()
    ref_dir = Path(args.ref_dir).resolve()
    stds = parse_sweep_stds(args.sweep_std)
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    gen_paths = [resolve_gen_path(Path(g)) for g in args.gens]

    if args.calibrate:
        real_dir = Path(args.real_dir).resolve()
        clean = load_all_uint8(real_dir)
        if clean.shape[0] < args.tail:
            raise SystemExit("real images 不足")
        start_idx = clean.shape[0] - args.tail
        calibrate_sigma(
            clean,
            start_idx,
            args.seed,
            stds,
            gen_paths,
            args.hf_max_images,
        )

    if args.make_refs:
        ref_paths = ensure_refs(
            Path(args.real_dir).resolve(),
            ref_dir,
            stds,
            args.tail,
            args.seed,
            args.batch_size,
            device,
            args.dims,
            args.hf_max_images,
        )
    else:
        ref_paths = list_ref_npz(ref_dir, tail=args.tail)
        if not ref_paths:
            raise SystemExit(
                f"{ref_dir} 下没有 real_noise_std*_tail{args.tail}.npz，"
                f"请加 --make_refs 或先运行 make_noisy_real_npz.py --sweep_std ..."
            )

    refs: list[tuple[float, Path, np.ndarray, np.ndarray, float | None]] = []
    for rp in ref_paths:
        std, mu, sigma, hf = load_ref_stats(rp)
        if np.isnan(std):
            m = re.search(r"std([0-9p]+)_", rp.name)
            std = float(m.group(1).replace("p", ".")) if m else float("nan")
        refs.append((std, rp, mu, sigma, hf))
    refs.sort(key=lambda t: t[0])

    print("\n=== 生成集特征 ===", flush=True)
    model = None
    gen_stats: list[tuple[str, Path, np.ndarray, np.ndarray]] = []
    save_gen = not args.no_save_gen_stats
    for gp in gen_paths:
        label = gen_label(gp)
        print(f"[{label}] {gp}", flush=True)
        mu_g, sigma_g, model = stats_from_gen(
            gp, args.batch_size, device, args.dims, model, save_stats=save_gen
        )
        gen_stats.append((label, gp, mu_g, sigma_g))

    print("\n=== FID 扫参表 ===", flush=True)
    labels = [g[0] for g in gen_stats]
    header = ["noise_std", "ref_hf"] + [f"FID_{lb}" for lb in labels]
    if len(labels) >= 2:
        header.append(f"d({labels[0]}-{labels[1]})")
    print("  ".join(f"{h:>16}" for h in header), flush=True)

    rows_out = []
    for std, rp, mu_r, sigma_r, hf in refs:
        fids = [
            frechet_distance(mu_r, sigma_r, mu_g, sigma_g)
            for _label, _gp, mu_g, sigma_g in gen_stats
        ]
        hf_s = "" if hf is None else f"{hf:.4f}"
        row = [f"{std:.4f}", hf_s] + [f"{v:.4f}" for v in fids]
        if len(fids) >= 2:
            row.append(f"{fids[0] - fids[1]:.4f}")
        print("  ".join(f"{c:>16}" for c in row), flush=True)
        rec = {"noise_std": std, "ref_path": str(rp), "ref_hf": hf}
        for lb, v in zip(labels, fids):
            rec[f"FID_{lb}"] = v
        if len(fids) >= 2:
            rec["FID_diff"] = fids[0] - fids[1]
        rows_out.append(rec)

    csv_path = Path(args.csv).resolve() if args.csv else (ref_dir / "fid_sweep.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["noise_std", "ref_hf", "ref_path"] + [f"FID_{lb}" for lb in labels]
    if len(labels) >= 2:
        fieldnames.append("FID_diff")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rec in rows_out:
            w.writerow(rec)
    print(f"\n已写入 CSV: {csv_path}", flush=True)

    print("\n提示（消融用，非主结论）: 各生成集在网格上 FID 最低的 noise_std:", flush=True)
    for lb in labels:
        best = min(rows_out, key=lambda r: r[f"FID_{lb}"])
        print(
            f"  {lb}: σ={best['noise_std']}  FID={best[f'FID_{lb}']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
