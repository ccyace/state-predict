"""
将图像目录整理为 img/ 子目录，并导出同级 .npz。

示例（处理 w4a8test）:
  python scripts/folder_to_npz.py \\
    --root C:/Users/ASUS/Desktop/ODE-scale/w4a8test

结果:
  w4a8test/
    img/          # 全部 PNG
    images.npz    # images: (N,H,W,3) uint8
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
try:
    from natsort import natsorted
except ImportError:
    import re

    def natsorted(values, key=lambda value: value):
        def natural_key(value):
            return [
                int(part) if part.isdigit() else part.lower()
                for part in re.split(r"(\d+)", str(key(value)))
            ]
        return sorted(values, key=natural_key)
from PIL import Image
from tqdm import tqdm

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def list_images(folder: Path) -> list[Path]:
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return natsorted(files, key=lambda p: p.name)


def move_images_into_img(root: Path) -> Path:
    """把 root 下散落的图像移入 root/img/；若已在 img/ 则直接返回。"""
    img_dir = root / "img"
    img_dir.mkdir(parents=True, exist_ok=True)

    top_files = list_images(root)
    if top_files:
        print(f"移动 {len(top_files)} 张图像 -> {img_dir}", flush=True)
        for src in tqdm(top_files, desc="move", mininterval=1.0):
            dst = img_dir / src.name
            if dst.exists():
                # 已存在同名则覆盖前先删目标，避免 Windows 报错
                dst.unlink()
            shutil.move(str(src), str(dst))
    else:
        print(f"根目录无散落图像，使用已有: {img_dir}", flush=True)

    n = len(list_images(img_dir))
    print(f"img/ 中共 {n} 张图", flush=True)
    if n == 0:
        raise RuntimeError(f"未找到任何图像: {root}")
    return img_dir


def images_to_npz(img_dir: Path, out_npz: Path) -> None:
    files = list_images(img_dir)
    print(f"导出 {len(files)} 张 -> {out_npz}", flush=True)
    arrays = []
    for path in tqdm(files, desc="pack", mininterval=1.0):
        with Image.open(path) as im:
            arrays.append(np.asarray(im.convert("RGB"), dtype=np.uint8))
    images = np.stack(arrays, axis=0)
    np.savez_compressed(out_npz, images=images)
    mb = out_npz.stat().st_size / (1024 * 1024)
    print(f"完成: shape={images.shape}, size={mb:.1f} MB", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="Move images into img/ and save sibling .npz")
    p.add_argument(
        "--root",
        type=str,
        default=r"C:\Users\ASUS\Desktop\ODE-scale\w4a8test",
        help="样本根目录（图像当前在此目录，或已在其 img/ 下）",
    )
    p.add_argument(
        "--npz_name",
        type=str,
        default="images.npz",
        help="保存在 root 下的 npz 文件名",
    )
    p.add_argument(
        "--no_move",
        action="store_true",
        help="不移动文件，仅从 root 或 root/img 读入并导出 npz",
    )
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    if not root.is_dir():
        raise SystemExit(f"目录不存在: {root}")

    if args.no_move:
        img_dir = root / "img" if (root / "img").is_dir() and list_images(root / "img") else root
    else:
        img_dir = move_images_into_img(root)

    out_npz = root / args.npz_name
    images_to_npz(img_dir, out_npz)
    print(f"结构:\n  {root}/img/\n  {out_npz}", flush=True)


if __name__ == "__main__":
    main()
