"""
Compute FID between a folder of generated images and a reference using torch-fidelity.

Typical CIFAR-10 usage (compare to full training set, 50k images):
  python scripts/compute_fid.py --gen path/to/img --ref cifar10-train

Compare to official test split (10k):
  python scripts/compute_fid.py --gen path/to/img --ref cifar10-val

Reference can also be a directory of real PNG/JPG images:
  python scripts/compute_fid.py --gen path/to/fakes --ref path/to/reals

Use only the first N generated images (numeric order: 0.png … 9999.png):
  python scripts/compute_fid.py --gen path/to/img --ref cifar10-val --gen_max_images 10000
"""

import argparse
import re
from pathlib import Path

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


def main():
    parser = argparse.ArgumentParser(
        description="FID via torch-fidelity (inception-v3 features)."
    )
    parser.add_argument(
        "--gen",
        type=str,
        required=True,
        help="Directory of generated samples (e.g. .../samples/<run>/img)",
    )
    parser.add_argument(
        "--ref",
        type=str,
        default="cifar10-train",
        help='Registered name: cifar10-train (50k) or cifar10-val (10k); or path to a folder of real images',
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--no_cuda", action="store_true", help="Run on CPU")
    parser.add_argument(
        "--datasets_root",
        type=str,
        default=None,
        help="Cache root for torchvision CIFAR-10 when --ref is cifar10-* (default: torch default)",
    )
    parser.add_argument(
        "--no_download",
        action="store_true",
        help="Do not download CIFAR-10 (must already exist under datasets_root)",
    )
    parser.add_argument(
        "--gen_max_images",
        type=int,
        default=None,
        metavar="N",
        help="Use only the first N images after natural sort (e.g. 0.png…9999.png). Default: use all files in --gen",
    )
    args = parser.parse_args()

    gen = Path(args.gen).resolve()
    if not gen.is_dir():
        raise SystemExit(f"Not a directory: {gen}")

    from torch_fidelity import calculate_metrics
    from torch_fidelity.datasets import ImagesPathDataset
    from torch_fidelity.utils import glob_samples_paths

    ref_arg = args.ref
    ref_path = Path(ref_arg)
    if ref_path.exists() and ref_path.is_dir():
        input2 = str(ref_path.resolve())
    else:
        input2 = ref_arg

    if args.gen_max_images is not None:
        if args.gen_max_images <= 0:
            raise SystemExit("--gen_max_images must be positive")
        files = glob_samples_paths(
            str(gen),
            samples_find_deep=False,
            samples_find_ext="png,jpg,jpeg",
            samples_ext_lossy=None,
            verbose=False,
        )
        files = natsorted(files)[: args.gen_max_images]
        if len(files) < args.gen_max_images:
            raise SystemExit(
                f"Need {args.gen_max_images} images but only found {len(files)} under {gen}"
            )
        input1 = ImagesPathDataset(files)
        # Avoid reusing feature cache from a full-folder run with the same path
        use_cache = False
    else:
        input1 = str(gen)
        use_cache = True

    kw = dict(
        input1=input1,
        input2=input2,
        cuda=not args.no_cuda,
        batch_size=args.batch_size,
        fid=True,
        verbose=True,
        datasets_download=not args.no_download,
        cache=use_cache,
    )
    if args.datasets_root is not None:
        kw["datasets_root"] = args.datasets_root

    if args.gen_max_images is not None:
        print(f"Using {args.gen_max_images} generated images (natural sort) from {gen}")
    metrics = calculate_metrics(**kw)
    fid = metrics["frechet_inception_distance"]
    print("frechet_inception_distance (FID):", fid)


if __name__ == "__main__":
    main()
