"""
Compute Inception Score (IS) for a folder of images using torch-fidelity.

Example (your run folder):
  python scripts/compute_is.py --samples "C:/Users/ASUS/Desktop/q-diffusion-master/output_cali_iters_a_7000/samples/2026-04-08-10-21-11/img"

Use only the first N images (natural sort):
  python scripts/compute_is.py --samples path/to/img --max_images 5000
"""

import argparse
from pathlib import Path

from natsort import natsorted


def main():
    parser = argparse.ArgumentParser(
        description="Inception Score (IS) via torch-fidelity (Inception-v3)."
    )
    parser.add_argument(
        "--samples",
        type=str,
        required=True,
        help="Directory of images (.png/.jpg/.jpeg)",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--no_cuda", action="store_true", help="Run on CPU")
    parser.add_argument(
        "--isc_splits",
        type=int,
        default=10,
        help="Number of splits for IS mean/std (torch-fidelity default is 10)",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        metavar="N",
        help="Use only first N images after natural sort (default: all)",
    )
    args = parser.parse_args()

    root = Path(args.samples).resolve()
    if not root.is_dir():
        raise SystemExit(f"Not a directory: {root}")

    from torch_fidelity import calculate_metrics
    from torch_fidelity.datasets import ImagesPathDataset
    from torch_fidelity.utils import glob_samples_paths

    if args.max_images is not None:
        if args.max_images <= 0:
            raise SystemExit("--max_images must be positive")
        files = glob_samples_paths(
            str(root),
            samples_find_deep=False,
            samples_find_ext="png,jpg,jpeg",
            samples_ext_lossy=None,
            verbose=False,
        )
        files = natsorted(files)[: args.max_images]
        if len(files) < args.max_images:
            raise SystemExit(
                f"Need {args.max_images} images but only found {len(files)} under {root}"
            )
        input1 = ImagesPathDataset(files)
        use_cache = False
    else:
        input1 = str(root)
        use_cache = True

    kw = dict(
        input1=input1,
        cuda=not args.no_cuda,
        batch_size=args.batch_size,
        isc=True,
        isc_splits=args.isc_splits,
        verbose=True,
        cache=use_cache,
    )
    if args.max_images is not None:
        print(f"Using {args.max_images} images (natural sort) from {root}")

    metrics = calculate_metrics(**kw)
    mean = metrics["inception_score_mean"]
    std = metrics["inception_score_std"]
    print("inception_score_mean (IS):", mean)
    print("inception_score_std:", std)


if __name__ == "__main__":
    main()
