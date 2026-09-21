"""Overlay W4A8 and W8A8 latent trajectory L2 curves on one figure."""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def load_traj(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return np.array(data["timesteps"]), np.array(data["traj_l2_mean"])


def plot_traj_compare(series, save_path):
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))

    for item in series:
        timesteps, traj_l2 = item["timesteps"], item["traj_l2"]
        ax.plot(
            timesteps,
            traj_l2,
            color=item["color"],
            linewidth=1.8,
            label=item["label"],
        )

    ax.set_xlabel("Diffusion timestep $t$ ($T \\rightarrow 0$)")
    ax.set_ylabel("$\\|x_q - x_{fp}\\|_2$ (per-sample mean)")
    ax.set_title("Latent trajectory L2 deviation")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    ax.invert_xaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {save_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Compare W4A8 vs W8A8 trajectory L2 on one plot.")
    parser.add_argument(
        "--w4a8_json",
        type=str,
        default="outputs/quant_noise_error_w4a8.json",
    )
    parser.add_argument(
        "--w8a8_json",
        type=str,
        default="outputs/quant_noise_error_w8a8.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/quant_traj_l2_w4a8_w8a8.png",
    )
    args = parser.parse_args()

    def resolve(path):
        return path if os.path.isabs(path) else os.path.join(ROOT, path)

    w4_json = resolve(args.w4a8_json)
    w8_json = resolve(args.w8a8_json)
    out_png = resolve(args.output)

    t4, traj4 = load_traj(w4_json)
    t8, traj8 = load_traj(w8_json)

    plot_traj_compare(
        [
            {"timesteps": t4, "traj_l2": traj4, "label": "W4A8", "color": "#d62728"},
            {"timesteps": t8, "traj_l2": traj8, "label": "W8A8", "color": "#ff7f0e"},
        ],
        out_png,
    )


if __name__ == "__main__":
    main()
