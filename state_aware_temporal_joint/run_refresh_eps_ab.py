#!/usr/bin/env python
"""A/B experiment: refresh+old corrector vs refresh-CL finetuned ε corrector (10k FID).

Stages:
  A) sample 10k with refresh δt + old ε corrector → FID
  B) closed-loop collect under that policy → finetune ε corrector → sample 10k → FID
  Compare and write summary JSON/MD.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
REAL = os.path.join(ROOT, "new_real_images/cifar10_python.npz")
SAMPLE = "state_aware_temporal_joint/sample_50k.py"
COLLECT = "state_aware_temporal_joint/collect_refresh_cl_traj.py"
TRAIN = "noise_eps_corr/scripts/train_corrector.py"


def run(name: str, cmd: list[str], log_path: str) -> None:
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    print(f"\n{'=' * 72}\n[{name}] {datetime.now().isoformat()}\n  {' '.join(cmd)}\n  log={log_path}\n{'=' * 72}", flush=True)
    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write(f"CMD: {' '.join(cmd)}\n\n")
        logf.flush()
        proc = subprocess.Popen(
            cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            logf.write(line)
            logf.flush()
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"{name} failed with exit={rc}; see {log_path}")


def count_pngs(folder: str) -> int:
    if not os.path.isdir(folder):
        return 0
    return sum(1 for n in os.listdir(folder) if n.endswith(".png"))


def compute_fid(gen_dir: str, out_txt: str, device: str = "cuda:0") -> float:
    cmd = [PY, "-m", "pytorch_fid", REAL, gen_dir, "--device", device]
    run("fid", cmd, out_txt)
    text = open(out_txt, encoding="utf-8").read()
    m = re.search(r"FID:\s*([0-9.eE+-]+)", text)
    if not m:
        raise RuntimeError(f"FID parse failed: {out_txt}")
    return float(m.group(1))


def sample_cmd(args, output_dir: str, corrector_ckpt: str) -> list[str]:
    return [
        PY, SAMPLE,
        "--disable_adapter",
        "--allow_joint_plus_corrector",
        "--dt_only_infer",
        "--dt_mode", "refresh",
        "--joint_dt_ckpt", args.joint_dt_ckpt,
        "--corrector_ckpt", corrector_ckpt,
        "--dt_eta", str(args.dt_eta),
        "--dt_carry_max", str(args.dt_carry_max),
        "--t_cutoff", str(args.t_cutoff),
        "--dt_refresh_n", str(args.dt_refresh_n),
        "--max_images", str(args.max_images),
        "--batch_size", str(args.batch_size),
        "--seed", str(args.seed),
        "--output_dir", output_dir,
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--work_dir",
        default="state_aware_temporal_joint/output/refresh_eps_ab",
    )
    p.add_argument(
        "--joint_dt_ckpt",
        default="state_aware_temporal_joint/runs/dt_eps_oracle_w8a8/ckpt_best.pt",
    )
    p.add_argument(
        "--old_corrector_ckpt",
        default="noise_eps_corr/w8a8_corr_ckpt_best.pt",
    )
    p.add_argument("--dt_eta", type=float, default=0.5)
    p.add_argument("--dt_carry_max", type=float, default=20.0)
    p.add_argument("--t_cutoff", type=int, default=300)
    p.add_argument("--dt_refresh_n", type=int, default=8)
    p.add_argument("--max_images", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--num_trajectories", type=int, default=2000)
    p.add_argument("--collect_batch_size", type=int, default=64)
    p.add_argument("--ft_epochs", type=int, default=20)
    p.add_argument("--ft_batch_size", type=int, default=128)
    p.add_argument("--ft_lr", type=float, default=3e-4)
    p.add_argument("--skip_a_sample", action="store_true")
    p.add_argument("--skip_a_fid", action="store_true")
    p.add_argument("--skip_collect", action="store_true")
    p.add_argument("--skip_train", action="store_true")
    p.add_argument("--skip_b_sample", action="store_true")
    p.add_argument("--skip_b_fid", action="store_true")
    p.add_argument("--fid_device", default="cuda:0")
    args = p.parse_args()

    work = os.path.join(ROOT, args.work_dir) if not os.path.isabs(args.work_dir) else args.work_dir
    os.makedirs(work, exist_ok=True)
    dir_a = os.path.join(work, "A_refresh_old_corr_10k")
    dir_b = os.path.join(work, "B_refresh_ft_corr_10k")
    traj = os.path.join(work, "traj_refresh_cl.pt")
    ft_dir = os.path.join(work, "eps_ft_refresh_cl")
    ft_ckpt = os.path.join(ft_dir, "ckpt_best.pt")
    summary_path = os.path.join(work, "compare_summary.json")
    started = time.time()
    results = {
        "started": datetime.now().isoformat(),
        "args": vars(args),
        "A": {},
        "B": {},
    }

    # ---- A sample ----
    if not args.skip_a_sample:
        if count_pngs(dir_a) < args.max_images:
            run("A_sample", sample_cmd(args, dir_a, args.old_corrector_ckpt),
                os.path.join(work, "A_sample.log"))
        else:
            print(f"[A_sample] resume: already have {count_pngs(dir_a)} pngs", flush=True)
    results["A"]["images"] = count_pngs(dir_a)
    results["A"]["output_dir"] = dir_a
    results["A"]["corrector"] = args.old_corrector_ckpt

    if not args.skip_a_fid:
        fid_a = compute_fid(dir_a, os.path.join(work, "A_fid.txt"), args.fid_device)
        results["A"]["fid"] = fid_a
        print(f"[A] FID={fid_a}", flush=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    # ---- collect ----
    if not args.skip_collect:
        if not os.path.isfile(traj):
            run(
                "collect_refresh_cl",
                [
                    PY, COLLECT,
                    "--joint_dt_ckpt", args.joint_dt_ckpt,
                    "--corrector_ckpt", args.old_corrector_ckpt,
                    "--output", traj,
                    "--num_trajectories", str(args.num_trajectories),
                    "--batch_size", str(args.collect_batch_size),
                    "--dt_eta", str(args.dt_eta),
                    "--dt_carry_max", str(args.dt_carry_max),
                    "--t_cutoff", str(args.t_cutoff),
                    "--dt_refresh_n", str(args.dt_refresh_n),
                    "--seed", str(args.seed),
                ],
                os.path.join(work, "collect.log"),
            )
        else:
            print(f"[collect] resume: {traj} exists", flush=True)
    results["traj"] = traj

    # ---- finetune ε ----
    if not args.skip_train:
        if not os.path.isfile(ft_ckpt):
            run(
                "finetune_eps",
                [
                    PY, TRAIN,
                    "--data", traj,
                    "--output_dir", ft_dir,
                    "--init_ckpt", args.old_corrector_ckpt,
                    "--epochs", str(args.ft_epochs),
                    "--batch_size", str(args.ft_batch_size),
                    "--lr", str(args.ft_lr),
                    "--t_cut", "999",
                    "--alpha", "0.5",
                    "--mmap",
                ],
                os.path.join(work, "train_ft.log"),
            )
        else:
            print(f"[finetune] resume: {ft_ckpt} exists", flush=True)
    results["B"]["corrector"] = ft_ckpt

    # ---- B sample ----
    if not args.skip_b_sample:
        if count_pngs(dir_b) < args.max_images:
            run("B_sample", sample_cmd(args, dir_b, ft_ckpt),
                os.path.join(work, "B_sample.log"))
        else:
            print(f"[B_sample] resume: already have {count_pngs(dir_b)} pngs", flush=True)
    results["B"]["images"] = count_pngs(dir_b)
    results["B"]["output_dir"] = dir_b

    if not args.skip_b_fid:
        fid_b = compute_fid(dir_b, os.path.join(work, "B_fid.txt"), args.fid_device)
        results["B"]["fid"] = fid_b
        print(f"[B] FID={fid_b}", flush=True)

    results["finished"] = datetime.now().isoformat()
    results["elapsed_hours"] = (time.time() - started) / 3600.0
    if "fid" in results["A"] and "fid" in results["B"]:
        results["delta_fid_B_minus_A"] = results["B"]["fid"] - results["A"]["fid"]
        results["winner"] = "B_finetuned" if results["B"]["fid"] < results["A"]["fid"] else "A_old_corrector"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    md = os.path.join(work, "COMPARE.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("# refresh + ε corrector A/B\n\n")
        f.write(f"- A (old corrector): FID = {results['A'].get('fid', 'n/a')}\n")
        f.write(f"- B (finetuned on refresh-CL): FID = {results['B'].get('fid', 'n/a')}\n")
        if "delta_fid_B_minus_A" in results:
            f.write(f"- Δ(B−A) = {results['delta_fid_B_minus_A']:.4f}\n")
            f.write(f"- winner = **{results['winner']}**\n")
        f.write(f"\nSee `{summary_path}`\n")
    print(f"\nDone. Summary -> {summary_path}\n{open(md, encoding='utf-8').read()}", flush=True)


if __name__ == "__main__":
    main()
