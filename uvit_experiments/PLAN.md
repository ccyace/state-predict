# U-ViT CIFAR 最小实验计划

## 目标

在 **U-ViT-S/2 @ CIFAR-10** 上验证「预测矫正 + tvar」可脱离 UNet，与现有 CIFAR DDIM UNet 实验同协议对比。

## 协议（与 W4A8 vsc_tvar 对齐）

| 项 | 设定 |
|----|------|
| 模型 | U-ViT-S/2，无条件，像素 32×32，ε-prediction |
| 噪声 schedule | linear β: 1e-4 → 0.02，1000 步 |
| 采样 | DDIM 100 步 quad，η=1 |
| 评测 | 50k 图，FID 参考 `new_real_images/real47500_vsc2500_fid_stats.npz` |

## 阶段

### Phase 0 — FP baseline（当前执行）

1. 下载 `cifar10_uvit_small.pth`（官方 FID≈3.11，DPM-Solver 协议；本阶段用 DDIM 100 η=1 自建 FP 数字）
2. `sample_fp_ddim.py` 生成 50k
3. 计算 FID / IS

### Phase 1 — W4A8 naive PTQ baseline

1. 为 U-ViT 写 `QuantLinear` / `QuantAttention` 包装（或复用 Q-Diffusion BRECQ 思路）
2. PTQ 校准 → `uvit_w4a8_ckpt.pth`
3. 50k 采样 + FID（**无公开 baseline，需自建**）

### Phase 2 — 闭环轨迹 + 校正头

1. `collect_dt_corrected_residual_data.py` 适配 U-ViT（`model(x,t)` 接口）
2. 在 W4A8 轨迹上训练 joint_dt + mean head（3 通道，无需改头）
3. `estimate_time_variance.py` → tvar 表

### Phase 3 — vsc_tvar 50k

1. 接入 `ddim_update_vsc`
2. 50k 采样 + FID/IS，对比 Phase 0/1

## 目录

```
uvit_experiments/
  checkpoints/cifar10_uvit_small.pth   # 预训练权重
  sample_fp_ddim.py                  # Phase 0 采样
  run_phase0_fp_baseline.sh
  download_ckpt.sh
  outputs/phase0_fp_ddim100_eta1_50k/
```

## 参考数字

| 来源 | FID | 备注 |
|------|-----|------|
| U-ViT 官方 FP | 3.11 | DPM-Solver，不可直接对比 |
| Q-Diffusion W4A8 UNet | 4.93 | 同 DDIM 100，不同 backbone |
| 本实验 U-ViT FP | TBD | DDIM 100 η=1 |
| 本实验 U-ViT W4A8 | TBD | 自建 baseline |
