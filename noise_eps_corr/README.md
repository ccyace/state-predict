# 噪声预测校正（`noise_eps_corr`）

集中实现量化扩散模型的噪声预测校正：\(\varepsilon_{\mathrm{corr}}=\varepsilon_q+\gamma_t\Delta\varepsilon\)。

## 目录

| 路径 | 作用 |
|------|------|
| `learned_noise_corrector.py` | DeltaEpsNet、Loss、ckpt 读写 |
| `scripts/collect_fullstep_training_data.py` | 全步闭环轨迹采集 |
| `scripts/collect_late_training_data.py` | late-t 轨迹采集 |
| `scripts/train_corrector.py` | 训练校正器 |
| `scripts/run_w8a8_fullstep_pipeline.py` | W8A8 全步流水线 |
| `scripts/run_w4a8_fullstep_pipeline.py` | W4A8 全步流水线 |
| `scripts/run_w8a8_250step_pipeline.py` | W8A8 250-step 流水线 |
| `scripts/run_w4a8_250step_pipeline.py` | W4A8 250-step 流水线 |

## 常用命令（在仓库根目录执行）

```bash
# 端到端（W8A8）
python noise_eps_corr/scripts/run_w8a8_fullstep_pipeline.py

# 仅训练
python noise_eps_corr/scripts/train_corrector.py --data <traj.pt> --output_dir <out>

# 采样（共享入口）
python scripts/sample_diffusion_ddim.py ... --enable_learned_noise_corr --learned_corr_ckpt <ckpt>
```

## 剩余残差随机性 MVP

先在与采样设置一致的闭环轨迹上，按实际推理强度和范数裁剪统计校正后残差：

```bash
python noise_eps_corr/scripts/estimate_residual_stats.py \
  --data <closed_loop_traj.pt> --ckpt <ckpt_best.pt> \
  --t_cut 999 --alpha 0.5 --output <residual_stats.json>
```

实验 D 使用 `budget` 模式：按通道方差生成跨 timestep 的 AR(1) 残差，
通过 DDIM 的 epsilon-to-state 系数映射后，从 `eta` 的显式高斯噪声预算中逐通道扣除。
因此总状态方差保持不变。

```bash
# deterministic baseline / learned-correction baseline: 不传 residual 参数

# 实验 D；要求 eta > 0，建议扫描 eta=0.2,0.5,1.0
python scripts/sample_diffusion_ddim.py ... --eta 0.5 \
  --residual_stochastic_mode budget \
  --residual_stats_json <residual_stats.json> \
  --residual_stochastic_scale 0.5 \
  --residual_ar_clip 0.95
```

`budget` 要求统计 JSON 与推理使用完全相同的量化位宽、校正器、DDIM timestep
数量和 skip grid。当前只支持标准 generalized DDIM，不支持 float-t/joint-t 分支。

旧路径 `scripts/collect_*` / `scripts/train_learned_*` / `qdiff/learned_noise_corrector.py` 仍为兼容转发，新开发请用本目录。

## QIR 可行性实验

QIR 的第一阶段是 oracle 机制验证：FP 模型只用于同状态误差和终点参照，
结果不能直接视为可部署采样器性能。

```bash
# 1. 量化闭环轨迹上的同状态 Q/FP 输出
python noise_eps_corr/scripts/collect_fullstep_training_data.py \
  --num_trajectories 256 --timesteps 20 --skip_type quad \
  --output output/qir_feasibility/pairs_w8a8_n256_s20.pt

# 2. 因果逐通道 bias + AR(1) 创新分解
python noise_eps_corr/scripts/analyze_qir_innovation.py \
  --data output/qir_feasibility/pairs_w8a8_n256_s20.pt \
  --output output/qir_feasibility/innovation_w8a8_n256_s20.json

# 3. 单脉冲收缩、方向对照和延迟反相关回收
python noise_eps_corr/scripts/run_qir_single_pulse.py \
  --innovation_stats output/qir_feasibility/innovation_w8a8_n256_s20.json \
  --output output/qir_feasibility/pulse_w8a8_n64_s20.json \
  --num_trajectories 64 --timesteps 20 --pulse_fractions 0.5 \
  --directions gaussian,error,innovation,orthogonal \
  --strengths 0.05,0.1,0.2 --recovery_delays 0,4,8 \
  --recovery_mus 0,0.25,0.5
```

`paired_gain_mean > 0` 表示相对未加脉冲的量化轨迹更接近配对 FP 终点；
`terminal_residual_ratio_mean < 1` 表示脉冲在到达终点前发生了范数收缩。
