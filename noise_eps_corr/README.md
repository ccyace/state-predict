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

旧路径 `scripts/collect_*` / `scripts/train_learned_*` / `qdiff/learned_noise_corrector.py` 仍为兼容转发，新开发请用本目录。
