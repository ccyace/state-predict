# PTQD 式方差预算吸收：eta=1 快速验证

## 目的

在现有 W8A8 CIFAR-10、dt 修正和 epsilon 均值修正基础上，验证 PTQD 式
Variance Schedule Calibration (VSC) 是否能通过扣减广义 DDIM 原有随机方差，
吸收均值校正后的不可预测残差。

本实验是单 seed、2k 图像的快速可行性实验，不作为最终统计显著性结论。

## 固定配置

- sampler：generalized DDIM，`eta=1.0`；
- 请求 100 steps，当前 quad 网格实际产生 96 个更新步；
- W8A8 CIFAR-10；
- dt refresh：8 个节点，`dt_eta=0.5`，`t_cutoff=300`；
- 原有 epsilon 均值修正头，strength `0.1`；
- seed：1234。

这里的 `eta=1` 是 generalized DDIM-100，不等同于完整相邻 1000 步的标准
DDPM。

## 代码修改

### `qdiff/ddim_helpers.py`

1. `ddim_update` 增加显式 `generator`，使各消融组共享相同初始噪声和逐步
   随机噪声；
2. 新增 `ddim_update_vsc`；
3. 用原始 eta 计算并保持 DDIM 的 `c2` 均值系数不变；
4. 只扣减独立随机项方差：

```
sigma_used^2 = sigma^2 - lambda * min(B^2 v_r(t), 0.9 sigma^2)
```

### `state_aware_temporal_joint/sample_50k.py`

- sparse dt refresh 支持 `eta>0`；
- 新增 `--vsc_stats`、`--vsc_absorb_strength` 和
  `--vsc_max_budget_fraction`；
- eta=0 时自动禁用 VSC。

### `state_aware_temporal_joint/collect_dt_corrected_residual_data.py`

- 支持指定 sampler eta；
- 支持在闭环更新前应用现有均值修正头；
- 保存实际部署修正后的残差 MSE：

```
r = epsilon_fp(x, t_nom) - epsilon_corrected
```

### `PTQD/estimate_time_variance.py`

- 计算各时间步残差 MSE 的均值和 10% 双侧截尾均值；
- 计算 time-only 和样本实际预算占用率；
- 最终零随机预算端点不计入占用率统计。

## 快速迁移检查

| 方法 | FID-1k |
|---|---:|
| dt-only | 32.664948 |
| dt + mean(0.1) | 32.842115 |

mean 分支轻微变差 0.177，但没有出现明显迁移失效。后续仍使用 mean 分支，
以符合“先校正可预测部分，再吸收不可预测部分”的方法定义。

## 方差预算审计

收集了 200 条闭环轨迹，共 19,200 个状态。

| 指标 | time-only | 使用样本真实残差 MSE |
|---|---:|---:|
| 预算占用率中位数 | 4.668e-5 | 4.648e-5 |
| 预算占用率 p95 | 0.01009 | 0.01136 |
| 占用率大于 1 | 0% | 0% |

结论：在 eta=1 下预算完全充足，但剩余误差消耗的随机预算非常小。大多数
时间步的 time-only 吸收量远低于 1%，因此预期 VSC 对最终分布影响有限。

## 最终 FID-2k

| 方法 | FID-2k | 相对 mean-only |
|---|---:|---:|
| dt + mean | **17.258963** | 0 |
| dt + mean + VSC 0.5 | 17.260425 | +0.001461 |
| dt + mean + VSC 1.0 | 17.359122 | +0.100159 |

配对像素检查：

| 方法 | 相对 mean-only 的平均绝对像素差，0–255 |
|---|---:|
| VSC 0.5 | 1.00692 |
| VSC 1.0 | 0.98912 |

两组各 2,000 张图像全部发生变化，因此 VSC 确实执行，并非查表或代码路径
失效。

## 当前结论

1. PTQD 式方差吸收在数值上可行，没有预算不足或截断问题；
2. time-only VSC 0.5 与基线基本持平，没有显示正收益；
3. 完全吸收使 FID-2k 变差约 0.10；
4. 当前不能声称 PTQD VSC 有效，只能说明实现可运行且预算充足；
5. 结果符合此前残差分析：按时间平均的标量方差忽略了强烈的状态异方差、
   空间结构和跨步相关，完全局部吸收可能过强；
6. 如果继续该方向，下一项最有价值的实验是 sample-wise VSC，并限制在高残差
   状态或预算占用较高的时间区间，而不是扩大 time-only 50k 实验。

## 主要命令

### 64 张冒烟测试

```bash
python state_aware_temporal_joint/sample_50k.py \
  --disable_adapter --disable_corrector --dt_only_infer --dt_mode refresh \
  --joint_dt_ckpt state_aware_temporal_joint/experiments/w8a8_cifar10_ddim100_dt_simple_v1/checkpoints/ckpt_best.pt \
  --time_residual_ckpt state_aware_temporal_joint/experiments/w8a8_cifar10_ddim100_dt_residual_v1/checkpoints/ckpt_best.pt \
  --time_residual_strength 0.1 --eta 1.0 --timesteps 100 --skip_type quad \
  --dt_eta 0.5 --dt_carry_max 20 --t_cutoff 300 --dt_refresh_n 8 \
  --max_images 64 --batch_size 64 --seed 1234 \
  --output_dir PTQD/smoke_eta1_64
```

### 收集 200 条闭环轨迹

```bash
python state_aware_temporal_joint/collect_dt_corrected_residual_data.py \
  --output PTQD/traj_eta1_dt_mean_s01_n200.pt \
  --dt_ckpt state_aware_temporal_joint/experiments/w8a8_cifar10_ddim100_dt_simple_v1/checkpoints/ckpt_best.pt \
  --time_residual_ckpt state_aware_temporal_joint/experiments/w8a8_cifar10_ddim100_dt_residual_v1/checkpoints/ckpt_best.pt \
  --time_residual_strength 0.1 --num_trajectories 200 --batch_size 64 \
  --timesteps 100 --skip_type quad --eta 1.0 --dt_eta 0.5 \
  --dt_max 20 --t_cutoff 300 --n_refresh 8 --seed 1234
```

### 估计方差和预算

```bash
python PTQD/estimate_time_variance.py \
  --data PTQD/traj_eta1_dt_mean_s01_n200.pt \
  --output_pt PTQD/vsc_time_stats_eta1.pt \
  --output_json PTQD/vsc_time_stats_eta1.json --eta 1.0
```

### VSC 采样关键参数

```bash
--vsc_stats PTQD/vsc_time_stats_eta1.pt \
--vsc_absorb_strength 0.5 \
--vsc_max_budget_fraction 0.9
```

将 strength 改为 `1.0` 得到完全吸收组；不传 VSC 参数得到 mean-only 组。

### FID

```bash
python -m pytorch_fid new_real_images/cifar10_python.npz \
  PTQD/final_mean_only_2k --device cuda:0
python -m pytorch_fid new_real_images/cifar10_python.npz \
  PTQD/final_vsc_05_2k --device cuda:0
python -m pytorch_fid new_real_images/cifar10_python.npz \
  PTQD/final_vsc_10_2k --device cuda:0
```

## 目录内容

- `results.json`：汇总数值；
- `vsc_time_stats_eta1.pt/json`：time-only 方差和预算；
- `traj_eta1_dt_mean_s01_n200.pt`：200 条闭环轨迹；
- `smoke_eta1_64/`：冒烟测试；
- `migration_*_1k/`：现有头迁移比较；
- `final_mean_only_2k/`：最终对照；
- `final_vsc_05_2k/`、`final_vsc_10_2k/`：VSC 两组图像；
- `logs/`：采样、统计和 FID 日志。
