# eta=1 专用均值头与 sample-wise VSC 前置验证

## 实验目的

上一轮 time-only PTQD VSC 没有改善 FID。按照预设的最合理下一步，本实验先
排除旧 eta=0 均值头在 eta=1 随机轨迹上的分布失配：重新收集 eta=1 dt-only
闭环数据、训练专用均值头，并通过闭环 FID 决定是否继续 sample-wise VSC。

预先设定的停止条件是：只有新均值头优于 dt-only，才在其剩余残差上训练
sample-wise 方差头。

## 数据与训练

- generalized DDIM，eta=1；
- W8A8 CIFAR-10；
- 请求 100 steps，quad 网格实际 96 个更新；
- dt refresh 8 次，dt shift scale 0.5；
- 500 条 dt-only 闭环轨迹，共 48,000 个状态；
- 训练 20 epochs，batch size 128，学习率 0.001；
- 训练目标仍为当前 epsilon MSE + 0.1 cosine。

最佳 checkpoint 位于 `checkpoints/ckpt_best.pt`。第 20 轮结果：

- validation loss：0.00246140；
- validation MSE：0.00229594；
- validation cosine term：0.00165459。

离线指标表明新网络能够显著预测当前 eta=1 状态上的量化误差。

## 闭环 FID-1k

所有组使用相同初始噪声、相同逐步随机噪声和 seed 1234。

| 方法 | FID-1k | 相对 dt-only |
|---|---:|---:|
| dt-only | **32.664948** | 0 |
| 旧 eta=0 mean，strength 0.1 | 32.842115 | +0.177167 |
| 新 eta=1 mean，strength 0.1 | 32.905703 | +0.240755 |
| 新 eta=1 mean，strength 1.0 | 35.026213 | +2.361265 |

## 结论与停止决定

新 eta=1 均值头虽然将离线验证残差 MSE 降至 0.002296，但两个推理强度都使
FID 变差，而且完全强度明显过校正。这说明当前主要问题不再是 eta=0 到 eta=1
的训练分布迁移，而是训练目标与闭环采样目标不一致：

```
降低当前步 epsilon MSE
!=
降低经过 DDIM 更新和后续 UNet 传播后的轨迹误差
```

因此按预设 gate 停止，没有训练 sample-wise 方差头，也没有继续 sample-wise
VSC 采样。若继续做方差头，其效果会与有害的均值修正纠缠，无法判断 PTQD
方差吸收本身是否有效。

## 下一步应改什么

在继续 PTQD 前，应先把均值头训练目标改成采样一致的状态增量/短轨迹目标：

1. 当前步仍预测 epsilon 偏差；
2. 用真实 generalized-DDIM eta=1 更新把预测映射到下一状态；
3. 对齐预测下一状态与 FP 参考下一状态；
4. 可再展开 2–3 步，对齐短轨迹终点；
5. 先证明新目标训练的均值头在闭环 FID 上优于 dt-only；
6. 然后才训练 sample-wise 方差头并测试 VSC。

最简两项损失可写成：

```
L = L_eps + lambda_x * ||x_next_pred - x_next_fp||^2
```

其中 `x_next_pred` 和 `x_next_fp` 必须使用相同 eta、相同显式高斯噪声以及相同
DDIM 系数，使差异只来自 epsilon 修正。

## 复现命令

### 收集训练轨迹

```bash
python state_aware_temporal_joint/collect_dt_corrected_residual_data.py \
  --output PTQD/eta1_mean_samplewise_v2/traj_eta1_dt_only_n500.pt \
  --dt_ckpt state_aware_temporal_joint/experiments/w8a8_cifar10_ddim100_dt_simple_v1/checkpoints/ckpt_best.pt \
  --num_trajectories 500 --batch_size 64 --timesteps 100 --skip_type quad \
  --eta 1.0 --dt_eta 0.5 --dt_max 20 --t_cutoff 300 --n_refresh 8 \
  --seed 1234
```

### 训练 eta=1 均值头

```bash
python state_aware_temporal_joint/train_dt_corrected_residual.py \
  --data PTQD/eta1_mean_samplewise_v2/traj_eta1_dt_only_n500.pt \
  --output_dir PTQD/eta1_mean_samplewise_v2/checkpoints \
  --epochs 20 --batch_size 128 --lr 0.001
```

### 采样

采样公共设置与上一层 `PTQD/README.md` 相同，将均值 checkpoint 改为：

```bash
--time_residual_ckpt PTQD/eta1_mean_samplewise_v2/checkpoints/ckpt_best.pt
```

分别使用：

```bash
--time_residual_strength 0.1
--time_residual_strength 1.0
```

输出、完整训练历史和日志均保存在本目录。
