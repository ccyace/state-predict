# 剩余残差方差的状态依赖核心消融

## 问题

检验均值校正后的剩余残差采用

```text
Sigma_u(c_t) = v(c_t) I
```

时，`v` 是否确实需要依赖当前状态，还是第一版使用仅随时间变化的
`Sigma_u(t) = v_t I` 已经足够。

逐样本监督量为

```text
y_t = log(mean((epsilon_fp - epsilon_q - mean_net(c_t))^2)).
```

它对应各向同性协方差的每维平均方差，即 `tr(Sigma)/D`。状态特征只使用推理时
可见的 `x_t`、量化噪声预测、名义/修正时间和 refresh 标志，不使用 FP 输出或
残差本身。

## 公平比较

- 校准集：`traj_id % 10 == 0`，50 条轨迹、4,800 个状态；
- 测试集：`traj_id % 10 == 5`，50 条轨迹、4,800 个状态；
- time-only：校准集上每个名义时间步的 log 方差查表；
- time+state：保留完全相同的时间查表，只用低容量岭回归预测时间基线的剩余误差；
- 置信区间：以完整轨迹为单位进行 2,000 次 bootstrap。

这种嵌套设计保证 `Delta R2` 只衡量时间步之外的状态信息，而不是模型拟合时间曲线
能力的差异。

## 结果

| 模型 | 测试 log-variance R2 | MAE | RMSE |
|---|---:|---:|---:|
| 仅时间步 | **0.965376** | **0.178689** | **0.255018** |
| 时间步 + 状态 | 0.961852 | 0.184313 | 0.267681 |

- `Delta R2 = -0.003524`；
- 轨迹 bootstrap 95% CI：`[-0.007801, 0.001053]`；
- `P(Delta R2 > 0) = 0.0665`。

因此，在当前数据、当前均值头和这些低阶可观测状态统计下，没有证据表明状态条件
能够提供超越时间步的可泛化方差预测。此前“联合特征可预测方差”的较高 R2 主要可由
时间步解释，不能单独作为状态依赖的证据。

## 对第一版方差模型的含义

当前最有证据支持、也最容易实现的第一版是

```text
r_t | t approximately follows N(0, v_t I),
v_t = E[||r_t||^2 / D | t].
```

这里高斯分布仍应表述为用于方差预算匹配的二阶近似，而不是已验证的精确分布。
本实验只比较标量方差的条件形式，不证明残差严格高斯或空间/通道严格各向同性。
若后续仍要使用 `v(c_t)`，应训练专门的非线性方差头，并在独立轨迹上先通过本实验
同样的 `Delta R2 > 0` gate，再进入闭环 VSC/FID 实验。

## 复现命令

```bash
python PTQD/eta1_mean_samplewise_v2/analyze_state_dependent_variance.py \
  --data PTQD/eta1_mean_samplewise_v2/traj_eta1_dt_only_n500.pt \
  --ckpt PTQD/eta1_mean_samplewise_v2/checkpoints/ckpt_best.pt \
  --output_dir PTQD/eta1_mean_samplewise_v2/state_dependent_variance_core
```

产物：

- `metrics.json`：完整数值、划分和判据；
- `state_variance_ablation.png`：核心消融和方差校准图；
- `logs/run.log`：运行日志。
