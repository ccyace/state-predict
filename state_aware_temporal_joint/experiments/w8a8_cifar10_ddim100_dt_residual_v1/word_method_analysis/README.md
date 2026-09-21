# 面向《量化误差分解》方法的剩余残差实验

本目录检验 Word 推导中两个核心假设：

1. 均值网络后剩余项是否近似满足 `E[r_t | c_t] = 0`；
2. 剩余项能否用按时间独立、协方差为 `Sigma_r(t)` 的高斯噪声替换。

其中 `c_t=(x_t^q, epsilon_t^q, t, t_corr, refresh)`，所有统计均基于固定
dt 头和当前 epsilon 均值修正头产生的 DDIM-100 轨迹。

## 主要结论

### 1. 当前均值头在低阶条件均值上基本通过

在 100 条轨迹上拟合只使用可观测状态摘要的岭回归，并在另外 100 条轨迹
上测试。预测剩余残差的通道均值没有优于恒为零的基线：MSE 相对变化
`-7.00%`，测试 `R2=-0.076`。

这不能证明完整的高维条件均值严格为零，但说明当前阶段优先级不应是继续
叠加同类均值头，而应转向条件尺度/协方差建模。

### 2. 协方差强烈依赖当前状态

仅用 `logSNR(t)`、dt 偏移、refresh 标志以及 x/epsilon 的通道均值、标准差
和 RMS，测试集残差对数 MSE 的预测 `R2=0.807`。按预测尺度分成五组后，
最低组与最高组的实际 MSE 分别为 `2.18e-4` 和 `1.63e-2`，相差约 75 倍。

因此 Word 中的 `Sigma_r` 应实现为 `Sigma_r(c_t)`，至少需要一个状态条件
尺度头；仅查表 `Sigma_r(t)` 会把不同样本的噪声强度混合起来。

### 3. 高斯性仅在去除空间谱形状后有所改善，但仍不成立

- 原始中心化残差：偏度 `-13.47`，超额峰度 `5466.23`；
- 通道白化后：偏度 `-9.79`，超额峰度 `3139.26`；
- 再做粗粒度径向频谱白化：偏度 `-0.012`，超额峰度 `8.86`。

空间频谱结构解释了大量重尾，但白化后仍明显不是标准高斯。后续应比较
高斯、Student-t 或高斯尺度混合分布，而不是直接固定为高斯。

### 4. IID 假设严重低估轨迹累计方差

用精确 DDIM epsilon 系数线性累积 `sum_i C_i r_i`：

| 模型 | 累计通道方差迹 |
|---|---:|
| 真实残差轨迹 | 0.0057160 |
| 按时间独立噪声 | 0.0002932 |
| 独立项 + 相邻步交叉协方差 | 0.0006258 |

真实值分别是后两者的 `19.50` 倍和 `9.13` 倍。因此只使用一步 AR(1) 仍
不足以描述长程相关；需要显式的轨迹潜变量、较高阶状态空间模型，或直接
学习带记忆的噪声过程。

### 5. 它是离散随机漂移误差，不是固定连续扩散率

在 DDIM-50/100/250 三种步数上控制时间区间后，除最低噪声边界外，残差
MSE 对步长基本不变，而残差状态增量方差的幂指数为 `2.055~2.110`，接近
`Delta t^2`，不是连续扩散过程的 `Delta t`。

所以文档中的离散定义
`D_q,i = C_i^2 |Delta t_i| Sigma_r,i` 可以使用，但它必须随采样器步长变化，
且连续极限下趋于零；不能把它解释为与离散化无关的固定 SDE 扩散系数。

## 对下一版实现的建议

均值头暂时保留，新增一个状态条件随机过程头：

1. 输出 `log s_t^2(c_t)`，先建模样本级异方差；
2. 输出低秩通道/频率因子，而不是完整像素协方差；
3. 引入跨步隐藏状态 `h_t`，使创新噪声独立，而最终 `r_t` 可长程相关；
4. 训练时同时最小化条件分布 NLL、一步 DDIM 增量协方差误差，以及多步
   累计增量协方差误差；
5. 只有离线统计通过后，再比较 deterministic、IID、AR、state-space 四组
   采样 FID。当前结果不支持直接进行 IID 注噪的 50k 采样。

## 复现命令

结构、白化和相邻相关分析：

```bash
python state_aware_temporal_joint/analyze_structured_residual.py \
  --data state_aware_temporal_joint/experiments/w8a8_cifar10_ddim100_dt_residual_v1/traj_dt_corrected_n1000.pt \
  --ckpt state_aware_temporal_joint/experiments/w8a8_cifar10_ddim100_dt_residual_v1/checkpoints/ckpt_best.pt \
  --output_dir state_aware_temporal_joint/experiments/w8a8_cifar10_ddim100_dt_residual_v1/word_method_analysis
```

条件均值和条件方差分析：

```bash
python state_aware_temporal_joint/analyze_conditional_residual.py \
  --data state_aware_temporal_joint/experiments/w8a8_cifar10_ddim100_dt_residual_v1/traj_dt_corrected_n1000.pt \
  --ckpt state_aware_temporal_joint/experiments/w8a8_cifar10_ddim100_dt_residual_v1/checkpoints/ckpt_best.pt \
  --output state_aware_temporal_joint/experiments/w8a8_cifar10_ddim100_dt_residual_v1/word_method_analysis/conditional_residual_analysis.json
```

跨 50/100/250 步的步长缩放分析：

```bash
python state_aware_temporal_joint/analyze_step_scaling.py \
  --datasets \
    state_aware_temporal_joint/experiments/w8a8_cifar10_ddim100_dt_residual_v1/word_method_analysis/traj_dt_corrected_50step_n200.pt \
    state_aware_temporal_joint/experiments/w8a8_cifar10_ddim100_dt_residual_v1/traj_dt_corrected_n1000.pt \
    state_aware_temporal_joint/experiments/w8a8_cifar10_ddim100_dt_residual_v1/word_method_analysis/traj_dt_corrected_250step_n200.pt \
  --ckpt state_aware_temporal_joint/experiments/w8a8_cifar10_ddim100_dt_residual_v1/checkpoints/ckpt_best.pt \
  --output state_aware_temporal_joint/experiments/w8a8_cifar10_ddim100_dt_residual_v1/word_method_analysis/step_scaling_analysis.json
```

累计 DDIM 残差增量分析：

```bash
python state_aware_temporal_joint/analyze_cumulative_increment.py \
  --data state_aware_temporal_joint/experiments/w8a8_cifar10_ddim100_dt_residual_v1/traj_dt_corrected_n1000.pt \
  --ckpt state_aware_temporal_joint/experiments/w8a8_cifar10_ddim100_dt_residual_v1/checkpoints/ckpt_best.pt \
  --estimates state_aware_temporal_joint/experiments/w8a8_cifar10_ddim100_dt_residual_v1/word_method_analysis/structured_residual_estimates.pt \
  --output state_aware_temporal_joint/experiments/w8a8_cifar10_ddim100_dt_residual_v1/word_method_analysis/cumulative_increment_analysis.json
```

## 文件

- `structured_residual_analysis.json`：通道、频谱、白化和相邻步统计；
- `structured_residual_estimates.pt`：按时间统计及 DDIM 转移系数；
- `conditional_residual_analysis.json`：条件均值/异方差检验；
- `step_scaling_analysis.json`：不同采样步长的缩放律；
- `cumulative_increment_analysis.json`：真实与 IID/相邻模型的累计方差比较。

注意：这些是初步模型选择实验。所用 held-out 轨迹与原 checkpoint 的验证
集合来自同一 `%5==0` 划分；条件探针内部又按 `%10` 分成互斥校准/测试轨迹，
但最终结论仍应在新收集、从未用于选 checkpoint 的轨迹上复验。
