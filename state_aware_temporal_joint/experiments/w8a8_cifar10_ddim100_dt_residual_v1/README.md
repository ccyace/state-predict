# W8A8 CIFAR-10 DDIM-100: dt correction + residual correction

## Function

This stage freezes the previously trained timestep correction model and trains
an independent epsilon residual predictor on trajectories produced after sparse
timestep refresh. At inference the order is:

1. W8A8 UNet at nominal time.
2. At 8 selected nodes, predict per-sample dt and run W8A8 UNet again at the
   corrected time.
3. Predict the remaining epsilon error from `x`, the actually used quantized
   epsilon, `logSNR(t_nom)`, `logSNR(t_corr)`, their difference, and the refresh
   flag.
4. Add a scaled residual and perform the nominal DDIM update.

The residual output layer is zero-initialized. The dt model, quantized UNet and
float teacher remain frozen during residual training.

## Training objective

`delta_star = eps_float(x, t_nom) - eps_quant_used`

`eps_final = eps_quant_used + delta_pred`

`loss = MSE(delta_pred, delta_star) + 0.1 * (1 - cosine(eps_final, eps_float))`

Best epoch: 30

- validation loss: 0.003977083321175693
- validation MSE: 0.0037655798806614862
- validation cosine term: 0.0021150342219819623

## Commands

Data collection:

```bash
python state_aware_temporal_joint/collect_dt_corrected_residual_data.py \
  --output state_aware_temporal_joint/experiments/w8a8_cifar10_ddim100_dt_residual_v1/traj_dt_corrected_n1000.pt \
  --dt_ckpt state_aware_temporal_joint/experiments/w8a8_cifar10_ddim100_dt_simple_v1/checkpoints/ckpt_best.pt \
  --num_trajectories 1000 --batch_size 64 \
  --timesteps 100 --skip_type quad \
  --dt_eta 0.5 --dt_max 20 --t_cutoff 300 --n_refresh 8
```

Residual training:

```bash
python state_aware_temporal_joint/train_dt_corrected_residual.py \
  --data state_aware_temporal_joint/experiments/w8a8_cifar10_ddim100_dt_residual_v1/traj_dt_corrected_n1000.pt \
  --output_dir state_aware_temporal_joint/experiments/w8a8_cifar10_ddim100_dt_residual_v1/checkpoints \
  --epochs 30 --batch_size 128 --lr 0.001
```

Final sampling (strength 0.1):

```bash
python state_aware_temporal_joint/sample_50k.py \
  --disable_adapter --disable_corrector --dt_only_infer --dt_mode refresh \
  --joint_dt_ckpt state_aware_temporal_joint/experiments/w8a8_cifar10_ddim100_dt_simple_v1/checkpoints/ckpt_best.pt \
  --time_residual_ckpt state_aware_temporal_joint/experiments/w8a8_cifar10_ddim100_dt_residual_v1/checkpoints/ckpt_best.pt \
  --time_residual_strength 0.1 \
  --cali_ckpt cifar_w8a8_ckpt.pth --weight_bit 8 --act_bit 8 --sm_abit 8 \
  --timesteps 100 --skip_type quad --dt_eta 0.5 --dt_carry_max 20 \
  --t_cutoff 300 --dt_refresh_n 8 --max_images 50000 --batch_size 256 \
  --seed 1234 \
  --output_dir state_aware_temporal_joint/experiments/w8a8_cifar10_ddim100_dt_residual_v1/sweep_strength_0.1
```

FID:

```bash
python -m pytorch_fid new_real_images/cifar10_python.npz \
  state_aware_temporal_joint/experiments/w8a8_cifar10_ddim100_dt_residual_v1/sweep_strength_0.1 \
  --device cuda:0
```

## Strength sweep (2,000 images, same seed)

| residual strength | FID-2k |
|---:|---:|
| dt-only / 0.0 | 16.702562011337136 |
| 0.1 | 16.72340514581191 |
| 0.25 | 16.729380578087387 |
| 0.5 | 16.912295096229343 |
| 0.75 | 17.268149238312787 |
| 1.0 | 17.606003496635367 |

The sweep showed clear over-correction at larger strengths. Strength 0.1 was
selected as the best non-zero setting and evaluated with 50,000 images.

## Final result

- Generated images: 50,000, IDs 0 through 49,999
- dt-only FID-50k: 3.637947641737071
- dt + residual (strength 0.1) FID-50k: **3.593168691400024**
- Absolute FID improvement over dt-only: **0.044778950337047**
