# W8A8 CIFAR-10 DDIM-100 timestep correction experiment

## Objective

Train only the timestep-offset (`dt`) head for the W8A8 quantized CIFAR-10
DDIM model, then generate 50,000 samples and evaluate FID. The epsilon residual
head is disabled at inference so this experiment isolates timestep correction.

## Code changes in this version

1. The lightweight predictor consumes the diffusion schedule's `logSNR(t)`
   Fourier features instead of only `t / 1000`.
2. Every lightweight residual block receives the time condition through FiLM.
3. The predicted timestep offset remains per-sample (`[B]`) throughout
   inference; it is no longer averaged over the batch.
4. Offline window search evaluates the deterministic DDIM next-state error.
   Only the UNet conditioning time is changed; the DDIM integration grid stays
   nominal, matching sparse-refresh sampling.
5. Label files also store candidate-wise current epsilon-MSE and next-state-MSE
   curves. Training interpolates these curves at the predicted offset and uses

   `L = 1.0 * SmoothL1(dt_pred, dt_star)`
   `  + 0.1 * normalized_current_epsilon_MSE`
   `  + 0.1 * normalized_next_state_MSE`.

   Curve interpolation avoids backpropagating through the full frozen quantized
   UNet on every training iteration.

## Experiment configuration

- Dataset/model: CIFAR-10 DDPM UNet
- Quantization: W8A8, softmax activation 8-bit
- Sampler: deterministic DDIM, quadratic grid, 100 steps
- Training trajectories: open-loop quantized trajectories
- Search: window 40, 11 candidates, `update_mse`
- dt training: 20 epochs, SmoothL1 plus the two alignment terms above
- Inference: 8 sparse timestep refreshes, `dt_eta=0.5`, `t_cutoff=300`
- Samples: 50,000, seed 1234

## Commands

The complete pipeline is launched with:

```bash
python state_aware_temporal_joint/run_dt_eps_oracle_w8a8_250step.py \
  --timesteps 100 \
  --skip_type quad \
  --num_trajectories 1000 \
  --learned_corr_ckpt "" \
  --max_samples 80000 \
  --window 40 \
  --n_grid 11 \
  --criterion update_mse \
  --epochs 20 \
  --train_batch_size 256 \
  --dt_eta 0.5 \
  --t_cutoff 300 \
  --dt_refresh_n 8 \
  --max_images 50000 \
  --sample_batch_size 128 \
  --work_dir state_aware_temporal_joint/experiments/w8a8_cifar10_ddim100_dt_simple_v1 \
  --traj state_aware_temporal_joint/experiments/w8a8_cifar10_ddim100_dt_simple_v1/traj_w8a8_ddim100_n1000.pt \
  --dt_star state_aware_temporal_joint/experiments/w8a8_cifar10_ddim100_dt_simple_v1/dt_labels_update_mse.pt \
  --run_dir state_aware_temporal_joint/experiments/w8a8_cifar10_ddim100_dt_simple_v1/checkpoints \
  --sample_dir state_aware_temporal_joint/experiments/w8a8_cifar10_ddim100_dt_simple_v1/images_50k
```

The pipeline writes `collect.log`, `label.log`, `train.log`, `sample.log`, the
checkpoints, label statistics, image manifest, 50k PNG files, and `fid_50k.txt`
under this directory.

Sampling was initially started at batch size 128. After 2,304 images it was
cleanly interrupted and deterministically resumed at batch size 256 with:

```bash
python state_aware_temporal_joint/sample_50k.py \
  --disable_adapter --disable_corrector --dt_only_infer \
  --dt_mode refresh \
  --joint_dt_ckpt state_aware_temporal_joint/experiments/w8a8_cifar10_ddim100_dt_simple_v1/checkpoints/ckpt_best.pt \
  --cali_ckpt cifar_w8a8_ckpt.pth \
  --weight_bit 8 --act_bit 8 --sm_abit 8 \
  --timesteps 100 --skip_type quad --eta 0 \
  --dt_eta 0.5 --dt_carry_max 20 --t_cutoff 300 --dt_refresh_n 8 \
  --max_images 50000 --batch_size 256 --seed 1234 \
  --output_dir state_aware_temporal_joint/experiments/w8a8_cifar10_ddim100_dt_simple_v1/images_50k
```

FID was computed with:

```bash
python -m pytorch_fid \
  new_real_images/cifar10_python.npz \
  state_aware_temporal_joint/experiments/w8a8_cifar10_ddim100_dt_simple_v1/images_50k \
  --device cuda:0
```

## Results

- Trajectory samples: 96,000 (1,000 trajectories × 96 saved DDIM nodes)
- Labeled/training samples: 80,000
- Final label mean: -0.977
- Final label mean absolute value: 2.608
- Search-window edge fraction: 2.4%
- Best checkpoint epoch: 10
- Best validation dt MAE: 1.9764756393060088
- Best validation combined loss: 1.93126757055521
- Generated PNG count: 50,000
- Effective DDIM steps: 96
- Sampling mode: dt-only sparse refresh, 8 extra UNet calls
- FID-50k: **3.637947641737071**
