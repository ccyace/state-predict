LDM4 on LSUN-Bedroom256×256, steps=200,η=1.0
python scripts/sample_diffusion_ldm.py \
  -r models/ldm/lsun_beds256/model.ckpt -n 3000 --batch_size 2 \
  -c 200 -e 1.0 --seed 41 \
  --ptq --resume --quant_act --act_bit 8 --a_sym --weight_bit 4 \
  --cali_ckpt bedroom_w4a8_ckpt.pth \
  --enable_learned_noise_corr \
  --learned_corr_ckpt output/w4a8_bedroom_compare/learned_corr/ckpt_best.pt \
  --learned_corr_t_cut 999 --learned_corr_alpha 1.0 \
  -l output/w4a8_bedroom_compare/sample_learned_3k


cifar10 w4a7:
python state_aware_temporal_joint/collect_dt_corrected_residual_data.py \
  --output PTQD/vsc_tvar/traj_w4a7_eta1_dt_mean_s01_n200.pt \
  --dt_ckpt state_aware_temporal_joint/experiments/w8a8_cifar10_ddim100_dt_simple_v1/checkpoints/ckpt_best.pt \
  --time_residual_ckpt PTQD/eta1_mean_samplewise_v2/w4a7_run/checkpoints/ckpt_best.pt \
  --time_residual_strength 0.1 \
  --num_trajectories 200 --batch_size 64 \
  --timesteps 100 --skip_type quad --eta 1.0 \
  --dt_eta 0.5 --dt_max 20 --t_cutoff 300 --n_refresh 8 \
  --cali_ckpt cifar_w4a7_ckpt.pth --weight_bit 4 --act_bit 7 --seed 1234

python PTQD/estimate_time_variance.py \
  --data PTQD/vsc_tvar/traj_w4a7_eta1_dt_mean_s01_n200.pt \
  --output_pt PTQD/vsc_tvar/vsc_time_stats_eta1_tvar_w4a7.pt \
  --output_json PTQD/vsc_tvar/vsc_time_stats_eta1_tvar_w4a7.json \
  --eta 1.0 --timesteps 100 --skip_type quad

python state_aware_temporal_joint/sample_50k.py \
  --disable_adapter --disable_corrector --dt_only_infer --dt_mode refresh \
  --joint_dt_ckpt state_aware_temporal_joint/experiments/w8a8_cifar10_ddim100_dt_simple_v1/checkpoints/ckpt_best.pt \
  --time_residual_ckpt PTQD/eta1_mean_samplewise_v2/w4a7_run/checkpoints/ckpt_best.pt \
  --time_residual_strength 0.1 \
  --eta 1.0 --timesteps 100 --skip_type quad \
  --dt_eta 0.5 --dt_carry_max 20 --t_cutoff 300 --dt_refresh_n 8 \
  --vsc_stats PTQD/vsc_tvar/vsc_time_stats_eta1_tvar_w4a7.pt \
  --vsc_var_field var_mle --vsc_absorb_strength 1.0 --vsc_max_budget_fraction 0.9 \
  --cali_ckpt cifar_w4a7_ckpt.pth --weight_bit 4 --act_bit 7 \
  --max_images 50000 --batch_size 64 --seed 1234 --skip_fid \
  --output_dir PTQD/vsc_tvar/cifar_w4a7_vsc_tvar_50k/vsc_tvar_50k

U-Vit cifar w8a8
python state_aware_temporal_joint/sample_50k.py \
  --backbone uvit \
  --fp_ckpt cifar10_uvit_small.pth \
  --cali_ckpt uvit_experiments/checkpoints/uvit_w8a8_ckpt.pth \
  --cali_data_path cifar_sd1236_sample2048_allst.pt \
  --weight_bit 8 --act_bit 8 --sm_abit 8 \
  --cali_st 10 --cali_n 256 --quant_act --a_sym \
  --disable_adapter --disable_corrector \
  --dt_only_infer --dt_mode refresh \
  --joint_dt_ckpt uvit_experiments/outputs/phase2_w8a8/dt_ckpt/ckpt_best.pt \
  --time_residual_ckpt uvit_experiments/outputs/phase2_w8a8/mean_ckpt/ckpt_best.pt \
  --time_residual_strength 0.1 \
  --eta 1.0 --timesteps 100 --skip_type quad \
  --dt_eta 0.5 --dt_carry_max 20 --t_cutoff 300 --dt_refresh_n 8 \
  --vsc_stats uvit_experiments/outputs/phase2_w8a8/vsc_time_stats_eta1_tvar.pt \
  --vsc_var_field var_mle --vsc_absorb_strength 1.0 --vsc_max_budget_fraction 0.9 \
  --max_images 50000 --batch_size 64 --seed 1234 \
  --output_dir uvit_experiments/outputs/phase3_vsc_tvar_50k

U-Vit cifar w4a8
python state_aware_temporal_joint/sample_50k.py \
  --backbone uvit \
  --fp_ckpt cifar10_uvit_small.pth \
  --cali_ckpt uvit_experiments/checkpoints/uvit_w4a8_ckpt.pth \
  --cali_data_path cifar_sd1236_sample2048_allst.pt \
  --weight_bit 4 --act_bit 8 --sm_abit 8 \
  --cali_st 10 --cali_n 256 --quant_act --a_sym \
  --disable_adapter --disable_corrector \
  --dt_only_infer --dt_mode refresh \
  --joint_dt_ckpt uvit_experiments/outputs/phase2_w4a8/dt_ckpt/ckpt_best.pt \
  --time_residual_ckpt uvit_experiments/outputs/phase2_w4a8/mean_ckpt/ckpt_best.pt \
  --time_residual_strength 0.1 \
  --eta 1.0 --timesteps 100 --skip_type quad \
  --dt_eta 0.5 --dt_carry_max 20 --t_cutoff 300 --dt_refresh_n 8 \
  --vsc_stats uvit_experiments/outputs/phase2_w4a8/vsc_time_stats_eta1_tvar.pt \
  --vsc_var_field var_mle --vsc_absorb_strength 1.0 --vsc_max_budget_fraction 0.9 \
  --max_images 50000 --batch_size 64 --seed 1234 \
  --fid_ref new_real_images/real47500_vsc2500_fid_stats.npz \
  --fid_log uvit_experiments/outputs/phase3_w4a8_vsc_tvar_50k/logs/fid.log \
  --output_dir uvit_experiments/outputs/phase3_w4a8_vsc_tvar_50k
