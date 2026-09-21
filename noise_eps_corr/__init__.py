"""
噪声预测校正（learned eps corrector）集中实现包。

核心：`learned_noise_corrector`（DeltaEpsNet / LearnedNoiseCorrector）
脚本：`noise_eps_corr/scripts/`（采集、训练、端到端流水线）

推理入口仍走仓库根目录的 `scripts/sample_diffusion_ddim.py`
（通过 `--enable_learned_noise_corr` / `--learned_corr_ckpt`），
由 `qdiff/error_correction.py` 加载本包中的校正器。
"""

from noise_eps_corr.learned_noise_corrector import (
    DeltaEpsNet,
    LearnedCorrectorMeta,
    LearnedNoiseCorrector,
    TrajectoryLateDataset,
    batch_cos,
    correction_loss,
    load_corrector_from_ckpt,
    load_learned_corrector,
    save_learned_corrector,
)

__all__ = [
    "DeltaEpsNet",
    "LearnedCorrectorMeta",
    "LearnedNoiseCorrector",
    "TrajectoryLateDataset",
    "batch_cos",
    "correction_loss",
    "load_corrector_from_ckpt",
    "load_learned_corrector",
    "save_learned_corrector",
]
