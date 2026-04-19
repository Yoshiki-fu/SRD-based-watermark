"""
src/losses パッケージ

現在実装済み:
    ReconstructionLoss  — L_rec (Mel L1)
    WatermarkLoss       — L_wm (BCE + BER)
    InfoNCELoss         — L_MI_max (z_c_fused と W の MI 最大化)
    VCLUBLoss           — L_disentangle (z_c と z_r/z_f の MI 最小化)
    AdversarialLoss     — L_MI_min (GRL + Speaker Discriminator)
"""

from src.losses.reconstruction import ReconstructionLoss
from src.losses.watermark_loss import WatermarkLoss
from src.losses.info_nce import InfoNCELoss
from src.losses.vclub import VCLUBLoss
from src.losses.adversarial import AdversarialLoss, GradientReversalLayer, SpeakerDiscriminator

__all__ = [
    'ReconstructionLoss',
    'WatermarkLoss',
    'InfoNCELoss',
    'VCLUBLoss',
    'AdversarialLoss',
    'GradientReversalLayer',
    'SpeakerDiscriminator',
]
