"""
VC-Robust Neural Audio Watermarking — Reconstruction Loss

L_rec = L1(mel_target, mel_postnet)
      + aux_weight * L1(mel_target, mel_before_postnet)

SRD-VC My_model/solver.py (ll.289-299) を参考に、
MSE + L1 の両方を使う原実装から L1 のみに簡略化。
補助出力への均等重み付け (aux_weight=1.0) は SRD-VC 準拠。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionLoss(nn.Module):
    """L_rec = L1(target, postnet) + aux_weight * L1(target, before_postnet)

    全入力は time-first (B, T, mel_bands) で受け取る。
    DataLoader の channel-first mel を transpose する責務は trainer.py にある。

    Args:
        aux_weight: mel_before_postnet に対する補助 loss の重み (default 1.0)
    """

    def __init__(self, aux_weight: float = 1.0) -> None:
        super().__init__()
        self.aux_weight = aux_weight

    def forward(
        self,
        mel_target: torch.Tensor,
        mel_postnet: torch.Tensor,
        mel_before_postnet: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            mel_target:          (B, T, mel_bands)  元のメルスペクトログラム (time-first)
            mel_postnet:         (B, T, mel_bands)  Decoder 主出力 (time-first)
            mel_before_postnet:  (B, T, mel_bands)  Decoder 補助出力 (time-first)
        Returns:
            loss: scalar
        """
        loss_postnet = F.l1_loss(mel_postnet, mel_target)
        loss_before = F.l1_loss(mel_before_postnet, mel_target)
        return loss_postnet + self.aux_weight * loss_before
