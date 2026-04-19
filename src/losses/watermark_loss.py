"""
VC-Robust Neural Audio Watermarking — Watermark Loss

L_wm = BCEWithLogitsLoss(W_hat, W)

副次出力として BER（Bit Error Rate）を勾配なしで返す。
"""

from typing import Tuple

import torch
import torch.nn as nn


class WatermarkLoss(nn.Module):
    """L_wm = BCEWithLogitsLoss(W_hat, W)

    W_hat は sigmoid 前の logits として受け取る。
    BCEWithLogitsLoss が内部で sigmoid を適用するため数値的に安定。

    BER は評価指標（勾配不要）として torch.no_grad() ブロック内で計算する。
    """

    def __init__(self) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        W_hat: torch.Tensor,
        W: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            W_hat: (B, num_bits)  透かしビット logits (sigmoid 前)
            W:     (B, num_bits)  バイナリターゲット float {0.0, 1.0}
        Returns:
            loss: scalar BCE loss（勾配あり）
            ber:  scalar Bit Error Rate in [0, 1]（勾配なし、評価用）
        """
        loss = self.bce(W_hat, W)

        with torch.no_grad():
            predicted = (torch.sigmoid(W_hat) > 0.5).float()
            ber = (predicted != W).float().mean()

        return loss, ber
