"""
VC-Robust Neural Audio Watermarking — InfoNCE Loss

L_MI_max = InfoNCE(z_c_fused, W)

透かし混入後の Content 特徴量 z_c_fused と透かし W の相互情報量を
InfoNCE で最大化する。負例は In-batch negatives。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """L_MI_max = InfoNCE(z_c_fused, W)

    z_c_fused を時間方向に平均（Global Average Pooling）して (B, 16) に変換し、
    W を {0,1} → {-1,+1} にマッピングした後、L2正規化したコサイン類似度行列で
    contrastive loss を計算する。symmetric=True のとき双方向損失の平均を返す。

    Args:
        temperature: コサイン類似度のスケーリング温度 τ (default 0.1)
        symmetric:   双方向 InfoNCE を使うか (default True)
    """

    def __init__(self, temperature: float = 0.1, symmetric: bool = True) -> None:
        super().__init__()
        self.temperature = temperature
        self.symmetric = symmetric

    def forward(
        self,
        z_c_fused: torch.Tensor,
        W: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_c_fused: (B, T', D_c) 透かし混入後 Content 特徴量, T'=24, D_c=16
            W:         (B, N) float {0.0, 1.0} バイナリ透かし, N=16
        Returns:
            loss: scalar InfoNCE 損失（最小化により MI を最大化）
        """
        # Global Average Pooling: (B, T', 16) → (B, 16)
        z_avg = z_c_fused.mean(dim=1)

        # L2正規化
        z_norm = F.normalize(z_avg, dim=-1)        # (B, 16)

        # {0,1} → {-1,+1} → L2正規化
        w_pm1 = 2.0 * W - 1.0
        w_norm = F.normalize(w_pm1, dim=-1)        # (B, 16)

        # コサイン類似度行列: sim[i,j] = cos(z_i, w_j) / τ
        sim = z_norm @ w_norm.T / self.temperature  # (B, B)

        # 対角が正例
        B = z_avg.shape[0]
        labels = torch.arange(B, device=z_avg.device)

        loss_z2w = F.cross_entropy(sim, labels)

        if self.symmetric:
            loss_w2z = F.cross_entropy(sim.T, labels)
            return (loss_z2w + loss_w2z) / 2.0

        return loss_z2w
