"""
VC-Robust Neural Audio Watermarking — 透かし埋め込み・抽出モジュール

WatermarkEncoder : バイナリ透かし W → 潜在表現 E_w
FusionLayer      : Content codes z_c + E_w → z_c_fused
WatermarkExtractor: 再合成メルスペクトログラム → 透かし推定 W_hat

設計制約 (CLAUDE.md):
  - 透かしは z_c にのみ Fusion する (z_s, z_r, z_f には混ぜない)
  - Extractor は ContentEncoder と完全に独立 (weight 共有なし)
  - Extractor 前段は ContentEncoder と同じアーキテクチャ (重みは独立)
  - N=16, D_c=16

transpose の責務:
  Decoder 出力 (B, 192, 80) → (B, 80, 192) の transpose は full_model.py が行う。
  AttackLayer および WatermarkExtractor は常に channel-first (B, 80, T) を受け取る。
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoders import ContentEncoder


# ---------------------------------------------------------------------------
# WatermarkEncoder
# ---------------------------------------------------------------------------

class WatermarkEncoder(nn.Module):
    """バイナリ透かし W → 時間軸展開済み潜在表現 E_w。

    W ∈ {0, 1}^N を MLP で D_w 次元の連続空間に埋め込み、
    時間軸 T'=24 に repeat 展開して FusionLayer の入力とする。

    Args:
        num_bits:   透かしビット数 N (default 16、= D_c)
        dim_w:      MLP 出力次元 D_w (default 16)
                    D_w = D_c = 16 を推奨:
                    FusionLayer での concat が 16+16=32 で z_c と対称、
                    2:1 圧縮で学習安定、InfoNCE と同一次元空間。
        mlp_hidden: MLP 中間層次元 (default 32)
        time_steps: repeat 展開後の時間ステップ数 T' (default 24 = 192//8)
    """

    def __init__(
        self,
        num_bits: int = 16,
        dim_w: int = 16,
        mlp_hidden: int = 32,
        time_steps: int = 24,
    ) -> None:
        super().__init__()
        self.time_steps = time_steps

        self.mlp = nn.Sequential(
            nn.Linear(num_bits, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, dim_w),
            # 最終層に活性化なし: z_c と同等のスケールに揃えるため
        )

    def forward(self, W: torch.Tensor) -> torch.Tensor:
        """
        Args:
            W: (B, num_bits)  バイナリ透かし {0, 1}^N (float で保持)
        Returns:
            E_w: (B, T', dim_w)  時間軸展開済み透かし埋め込み、default (B, 24, 16)
        """
        e_w = self.mlp(W)                                        # (B, dim_w)
        E_w = e_w.unsqueeze(1).expand(-1, self.time_steps, -1)  # (B, T', dim_w)
        return E_w


# ---------------------------------------------------------------------------
# FusionLayer
# ---------------------------------------------------------------------------

class FusionLayer(nn.Module):
    """Content codes z_c と透かし埋め込み E_w を融合して z_c_fused を生成。

    z_c と E_w をチャネル次元で concat し、2 層の 1D-CNN で D_c 次元に
    圧縮する。Residual 接続なし — concat から全情報を再構成させることで
    透かし情報を無視する local minimum を防ぐ。

    Args:
        dim_c:         Content codes の特徴次元 D_c (default 16)
        dim_w:         透かし埋め込みの特徴次元 D_w (default 16)
        conv_kernel:   Conv1d カーネルサイズ (default 3)
        conv_channels: Conv1d 中間/出力チャネル数 (default 16 = D_c)
    """

    def __init__(
        self,
        dim_c: int = 16,
        dim_w: int = 16,
        conv_kernel: int = 3,
        conv_channels: int = 16,
    ) -> None:
        super().__init__()
        padding = conv_kernel // 2

        # 1 層目: GroupNorm(groups=1) は LayerNorm 相当で 16ch に安定
        self.conv1 = nn.Conv1d(dim_c + dim_w, conv_channels, kernel_size=conv_kernel, padding=padding)
        self.norm1 = nn.GroupNorm(1, conv_channels)

        # 2 層目: 活性化なし — Decoder に入力するため z_c の分布スケールを保持
        self.conv2 = nn.Conv1d(conv_channels, dim_c, kernel_size=conv_kernel, padding=padding)

    def forward(self, z_c: torch.Tensor, E_w: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_c: (B, T', dim_c)  Content codes、default (B, 24, 16)
            E_w: (B, T', dim_w)  透かし埋め込み、default (B, 24, 16)
        Returns:
            z_c_fused: (B, T', dim_c)  融合済み Content codes、default (B, 24, 16)
        """
        x = torch.cat([z_c, E_w], dim=-1)   # (B, T', dim_c + dim_w)
        x = x.transpose(1, 2)               # (B, dim_c + dim_w, T')  channel-first

        x = F.relu(self.norm1(self.conv1(x)))  # (B, conv_channels, T')
        x = self.conv2(x)                       # (B, dim_c, T')

        z_c_fused = x.transpose(1, 2)          # (B, T', dim_c)
        return z_c_fused


# ---------------------------------------------------------------------------
# WatermarkExtractor
# ---------------------------------------------------------------------------

class WatermarkExtractor(nn.Module):
    """再合成メルスペクトログラムから透かしビット推定値 W_hat を抽出。

    前段: ContentEncoder と同じアーキテクチャ (重みは完全に独立、共有なし)
          CLAUDE.md 制約: Extractor は ContentEncoder と weight 共有禁止。
          独立インスタンスを保持することで勾配干渉を回避。
    後段: Global Average Pooling (時間軸) → MLP → W_hat

    入力は channel-first (B, 80, T) とする。
    Decoder 出力 (B, 192, 80) → (B, 80, 192) の transpose は
    呼び出し元の full_model.py が責務を持つ。
    AttackLayer もこの仕様に従い channel-first で入出力する。

    Args:
        dim_freq:      入力メルバンド数 (default 80)
        dim_enc:       ContentEncoder Conv 隠れ次元 (default 512)
        dim_neck:      ContentEncoder BiLSTM hidden per direction (default 8)
                       → z_c_hat 特徴次元 = dim_neck * 2 = 16
        freq:          時間ダウンサンプリング率 (default 8)
        chs_grp:       GroupNorm グループ数の分母 (default 16)
        n_conv_layers: ContentEncoder ConvBlock 数 (default 3)
        lstm_layers:   ContentEncoder BiLSTM 層数 (default 2)
        num_bits:      出力ビット数 N = W_hat 次元 (default 16)
        mlp_hidden:    後段 MLP 中間層次元 (default 32)
    """

    def __init__(
        self,
        dim_freq: int = 80,
        dim_enc: int = 512,
        dim_neck: int = 8,
        freq: int = 8,
        chs_grp: int = 16,
        n_conv_layers: int = 3,
        lstm_layers: int = 2,
        num_bits: int = 16,
        mlp_hidden: int = 32,
    ) -> None:
        super().__init__()

        # 前段: ContentEncoder と同じアーキテクチャ、重みは独立
        self.content_encoder = ContentEncoder(
            dim_freq=dim_freq,
            dim_enc=dim_enc,
            dim_neck=dim_neck,
            freq=freq,
            chs_grp=chs_grp,
            n_conv_layers=n_conv_layers,
            lstm_layers=lstm_layers,
        )

        # 後段: GAP → MLP
        dim_c = dim_neck * 2  # = 16
        self.mlp = nn.Sequential(
            nn.Linear(dim_c, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, num_bits),
            # 出力は logits — BCE with logits loss 側で sigmoid を適用
            # 推論時は .sigmoid() を呼んで確率値、> 0.5 で {0,1} に丸める
        )

    def forward(self, mel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            mel: (B, dim_freq, T)  channel-first メルスペクトログラム、T=192。
                 Decoder 出力 (B, 192, 80) → (B, 80, 192) の transpose は
                 full_model.py が行う。
        Returns:
            W_hat:   (B, num_bits)  透かしビット logits (sigmoid 前)
            z_c_hat: (B, T', dim_neck*2)  前段 Content codes、default (B, 24, 16)
                     GRL + Speaker Discriminator (L_MI_min) の入力として使用
        """
        z_c_hat = self.content_encoder(mel)   # (B, T', dim_c)

        # Global Average Pooling: 時間方向を平均化
        pooled = z_c_hat.mean(dim=1)          # (B, dim_c)

        W_hat = self.mlp(pooled)              # (B, num_bits) logits
        return W_hat, z_c_hat


# ---------------------------------------------------------------------------
# 透かし生成ユーティリティ
# ---------------------------------------------------------------------------

def generate_watermark(batch_size: int, num_bits: int = 16) -> torch.Tensor:
    """In-batch unique なバイナリ透かしを生成する。

    CLAUDE.md 仕様:
        torch.randperm(2**N)[:B] でユニーク整数をサンプリングし
        N ビットバイナリに変換。In-batch negatives の False Negative を回避。

    Args:
        batch_size: バッチサイズ B (B <= 2**num_bits が必要)
        num_bits:   透かしビット数 N (default 16)
    Returns:
        W: (B, num_bits)  float {0.0, 1.0}  バイナリ透かし
    """
    assert batch_size <= 2 ** num_bits, (
        f"batch_size ({batch_size}) must be <= 2**num_bits ({2**num_bits})"
    )
    indices = torch.randperm(2 ** num_bits)[:batch_size]  # (B,) unique integers

    # 整数 → N ビットバイナリ展開
    bits = torch.arange(num_bits - 1, -1, -1)             # [N-1, N-2, ..., 0]
    W = ((indices.unsqueeze(1) >> bits.unsqueeze(0)) & 1).float()  # (B, N)
    return W
