"""
VC-Robust Neural Audio Watermarking — 微分可能な攻撃レイヤー

学習パイプライン内でエポック数に応じたカリキュラム的攻撃を適用し、
透かし抽出器がロバストな表現を学習できるよう強制する。

参考: CLAUDE.md §攻撃カリキュラム
"""

from typing import Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistortionLayer(nn.Module):
    """エポック依存カリキュラムに基づく微分可能な攻撃レイヤー。

    trainer.py 側で各エポック開始時に set_epoch(epoch) を呼ぶこと。
    model.eval() 時は self.training=False となり、常に identity を返す。

    カリキュラム:
        Epoch 1 - clean_end:              攻撃なし（identity）
        Epoch clean_end+1 - light_end:    Gaussian noise（要素単位）、Dropout（要素単位）
        Epoch light_end+1 -:              上記 + SpecAugment Time/Freq masking、Down/Up sampling

    各攻撃は apply_prob の確率で独立に適用される。適用順序は固定:
    noise → dropout → time_mask → freq_mask → resample

    trainer.py から呼び出す方法::

        distortion = DistortionLayer(cfg['distortion'])
        # 各エポック開始時:
        distortion.set_epoch(epoch)
        # forward:
        out = model(mel, f0_norm, W, attack_fn=distortion)

    Args:
        cfg: configs/default.yaml の distortion セクション（dict）
    """

    def __init__(self, cfg: Dict) -> None:
        super().__init__()
        self.noise_std: float = cfg['noise_std']
        self.dropout_p: float = cfg['dropout_p']
        self.time_mask_width: int = cfg['time_mask_width']
        self.freq_mask_width: int = cfg['freq_mask_width']
        self.downsample_rate: float = cfg['downsample_rate']
        self.apply_prob: float = cfg['apply_prob']
        self.clean_end: int = cfg['curriculum']['clean_end']
        self.light_end: int = cfg['curriculum']['light_end']
        self.current_epoch: int = 1

    def set_epoch(self, epoch: int) -> None:
        """trainer.py から各エポック開始時に呼び出してカリキュラム段階を更新する。

        Args:
            epoch: 現在のエポック番号（1始まり）
        """
        self.current_epoch = epoch

    def _maybe_apply(
        self,
        mel: torch.Tensor,
        attack_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """apply_prob の確率で attack_fn を適用し、それ以外は mel をそのまま返す。"""
        if torch.rand(1).item() < self.apply_prob:
            return attack_fn(mel)
        return mel

    def _gaussian_noise(self, mel: torch.Tensor) -> torch.Tensor:
        """Gaussian noise を加算する。

        noise_std は log-mel スペクトログラムの値域（通常 ~[-11, 2]）に対して軽微な設定。

        Args:
            mel: (B, 80, T) channel-first メルスペクトログラム
        Returns:
            (B, 80, T)
        """
        return mel + torch.randn_like(mel) * self.noise_std

    def _dropout(self, mel: torch.Tensor) -> torch.Tensor:
        """要素単位の Dropout を適用する。

        F.dropout は各 mel-bin 値を個別にゼロ化する（要素単位）。
        フレーム単位のゼロ化（タイムステップごとの除去）は _time_mask を使うこと。
        training=self.training を渡すことで eval mode 時は自動的に identity になるが、
        forward() の早期リターンにより eval mode では実際には呼ばれない。

        Args:
            mel: (B, 80, T) channel-first メルスペクトログラム
        Returns:
            (B, 80, T)
        """
        return F.dropout(mel, p=self.dropout_p, training=self.training)

    def _time_mask(self, mel: torch.Tensor) -> torch.Tensor:
        """SpecAugment Time masking。バッチ内でサンプル毎に独立なマスクをベクトル化生成。

        time_mask_width フレームを固定幅でゼロ化する。マスクとの乗算により
        マスクされた位置には勾配が流れず、それ以外は素通しで伝搬する。

        # TODO(Phase 2): SpecAugment論文に倣いマスク幅を [0, time_mask_width] の
        #   一様分布からサンプリングする方式に拡張する。

        Args:
            mel: (B, 80, T) channel-first メルスペクトログラム
        Returns:
            (B, 80, T)
        """
        B, _, T = mel.shape
        W = self.time_mask_width
        t_idx = torch.arange(T, device=mel.device).unsqueeze(0)                    # (1, T)
        start = torch.randint(0, T - W + 1, (B,), device=mel.device).unsqueeze(1)  # (B, 1)
        keep = ~((t_idx >= start) & (t_idx < start + W))                           # (B, T) bool
        time_mask = keep.float().unsqueeze(1)                                       # (B, 1, T)
        return mel * time_mask

    def _freq_mask(self, mel: torch.Tensor) -> torch.Tensor:
        """SpecAugment Frequency masking。バッチ内でサンプル毎に独立なマスクをベクトル化生成。

        freq_mask_width バンドを固定幅でゼロ化する。マスクとの乗算により
        マスクされた位置には勾配が流れず、それ以外は素通しで伝搬する。

        # TODO(Phase 2): SpecAugment論文に倣いマスク幅を [0, freq_mask_width] の
        #   一様分布からサンプリングする方式に拡張する。

        Args:
            mel: (B, 80, T) channel-first メルスペクトログラム
        Returns:
            (B, 80, T)
        """
        B, F_dim, _ = mel.shape
        W = self.freq_mask_width
        f_idx = torch.arange(F_dim, device=mel.device).unsqueeze(0)                     # (1, F)
        start = torch.randint(0, F_dim - W + 1, (B,), device=mel.device).unsqueeze(1)   # (B, 1)
        keep = ~((f_idx >= start) & (f_idx < start + W))                                # (B, F) bool
        freq_mask = keep.float().unsqueeze(2)                                            # (B, F, 1)
        return mel * freq_mask

    def _resample(self, mel: torch.Tensor) -> torch.Tensor:
        """T軸のみダウン→アップサンプリングによる情報ボトルネック。

        mel スペクトログラム (B, 80, T) を対象とするため、波形用の
        torchaudio.transforms.Resample は使用不可。F.interpolate で時間軸のみ操作する。

        Args:
            mel: (B, 80, T) channel-first メルスペクトログラム
        Returns:
            (B, 80, T) ダウン→アップサンプリング後（元の T に戻す）
        """
        T = mel.shape[2]
        mel_down = F.interpolate(
            mel,
            scale_factor=self.downsample_rate,
            mode='linear',
            align_corners=False,
        )
        return F.interpolate(mel_down, size=T, mode='linear', align_corners=False)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """カリキュラムに応じた攻撃を適用する。

        eval mode（self.training=False）または clean フェーズでは identity を返す。
        各攻撃は apply_prob の確率で独立に適用される。

        Args:
            mel: (B, 80, T) channel-first メルスペクトログラム
        Returns:
            mel_attacked: (B, 80, T) 同 shape
        """
        if not self.training:
            return mel

        if self.current_epoch <= self.clean_end:
            return mel

        if self.current_epoch <= self.light_end:
            mel = self._maybe_apply(mel, self._gaussian_noise)
            mel = self._maybe_apply(mel, self._dropout)
        else:
            mel = self._maybe_apply(mel, self._gaussian_noise)
            mel = self._maybe_apply(mel, self._dropout)
            mel = self._maybe_apply(mel, self._time_mask)
            mel = self._maybe_apply(mel, self._freq_mask)
            mel = self._maybe_apply(mel, self._resample)

        return mel
