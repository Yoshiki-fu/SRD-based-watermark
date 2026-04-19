"""
VC-Robust Neural Audio Watermarking — Mel スペクトログラム Decoder

SRD-VC My_model/model.py の Decoder_3 (l.260-288) および
Postnet (l.541-585) を本プロジェクトの規約に従いリライト。

事前学習重みは 800000-G.ckpt の G2['module.decoder.*'] からロード。
fine-tuning 方式で学習可能 (lr=1e-5)。

入出力の transpose 責務:
  入力 z_combined (B, 192, 338) は time-first で受け取る。
  出力 (B, 192, 80) も time-first。
  channel-first への変換は呼び出し元 full_model.py の責務。

SRD-VC: Yang et al., "Speech Representation Disentanglement with
Adversarial Mutual Information Learning for One-Shot Voice Conversion",
Interspeech 2022.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# プライベートヘルパー
# ---------------------------------------------------------------------------

class _ConvNorm(nn.Module):
    """Conv1d ラッパー。checkpoint キー対応のため .conv 属性名を厳守。

    SRD-VC ConvNorm のポート（Postnet 専用）。
    encoders.py の ConvBlock とは別物 — Postnet は BatchNorm1d を使用。

    checkpoint キー: postnet.convolutions.{i}.0.conv.weight/bias
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        w_init_gain: str = 'linear',
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=1,
            padding=padding, bias=True,
        )
        nn.init.xavier_uniform_(
            self.conv.weight,
            gain=nn.init.calculate_gain(w_init_gain),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _LinearNorm(nn.Module):
    """Linear ラッパー。checkpoint キー対応のため .linear_layer 属性名を厳守。

    checkpoint キー: linear_projection.linear_layer.weight/bias
    """

    def __init__(self, in_dim: int, out_dim: int, w_init_gain: str = 'linear') -> None:
        super().__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init_gain),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)


# ---------------------------------------------------------------------------
# Postnet
# ---------------------------------------------------------------------------

class Postnet(nn.Module):
    """5段の 1D-Conv + BatchNorm1d による残差補正ネットワーク。

    SRD-VC My_model/model.py Postnet (l.541-585) のポート。

    checkpoint キー構造 (i=0..4):
        postnet.convolutions.{i}.0.conv.*  ← _ConvNorm
        postnet.convolutions.{i}.1.*       ← nn.BatchNorm1d
            (weight, bias, running_mean, running_var, num_batches_tracked)
    """

    def __init__(self) -> None:
        super().__init__()
        self.convolutions = nn.ModuleList()

        # ブロック 0: 80 → 512, tanh
        self.convolutions.append(nn.Sequential(
            _ConvNorm(80, 512, kernel_size=5, padding=2, w_init_gain='tanh'),
            nn.BatchNorm1d(512),
        ))
        # ブロック 1-3: 512 → 512, tanh
        for _ in range(3):
            self.convolutions.append(nn.Sequential(
                _ConvNorm(512, 512, kernel_size=5, padding=2, w_init_gain='tanh'),
                nn.BatchNorm1d(512),
            ))
        # ブロック 4: 512 → 80, 活性化なし
        self.convolutions.append(nn.Sequential(
            _ConvNorm(512, 80, kernel_size=5, padding=2, w_init_gain='linear'),
            nn.BatchNorm1d(80),
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 80, T)  channel-first mel スペクトログラム
        Returns:
            out: (B, 80, T)  残差補正量（呼び出し元で元の出力と加算）
        """
        for i in range(len(self.convolutions) - 1):
            x = torch.tanh(self.convolutions[i](x))
        x = self.convolutions[-1](x)   # 最終層は活性化なし
        return x


# ---------------------------------------------------------------------------
# MelDecoder
# ---------------------------------------------------------------------------

class MelDecoder(nn.Module):
    """連結済み codes から Mel スペクトログラムを生成する Decoder。

    SRD-VC My_model/model.py Decoder_3 (l.260-288) のリライト。

    入力は full_model.py で構築した連結済みテンソル z_combined を受け取る:
        z_combined = cat([z_c_fused_up, z_r_up, z_f_up, z_s_up], dim=-1)
                   = (B, 192, 16 + 2 + 64 + 256) = (B, 192, 338)

    出力は time-first (B, 192, 80)。channel-first への変換は full_model.py の責務。

    checkpoint キー対応のため属性名を厳守:
        self.lstm              ← G2['module.decoder.lstm.*']
        self.linear_projection ← G2['module.decoder.linear_projection.*']
        self.postnet           ← G2['module.decoder.postnet.*']

    Args:
        dim_neck:    ContentEncoder BiLSTM hidden per direction (default 8)
                     → z_c 次元 = dim_neck * 2 = 16
        dim_neck_2:  RhythmEncoder BiLSTM hidden per direction (default 1)
                     → z_r 次元 = dim_neck_2 * 2 = 2
        dim_neck_3:  F0Encoder BiLSTM hidden per direction (default 32)
                     → z_f 次元 = dim_neck_3 * 2 = 64
        dim_spk_emb: Speaker embedding 次元 (default 256)
        dim_freq:    mel バンド数 = 出力次元 (default 80)
        dim_dec:     BiLSTM hidden per direction (default 512)
        lstm_layers: BiLSTM 積層数 (default 3)
    """

    def __init__(
        self,
        dim_neck: int = 8,
        dim_neck_2: int = 1,
        dim_neck_3: int = 32,
        dim_spk_emb: int = 256,
        dim_freq: int = 80,
        dim_dec: int = 512,
        lstm_layers: int = 3,
    ) -> None:
        super().__init__()

        lstm_input_size = dim_neck * 2 + dim_neck_2 * 2 + dim_neck_3 * 2 + dim_spk_emb
        # = 16 + 2 + 64 + 256 = 338

        self.lstm = nn.LSTM(
            lstm_input_size, dim_dec,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.linear_projection = _LinearNorm(dim_dec * 2, dim_freq)
        self.postnet = Postnet()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T=192, 338)  連結済み codes + speaker embedding (time-first)
        Returns:
            decoder_output:      (B, 192, 80)  Postnet 前メル (time-first)
                                 reconstruction loss の補助ターゲットとして使用可能
            mel_outputs_postnet: (B, 192, 80)  Postnet 後メル (time-first)  ← 主出力
                                 L_rec の主ターゲット、WatermarkExtractor への入力
        """
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)                          # (B, 192, dim_dec*2=1024)
        decoder_output = self.linear_projection(outputs)   # (B, 192, dim_freq=80)

        # Postnet: channel-first で処理し、残差加算して time-first に戻す
        postnet_out = self.postnet(decoder_output.transpose(2, 1))         # (B, 80, 192)
        mel_outputs_postnet = decoder_output + postnet_out.transpose(2, 1)  # (B, 192, 80)

        return decoder_output, mel_outputs_postnet


# ---------------------------------------------------------------------------
# 事前学習重みロード
# ---------------------------------------------------------------------------

def load_pretrained_decoder(
    ckpt_path: str,
    decoder: MelDecoder,
    strict: bool = True,
) -> None:
    """SRD-VC 事前学習チェックポイントから MelDecoder の重みをロードする。

    対象ファイル: SRD-VC/My_model/my_demo/800000-G.ckpt

    チェックポイントのキー構造 (G2 state_dict):
        module.decoder.lstm.*                         → lstm.*
        module.decoder.linear_projection.*            → linear_projection.*
        module.decoder.postnet.*                      → postnet.*

    キーリマップ: 'module.decoder.' プレフィックスを除去するだけ
    （属性構造が完全一致するため追加リマップ不要）。

    Args:
        ckpt_path: 800000-G.ckpt へのパス
        decoder:   ロード対象 MelDecoder
        strict:    True のとき不一致キーがあれば RuntimeError を送出
    """
    ckpt = torch.load(ckpt_path, map_location='cpu')
    g2: Dict[str, torch.Tensor] = ckpt['G2']

    prefix = 'module.decoder.'
    sd = {k[len(prefix):]: v for k, v in g2.items() if k.startswith(prefix)}

    decoder.load_state_dict(sd, strict=strict)
