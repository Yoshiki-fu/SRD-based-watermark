"""
VC-Robust Neural Audio Watermarking — エンコーダ群

Content / Rhythm / F0 / Speaker の4エンコーダと
F0量子化ユーティリティ、事前学習重みロード関数を提供する。

参考実装:
  - ContentEncoder: SRD-VC My_model/model.py Encoder_7 (melストリームのみ)
  - RhythmEncoder:  SRD-VC My_model/model.py Encoder_t
  - F0Encoder:      SRD-VC My_model/model.py Encoder_7 (F0ストリームのみ)
  - SpeakerEncoder: SRD-VC My_model/model.py SpeakerEncoder
  - F0Quantizer:    SRD-VC My_model/utils.py quantize_f0_torch

SRD-VC: Yang et al., "Speech Representation Disentanglement with
Adversarial Mutual Information Learning for One-Shot Voice Conversion",
Interspeech 2022.
"""

import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ConvBlock
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv1d + GroupNorm + ReLU の共有ユニット。

    属性名 (.conv, .norm) はcheckpointキー対応のため変更禁止。
    SRD-VC checkpoint上のキー構造:
        convolutions_{n}.{i}.0.conv.*  ← self.conv
        convolutions_{n}.{i}.1.*       ← self.norm

    Args:
        in_channels:  入力チャネル数
        out_channels: 出力チャネル数
        kernel_size:  Conv1d カーネルサイズ (default 5)
        padding:      Conv1d パディング (default 2)
        chs_grp:      GroupNorm グループ数の分母。out_channels % chs_grp == 0 が必要
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        padding: int = 2,
        chs_grp: int = 16,
    ) -> None:
        super().__init__()
        assert out_channels % chs_grp == 0, (
            f"out_channels ({out_channels}) must be divisible by chs_grp ({chs_grp})"
        )
        self.conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=1,
            padding=padding, dilation=1, bias=True,
        )
        nn.init.xavier_uniform_(self.conv.weight, gain=nn.init.calculate_gain('relu'))
        self.norm = nn.GroupNorm(out_channels // chs_grp, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, T) channel-first
        Returns:
            out: (B, C_out, T) channel-first
        """
        return F.relu(self.norm(self.conv(x)))


# ---------------------------------------------------------------------------
# F0Quantizer
# ---------------------------------------------------------------------------

class F0Quantizer:
    """話者正規化済み連続F0 → 257次元 one-hot への変換ユーティリティ。

    SRD-VC My_model/utils.py の quantize_f0_torch を
    本プロジェクトの型ヒント規約に従いリライト。
    """

    @staticmethod
    def quantize(
        f0_norm: torch.Tensor,
        num_bins: int = 256,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """連続F0 → one-hot エンコーディング。

        Args:
            f0_norm: (B, T) speaker正規化済みF0 in [0, 1]。0 = 無声フレーム。
            num_bins: voiced bin数 (default 256)。total classes = num_bins + 1 = 257。
        Returns:
            one_hot: (B, T, num_bins+1)  one-hot encoding
            indices: (B, T)              整数クラスインデックス
        """
        B = f0_norm.size(0)
        x = f0_norm.reshape(-1).clone()

        uv = (x <= 0)          # unvoiced mask
        x[uv] = 0.0
        x = torch.round(x * (num_bins - 1))
        x = x + 1              # voiced → class 1..num_bins
        x[uv] = 0              # unvoiced → class 0

        enc = torch.zeros(x.size(0), num_bins + 1, device=f0_norm.device)
        enc[torch.arange(x.size(0)), x.long()] = 1.0

        one_hot = enc.view(B, -1, num_bins + 1)     # (B, T, num_bins+1)
        indices = x.view(B, -1).long()               # (B, T)
        return one_hot, indices


# ---------------------------------------------------------------------------
# ContentEncoder
# ---------------------------------------------------------------------------

class ContentEncoder(nn.Module):
    """Mel スペクトログラム → Content codes z_c。

    SRD-VC Encoder_7 の mel ストリームのみを独立クラスとして実装。
    InterpLnr（確率的補間）は省略 — MI loss (InfoNCE/vCLUB) が同等の役割を担う。
    VQ-CPC ブランチは省略。

    属性名 (.convolutions, .lstm) は checkpoint キー対応のため変更禁止:
        G1['module.encoder_1.convolutions_1.*'] → self.convolutions
        G1['module.encoder_1.lstm_1.*']         → self.lstm

    Args:
        dim_freq:      メルバンド数 (default 80)
        dim_enc:       Conv隠れ次元 (default 512)
        dim_neck:      BiLSTM hidden per direction。z_c dim = dim_neck*2 (default 8 → 16)
        freq:          時間ダウンサンプリング率 (default 8)
        chs_grp:       GroupNorm グループ数の分母 (default 16)
        n_conv_layers: ConvBlock数 (default 3)
        lstm_layers:   BiLSTM 層数 (default 2)
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
    ) -> None:
        super().__init__()
        self.dim_neck = dim_neck
        self.freq = freq

        convolutions: List[ConvBlock] = []
        for i in range(n_conv_layers):
            in_ch = dim_freq if i == 0 else dim_enc
            convolutions.append(ConvBlock(in_ch, dim_enc, kernel_size=5, padding=2, chs_grp=chs_grp))
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(
            dim_enc, dim_neck,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (B, dim_freq, T)  channel-first メルスペクトログラム、T=192
        Returns:
            z_c: (B, T//freq, dim_neck*2)  Content codes、default (B, 24, 16)
        """
        x = mel
        for conv in self.convolutions:
            x = conv(x)                     # (B, dim_enc, T)

        x = x.transpose(1, 2)              # (B, T, dim_enc)
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)           # (B, T, dim_neck*2)

        # 因果的ダウンサンプリング (SRD-VC 準拠)
        # forward方向: 各 window の末尾フレームが情報を集約
        # backward方向: 各 window の先頭フレームが情報を集約
        out_fwd = outputs[:, self.freq - 1::self.freq, :self.dim_neck]   # (B, T', dim_neck)
        out_bwd = outputs[:, ::self.freq, self.dim_neck:]                 # (B, T', dim_neck)
        z_c = torch.cat([out_fwd, out_bwd], dim=-1)                       # (B, T', dim_neck*2)
        return z_c


# ---------------------------------------------------------------------------
# RhythmEncoder
# ---------------------------------------------------------------------------

class RhythmEncoder(nn.Module):
    """Mel スペクトログラム → Rhythm codes z_r。

    SRD-VC Encoder_t の忠実なポート。

    属性名 (.convolutions, .lstm) は checkpoint キー対応のため変更禁止:
        G1['module.encoder_2.convolutions.*'] → self.convolutions
        G1['module.encoder_2.lstm.*']         → self.lstm

    Args:
        dim_freq:   メルバンド数 (default 80)
        dim_enc_2:  Conv隠れ次元 (default 128)
        dim_neck_2: BiLSTM hidden per direction。z_r dim = dim_neck_2*2 (default 1 → 2)
        freq_2:     時間ダウンサンプリング率 (default 8)
        chs_grp:    GroupNorm グループ数の分母 (default 16)
    """

    def __init__(
        self,
        dim_freq: int = 80,
        dim_enc_2: int = 128,
        dim_neck_2: int = 1,
        freq_2: int = 8,
        chs_grp: int = 16,
    ) -> None:
        super().__init__()
        self.dim_neck_2 = dim_neck_2
        self.freq_2 = freq_2

        self.convolutions = nn.ModuleList([
            ConvBlock(dim_freq, dim_enc_2, kernel_size=5, padding=2, chs_grp=chs_grp),
        ])

        self.lstm = nn.LSTM(
            dim_enc_2, dim_neck_2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(
        self,
        mel: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            mel:  (B, dim_freq, T)  channel-first メルスペクトログラム、T=192
            mask: Optional (B, T, 1)  マスク（SRD-VC 互換用、通常 None）
        Returns:
            z_r: (B, T//freq_2, dim_neck_2*2)  Rhythm codes、default (B, 24, 2)
        """
        x = mel
        for conv in self.convolutions:
            x = conv(x)                     # (B, dim_enc_2, T)

        x = x.transpose(1, 2)              # (B, T, dim_enc_2)
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)           # (B, T, dim_neck_2*2)

        if mask is not None:
            outputs = outputs * mask

        out_fwd = outputs[:, self.freq_2 - 1::self.freq_2, :self.dim_neck_2]   # (B, T', dim_neck_2)
        out_bwd = outputs[:, ::self.freq_2, self.dim_neck_2:]                    # (B, T', dim_neck_2)
        z_r = torch.cat([out_fwd, out_bwd], dim=-1)                              # (B, T', dim_neck_2*2)
        return z_r


# ---------------------------------------------------------------------------
# F0Encoder
# ---------------------------------------------------------------------------

class F0Encoder(nn.Module):
    """One-hot F0 → F0 codes z_f。

    SRD-VC Encoder_7 の F0 ストリームを独立クラスとして抽出。
    InterpLnr は省略 (ContentEncoder と同理由)。

    入力は time-first (B, T, dim_f0) を受け取り、内部で channel-first に変換する。
    理由: F0Quantizer.quantize() が (B, T, dim_f0) を返すため、
    呼び出し側の transpose を不要にする。

    属性名 (.convolutions, .lstm) は checkpoint キー対応のため変更禁止:
        G1['module.encoder_1.convolutions_2.*'] → self.convolutions
        G1['module.encoder_1.lstm_2.*']         → self.lstm

    Args:
        dim_f0:      one-hot F0 次元 (default 257 = 256 voiced bins + 1 unvoiced)
        dim_enc_3:   Conv隠れ次元 (default 256)
        dim_neck_3:  BiLSTM hidden per direction。z_f dim = dim_neck_3*2 (default 32 → 64)
        freq_3:      時間ダウンサンプリング率 (default 8)
        chs_grp:     GroupNorm グループ数の分母 (default 16)
        n_conv_layers: ConvBlock数 (default 3)
    """

    def __init__(
        self,
        dim_f0: int = 257,
        dim_enc_3: int = 256,
        dim_neck_3: int = 32,
        freq_3: int = 8,
        chs_grp: int = 16,
        n_conv_layers: int = 3,
    ) -> None:
        super().__init__()
        self.dim_neck_3 = dim_neck_3
        self.freq_3 = freq_3

        convolutions: List[ConvBlock] = []
        for i in range(n_conv_layers):
            in_ch = dim_f0 if i == 0 else dim_enc_3
            convolutions.append(ConvBlock(in_ch, dim_enc_3, kernel_size=5, padding=2, chs_grp=chs_grp))
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(
            dim_enc_3, dim_neck_3,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, f0_onehot: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f0_onehot: (B, T, dim_f0)  time-first one-hot F0。T=192、dim_f0=257。
                       F0Quantizer.quantize() の出力をそのまま渡す。
        Returns:
            z_f: (B, T//freq_3, dim_neck_3*2)  F0 codes、default (B, 24, 64)
        """
        x = f0_onehot.transpose(1, 2)      # (B, dim_f0, T)  channel-first に変換

        for conv in self.convolutions:
            x = conv(x)                     # (B, dim_enc_3, T)

        x = x.transpose(1, 2)              # (B, T, dim_enc_3)
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)           # (B, T, dim_neck_3*2)

        out_fwd = outputs[:, self.freq_3 - 1::self.freq_3, :self.dim_neck_3]   # (B, T', dim_neck_3)
        out_bwd = outputs[:, ::self.freq_3, self.dim_neck_3:]                    # (B, T', dim_neck_3)
        z_f = torch.cat([out_fwd, out_bwd], dim=-1)                              # (B, T', dim_neck_3*2)
        return z_f


# ---------------------------------------------------------------------------
# SpeakerEncoder — ヘルパー関数
# ---------------------------------------------------------------------------

def _pad_layer(inp: torch.Tensor, layer: nn.Conv1d, pad_type: str = 'reflect') -> torch.Tensor:
    """Conv1d 前の reflect padding。

    SRD-VC My_model/model.py の pad_layer 関数のポート。

    Args:
        inp:      (B, C, T) channel-first
        layer:    nn.Conv1d（kernel_size 属性を参照）
        pad_type: F.pad の mode (default 'reflect')
    Returns:
        out: (B, C_out, T') layer 適用後のテンソル
    """
    kernel_size = layer.kernel_size[0]
    if kernel_size % 2 == 0:
        pad = (kernel_size // 2, kernel_size // 2 - 1)
    else:
        pad = (kernel_size // 2, kernel_size // 2)
    inp = F.pad(inp, pad=pad, mode=pad_type)
    return layer(inp)


def _conv_bank(
    x: torch.Tensor,
    module_list: nn.ModuleList,
    act: nn.Module,
    pad_type: str = 'reflect',
) -> torch.Tensor:
    """並列 Conv bank + concat。

    SRD-VC My_model/model.py の conv_bank 関数のポート。

    Args:
        x:           (B, C_in, T) channel-first
        module_list: 異なるカーネルサイズの Conv1d のリスト
        act:         活性化関数
        pad_type:    reflect padding
    Returns:
        out: (B, bank_size * c_bank + C_in, T) 各 bank 出力 + 元テンソル を concat
    """
    outs = [act(_pad_layer(x, layer, pad_type)) for layer in module_list]
    return torch.cat(outs + [x], dim=1)


# ---------------------------------------------------------------------------
# SpeakerEncoder
# ---------------------------------------------------------------------------

class SpeakerEncoder(nn.Module):
    """Mel スペクトログラム → Speaker embedding z_s。

    SRD-VC My_model/model.py SpeakerEncoder の忠実なポート
    (参考: AdaIN-VC, jjery2243542/adaptive_voice_conversion)。

    透かしとは無関係 — 透かしは z_c にのみ Fusion する (CLAUDE.md 制約)。

    first_conv_layers / second_conv_layers は生の nn.Conv1d
    (ConvNorm / ConvBlock ラッパーなし)。
    checkpoint キー: encoder_speaker.first_conv_layers.{i}.weight/bias

    Residual 接続:
        subsample > 1 の場合、入力側を avg_pool1d でダウンサンプルしてから加算。

    Args:
        c_in:           入力チャネル数 = メルバンド数 (default 80)
        c_h:            隠れチャネル数 (default 128)
        c_out:          出力次元 = Speaker embedding 次元 (default 256)
        kernel_size:    Conv1d カーネルサイズ (default 5)
        bank_size:      Conv bank のカーネルサイズ上限 (default 8)
        bank_scale:     Conv bank のカーネルサイズステップ (default 1)
        c_bank:         Conv bank の出力チャネル数 (default 128)
        n_conv_blocks:  Residual Conv ブロック数 (default 6)
        n_dense_blocks: Residual Dense ブロック数 (default 6)
        subsample:      各 Conv ブロックの stride リスト (default [1,2,1,2,1,2])
        act:            活性化関数名 'relu' or 'lrelu' (default 'relu')
        dropout_rate:   Dropout 率 (default 0.0)
    """

    def __init__(
        self,
        c_in: int = 80,
        c_h: int = 128,
        c_out: int = 256,
        kernel_size: int = 5,
        bank_size: int = 8,
        bank_scale: int = 1,
        c_bank: int = 128,
        n_conv_blocks: int = 6,
        n_dense_blocks: int = 6,
        subsample: Optional[List[int]] = None,
        act: str = 'relu',
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if subsample is None:
            subsample = [1, 2, 1, 2, 1, 2]

        self.n_conv_blocks = n_conv_blocks
        self.n_dense_blocks = n_dense_blocks
        self.subsample = subsample
        self.act = self._get_act(act)

        # Conv bank: kernel_size = bank_scale, 2*bank_scale, ..., bank_size
        self.conv_bank = nn.ModuleList([
            nn.Conv1d(c_in, c_bank, kernel_size=k)
            for k in range(bank_scale, bank_size + 1, bank_scale)
        ])
        in_channels = c_bank * (bank_size // bank_scale) + c_in
        self.in_conv_layer = nn.Conv1d(in_channels, c_h, kernel_size=1)

        # Residual Conv ブロック — 生の nn.Conv1d (ConvBlock 不使用)
        self.first_conv_layers = nn.ModuleList([
            nn.Conv1d(c_h, c_h, kernel_size=kernel_size)
            for _ in range(n_conv_blocks)
        ])
        self.second_conv_layers = nn.ModuleList([
            nn.Conv1d(c_h, c_h, kernel_size=kernel_size, stride=sub)
            for sub in subsample
        ])

        self.pooling_layer = nn.AdaptiveAvgPool1d(1)

        # Residual Dense ブロック
        self.first_dense_layers = nn.ModuleList([
            nn.Linear(c_h, c_h) for _ in range(n_dense_blocks)
        ])
        self.second_dense_layers = nn.ModuleList([
            nn.Linear(c_h, c_h) for _ in range(n_dense_blocks)
        ])
        self.output_layer = nn.Linear(c_h, c_out)
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    @staticmethod
    def _get_act(act: str) -> nn.Module:
        if act == 'lrelu':
            return nn.LeakyReLU()
        return nn.ReLU()

    def _conv_blocks(self, inp: torch.Tensor) -> torch.Tensor:
        """6 段の Residual Conv ブロック。

        Args:
            inp: (B, c_h, T)
        Returns:
            out: (B, c_h, T')  stride=2 のブロックで T が縮小
        """
        out = inp
        for l in range(self.n_conv_blocks):
            y = _pad_layer(out, self.first_conv_layers[l])   # (B, c_h, T)
            y = self.act(y)
            y = self.dropout_layer(y)
            y = _pad_layer(y, self.second_conv_layers[l])    # (B, c_h, T') stride適用
            y = self.act(y)
            y = self.dropout_layer(y)
            if self.subsample[l] > 1:
                # 入力側を avg_pool でダウンサンプルしてから Residual add
                out = F.avg_pool1d(out, kernel_size=self.subsample[l], ceil_mode=True)
            out = y + out
        return out

    def _dense_blocks(self, inp: torch.Tensor) -> torch.Tensor:
        """6 段の Residual Dense ブロック。

        Args:
            inp: (B, c_h)
        Returns:
            out: (B, c_h)
        """
        out = inp
        for l in range(self.n_dense_blocks):
            y = self.first_dense_layers[l](out)
            y = self.act(y)
            y = self.dropout_layer(y)
            y = self.second_dense_layers[l](y)
            y = self.act(y)
            y = self.dropout_layer(y)
            out = y + out
        return out

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (B, c_in, T)  channel-first メルスペクトログラム (T 可変)
        Returns:
            z_s: (B, c_out)  Speaker embedding、default (B, 256)
        """
        # Conv bank + concat + 次元削減
        out = _conv_bank(mel, self.conv_bank, self.act)   # (B, c_bank*bank_size + c_in, T)
        out = _pad_layer(out, self.in_conv_layer)          # (B, c_h, T)
        out = self.act(out)

        # Residual Conv ブロック
        out = self._conv_blocks(out)                       # (B, c_h, T')

        # Global Average Pooling
        out = self.pooling_layer(out).squeeze(2)           # (B, c_h)

        # Residual Dense ブロック
        out = self._dense_blocks(out)                      # (B, c_h)

        z_s = self.output_layer(out)                       # (B, c_out)
        return z_s


# ---------------------------------------------------------------------------
# 事前学習重みロード
# ---------------------------------------------------------------------------

def load_pretrained_encoders(
    ckpt_path: str,
    content_encoder: ContentEncoder,
    rhythm_encoder: RhythmEncoder,
    f0_encoder: F0Encoder,
    speaker_encoder: SpeakerEncoder,
    strict: bool = True,
) -> None:
    """SRD-VC 事前学習チェックポイントから各エンコーダの重みをロードする。

    対象ファイル: SRD-VC/My_model/my_demo/800000-G.ckpt

    チェックポイントのキー構造 (G1 state_dict):
        module.encoder_1.convolutions_1.{i}.0.conv.*  → ContentEncoder.convolutions.{i}.conv.*
        module.encoder_1.convolutions_1.{i}.1.*       → ContentEncoder.convolutions.{i}.norm.*
        module.encoder_1.lstm_1.*                     → ContentEncoder.lstm.*
        module.encoder_1.convolutions_2.{i}.0.conv.*  → F0Encoder.convolutions.{i}.conv.*
        module.encoder_1.convolutions_2.{i}.1.*       → F0Encoder.convolutions.{i}.norm.*
        module.encoder_1.lstm_2.*                     → F0Encoder.lstm.*
        module.encoder_2.convolutions.{i}.0.conv.*    → RhythmEncoder.convolutions.{i}.conv.*
        module.encoder_2.convolutions.{i}.1.*         → RhythmEncoder.convolutions.{i}.norm.*
        module.encoder_2.lstm.*                       → RhythmEncoder.lstm.*
        module.encoder_speaker.*                      → SpeakerEncoder.* (プレフィックス除去)

    VQ-CPC キー (codebook.*, rnn.*) はスキップ。

    Args:
        ckpt_path:        800000-G.ckpt へのパス
        content_encoder:  ロード対象 ContentEncoder
        rhythm_encoder:   ロード対象 RhythmEncoder
        f0_encoder:       ロード対象 F0Encoder
        speaker_encoder:  ロード対象 SpeakerEncoder
        strict:           True のとき不一致キーがあれば RuntimeError を送出
    """
    ckpt = torch.load(ckpt_path, map_location='cpu')
    g1: Dict[str, torch.Tensor] = ckpt['G1']

    # --- ContentEncoder ---
    content_sd: Dict[str, torch.Tensor] = {}
    # convolutions_1.{i}.0.conv.* → convolutions.{i}.conv.*
    # convolutions_1.{i}.1.*      → convolutions.{i}.norm.*
    pat_conv1 = re.compile(
        r'^module\.encoder_1\.convolutions_1\.(\d+)\.0\.conv\.(weight|bias)$'
    )
    pat_norm1 = re.compile(
        r'^module\.encoder_1\.convolutions_1\.(\d+)\.1\.(weight|bias)$'
    )
    pat_lstm1 = re.compile(r'^module\.encoder_1\.(lstm_1\..+)$')

    for k, v in g1.items():
        m = pat_conv1.match(k)
        if m:
            content_sd[f'convolutions.{m.group(1)}.conv.{m.group(2)}'] = v
            continue
        m = pat_norm1.match(k)
        if m:
            content_sd[f'convolutions.{m.group(1)}.norm.{m.group(2)}'] = v
            continue
        m = pat_lstm1.match(k)
        if m:
            content_sd[m.group(1).replace('lstm_1.', 'lstm.')] = v

    content_encoder.load_state_dict(content_sd, strict=strict)

    # --- F0Encoder ---
    f0_sd: Dict[str, torch.Tensor] = {}
    pat_conv2 = re.compile(
        r'^module\.encoder_1\.convolutions_2\.(\d+)\.0\.conv\.(weight|bias)$'
    )
    pat_norm2 = re.compile(
        r'^module\.encoder_1\.convolutions_2\.(\d+)\.1\.(weight|bias)$'
    )
    pat_lstm2 = re.compile(r'^module\.encoder_1\.(lstm_2\..+)$')

    for k, v in g1.items():
        m = pat_conv2.match(k)
        if m:
            f0_sd[f'convolutions.{m.group(1)}.conv.{m.group(2)}'] = v
            continue
        m = pat_norm2.match(k)
        if m:
            f0_sd[f'convolutions.{m.group(1)}.norm.{m.group(2)}'] = v
            continue
        m = pat_lstm2.match(k)
        if m:
            f0_sd[m.group(1).replace('lstm_2.', 'lstm.')] = v

    f0_encoder.load_state_dict(f0_sd, strict=strict)

    # --- RhythmEncoder ---
    rhythm_sd: Dict[str, torch.Tensor] = {}
    pat_convr = re.compile(
        r'^module\.encoder_2\.convolutions\.(\d+)\.0\.conv\.(weight|bias)$'
    )
    pat_normr = re.compile(
        r'^module\.encoder_2\.convolutions\.(\d+)\.1\.(weight|bias)$'
    )
    pat_lstmr = re.compile(r'^module\.encoder_2\.(lstm\..+)$')

    for k, v in g1.items():
        m = pat_convr.match(k)
        if m:
            rhythm_sd[f'convolutions.{m.group(1)}.conv.{m.group(2)}'] = v
            continue
        m = pat_normr.match(k)
        if m:
            rhythm_sd[f'convolutions.{m.group(1)}.norm.{m.group(2)}'] = v
            continue
        m = pat_lstmr.match(k)
        if m:
            rhythm_sd[m.group(1)] = v

    rhythm_encoder.load_state_dict(rhythm_sd, strict=strict)

    # --- SpeakerEncoder ---
    # G1 の encoder_speaker.* のプレフィックスを除去するだけ
    spk_sd: Dict[str, torch.Tensor] = {}
    prefix = 'module.encoder_speaker.'
    for k, v in g1.items():
        if k.startswith(prefix):
            spk_sd[k[len(prefix):]] = v

    speaker_encoder.load_state_dict(spk_sd, strict=strict)
