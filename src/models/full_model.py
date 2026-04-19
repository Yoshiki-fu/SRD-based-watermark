"""
VC-Robust Neural Audio Watermarking — 統合モデル

ContentEncoder / SpeakerEncoder / RhythmEncoder / F0Encoder /
WatermarkEncoder / FusionLayer / MelDecoder / WatermarkExtractor を統合する。

事前学習重みのロード（G1 = エンコーダ群、G2 = Decoder）と
凍結設定（Speaker / Rhythm / F0 Encoder を凍結）もここで完結させる。

SRD-VC: Yang et al., "Speech Representation Disentanglement with
Adversarial Mutual Information Learning for One-Shot Voice Conversion",
Interspeech 2022.
"""

from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn

from src.models.encoders import (
    F0Quantizer,
    ContentEncoder,
    SpeakerEncoder,
    RhythmEncoder,
    F0Encoder,
    load_pretrained_encoders,
)
from src.models.watermark import WatermarkEncoder, FusionLayer, WatermarkExtractor
from src.models.decoder import MelDecoder, load_pretrained_decoder


class VCWatermarkModel(nn.Module):
    """VC-Robust Neural Audio Watermarking 統合モデル。

    forward() は loss 計算に必要な全中間表現を Dict で返す。
    transpose の責務: Decoder 出力 (B,T,80) → (B,80,T) の変換は本クラスが行い、
    AttackLayer および WatermarkExtractor は常に channel-first (B,80,T) を受け取る。

    Args:
        dim_freq:           メルバンド数 (default 80)
        dim_f0:             one-hot F0 次元 (default 257)
        dim_spk_emb:        Speaker embedding 次元 (default 256)
        max_len_pad:        パディング後の系列長 T (default 192)
        chs_grp:            GroupNorm グループ数の分母 (default 16)
        dim_enc:            ContentEncoder Conv 隠れ次元 (default 512)
        dim_neck:           ContentEncoder BiLSTM hidden per direction (default 8)
        freq:               時間ダウンサンプリング率 (default 8)
        content_n_conv:     ContentEncoder ConvBlock 数 (default 3)
        content_lstm_layers: ContentEncoder BiLSTM 層数 (default 2)
        dim_enc_2:          RhythmEncoder Conv 隠れ次元 (default 128)
        dim_neck_2:         RhythmEncoder BiLSTM hidden per direction (default 1)
        freq_2:             RhythmEncoder 時間ダウンサンプリング率 (default 8)
        dim_enc_3:          F0Encoder Conv 隠れ次元 (default 256)
        dim_neck_3:         F0Encoder BiLSTM hidden per direction (default 32)
        freq_3:             F0Encoder 時間ダウンサンプリング率 (default 8)
        f0_n_conv:          F0Encoder ConvBlock 数 (default 3)
        f0_num_bins:        F0 量子化 voiced bin 数 (default 256)
        c_h:                SpeakerEncoder 隠れチャネル数 (default 128)
        kernel_size:        SpeakerEncoder Conv1d カーネルサイズ (default 5)
        bank_size:          SpeakerEncoder Conv bank カーネルサイズ上限 (default 8)
        bank_scale:         SpeakerEncoder Conv bank ステップ (default 1)
        c_bank:             SpeakerEncoder Conv bank 出力チャネル数 (default 128)
        n_conv_blocks:      SpeakerEncoder Residual Conv ブロック数 (default 6)
        n_dense_blocks:     SpeakerEncoder Residual Dense ブロック数 (default 6)
        subsample:          SpeakerEncoder stride リスト (default [1,2,1,2,1,2])
        act:                SpeakerEncoder 活性化関数名 (default 'relu')
        dropout_rate:       SpeakerEncoder Dropout 率 (default 0.0)
        dim_dec:            MelDecoder BiLSTM hidden per direction (default 512)
        lstm_layers:        MelDecoder BiLSTM 積層数 (default 3)
        num_bits:           透かしビット数 N (default 16)
        dim_w:              WatermarkEncoder 出力次元 (default 16)
        encoder_mlp_hidden: WatermarkEncoder MLP 中間層 (default 32)
        extractor_mlp_hidden: WatermarkExtractor MLP 中間層 (default 32)
        fusion_conv_kernel: FusionLayer Conv1d カーネルサイズ (default 3)
        fusion_conv_channels: FusionLayer 中間/出力チャネル数 (default 16)
        ckpt_path:          800000-G.ckpt へのパス。None のとき重みロードをスキップ
        strict:             load_state_dict の strict フラグ (default True)
    """

    def __init__(
        self,
        dim_freq: int = 80,
        dim_f0: int = 257,
        dim_spk_emb: int = 256,
        max_len_pad: int = 192,
        chs_grp: int = 16,
        dim_enc: int = 512,
        dim_neck: int = 8,
        freq: int = 8,
        content_n_conv: int = 3,
        content_lstm_layers: int = 2,
        dim_enc_2: int = 128,
        dim_neck_2: int = 1,
        freq_2: int = 8,
        dim_enc_3: int = 256,
        dim_neck_3: int = 32,
        freq_3: int = 8,
        f0_n_conv: int = 3,
        f0_num_bins: int = 256,
        c_h: int = 128,
        kernel_size: int = 5,
        bank_size: int = 8,
        bank_scale: int = 1,
        c_bank: int = 128,
        n_conv_blocks: int = 6,
        n_dense_blocks: int = 6,
        subsample: Optional[List[int]] = None,
        act: str = 'relu',
        dropout_rate: float = 0.0,
        dim_dec: int = 512,
        lstm_layers: int = 3,
        num_bits: int = 16,
        dim_w: int = 16,
        encoder_mlp_hidden: int = 32,
        extractor_mlp_hidden: int = 32,
        fusion_conv_kernel: int = 3,
        fusion_conv_channels: int = 16,
        ckpt_path: Optional[str] = None,
        strict: bool = True,
    ) -> None:
        super().__init__()
        if subsample is None:
            subsample = [1, 2, 1, 2, 1, 2]

        self.freq = freq
        self.max_len_pad = max_len_pad
        self.f0_num_bins = f0_num_bins

        self.content_encoder = ContentEncoder(
            dim_freq=dim_freq,
            dim_enc=dim_enc,
            dim_neck=dim_neck,
            freq=freq,
            chs_grp=chs_grp,
            n_conv_layers=content_n_conv,
            lstm_layers=content_lstm_layers,
        )
        self.speaker_encoder = SpeakerEncoder(
            c_in=dim_freq,
            c_h=c_h,
            c_out=dim_spk_emb,
            kernel_size=kernel_size,
            bank_size=bank_size,
            bank_scale=bank_scale,
            c_bank=c_bank,
            n_conv_blocks=n_conv_blocks,
            n_dense_blocks=n_dense_blocks,
            subsample=subsample,
            act=act,
            dropout_rate=dropout_rate,
        )
        self.rhythm_encoder = RhythmEncoder(
            dim_freq=dim_freq,
            dim_enc_2=dim_enc_2,
            dim_neck_2=dim_neck_2,
            freq_2=freq_2,
            chs_grp=chs_grp,
        )
        self.f0_encoder = F0Encoder(
            dim_f0=dim_f0,
            dim_enc_3=dim_enc_3,
            dim_neck_3=dim_neck_3,
            freq_3=freq_3,
            chs_grp=chs_grp,
            n_conv_layers=f0_n_conv,
        )
        self.wm_encoder = WatermarkEncoder(
            num_bits=num_bits,
            dim_w=dim_w,
            mlp_hidden=encoder_mlp_hidden,
            time_steps=max_len_pad // freq,  # T' = 24
        )
        self.fusion_layer = FusionLayer(
            dim_c=dim_neck * 2,
            dim_w=dim_w,
            conv_kernel=fusion_conv_kernel,
            conv_channels=fusion_conv_channels,
        )
        self.decoder = MelDecoder(
            dim_neck=dim_neck,
            dim_neck_2=dim_neck_2,
            dim_neck_3=dim_neck_3,
            dim_spk_emb=dim_spk_emb,
            dim_freq=dim_freq,
            dim_dec=dim_dec,
            lstm_layers=lstm_layers,
        )
        self.extractor = WatermarkExtractor(
            dim_freq=dim_freq,
            dim_enc=dim_enc,
            dim_neck=dim_neck,
            freq=freq,
            chs_grp=chs_grp,
            n_conv_layers=content_n_conv,
            lstm_layers=content_lstm_layers,
            num_bits=num_bits,
            mlp_hidden=extractor_mlp_hidden,
        )

        if ckpt_path is not None:
            load_pretrained_encoders(
                ckpt_path,
                self.content_encoder,
                self.rhythm_encoder,
                self.f0_encoder,
                self.speaker_encoder,
                strict=strict,
            )
            load_pretrained_decoder(ckpt_path, self.decoder, strict=strict)

        for module in [self.speaker_encoder, self.rhythm_encoder, self.f0_encoder]:
            for param in module.parameters():
                param.requires_grad_(False)

    def forward(
        self,
        mel: torch.Tensor,
        f0_norm: torch.Tensor,
        W: torch.Tensor,
        attack_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            mel:       (B, dim_freq, T)  channel-first メルスペクトログラム、T=192
            f0_norm:   (B, T)            話者正規化済み F0 in [0, 1]
            W:         (B, num_bits)     float {0, 1} バイナリ透かし
            attack_fn: Optional callable。(B, dim_freq, T) channel-first を受け取り
                       同 shape を返す微分可能な攻撃レイヤー
        Returns:
            dict with keys:
                'z_c':                (B, T', dim_neck*2)   Content codes (透かし混入前)
                'z_c_fused':          (B, T', dim_neck*2)   透かし混入後 Content codes
                'z_s':                (B, dim_spk_emb)      Speaker embedding
                'z_r':                (B, T', dim_neck_2*2) Rhythm codes
                'z_f':                (B, T', dim_neck_3*2) F0 codes
                'mel_before_postnet': (B, T, dim_freq)      Postnet 前メル (time-first)
                'mel_postnet':        (B, T, dim_freq)      Postnet 後メル (time-first)
                'W_hat':              (B, num_bits)          透かしビット logits (sigmoid 前)
                'z_c_hat':            (B, T', dim_neck*2)   Extractor 前段 Content codes
        """
        # エンコーダ
        z_c = self.content_encoder(mel)                          # (B, T', 16)
        z_s = self.speaker_encoder(mel)                          # (B, 256)
        z_r = self.rhythm_encoder(mel)                           # (B, T', 2)

        f0_onehot, _ = F0Quantizer.quantize(f0_norm, num_bins=self.f0_num_bins)
        z_f = self.f0_encoder(f0_onehot)                         # (B, T', 64)

        # 透かし融合
        E_w = self.wm_encoder(W)                                 # (B, T', 16)
        z_c_fused = self.fusion_layer(z_c, E_w)                  # (B, T', 16)

        # アップサンプリング & 連結 → z_combined (B, T, 338)
        z_c_up = z_c_fused.repeat_interleave(self.freq, dim=1)   # (B, T, 16)
        z_r_up = z_r.repeat_interleave(self.freq, dim=1)          # (B, T, 2)
        z_f_up = z_f.repeat_interleave(self.freq, dim=1)          # (B, T, 64)
        z_s_up = z_s.unsqueeze(1).expand(-1, self.max_len_pad, -1)  # (B, T, 256)
        z_combined = torch.cat([z_c_up, z_r_up, z_f_up, z_s_up], dim=-1)  # (B, T, 338)

        # Decoder
        mel_before_postnet, mel_postnet = self.decoder(z_combined)
        # 両方とも time-first (B, T, dim_freq)

        # time-first → channel-first (AttackLayer / Extractor 用)
        mel_cf = mel_postnet.transpose(1, 2)                     # (B, dim_freq, T)
        if attack_fn is not None:
            mel_cf = attack_fn(mel_cf)

        # Extractor
        W_hat, z_c_hat = self.extractor(mel_cf)                  # (B, 16), (B, T', 16)

        return {
            'z_c':                z_c,
            'z_c_fused':          z_c_fused,
            'z_s':                z_s,
            'z_r':                z_r,
            'z_f':                z_f,
            'mel_before_postnet': mel_before_postnet,
            'mel_postnet':        mel_postnet,
            'W_hat':              W_hat,
            'z_c_hat':            z_c_hat,
        }

    def get_param_groups(
        self,
        g_lr: float = 1e-4,
        content_lr: float = 1e-5,
        decoder_lr: float = 1e-5,
    ) -> List[Dict]:
        """Optimizer_G 用パラメータグループを返す。

        Speaker / Rhythm / F0 Encoder は凍結済みのため含めない。
        Optimizer_D (Speaker Discriminator, vCLUB) はモデル外で管理するため含めない。

        Returns:
            Adam に直接渡せる list of {'params': ..., 'lr': ...}
        """
        return [
            {
                'params': (
                    list(self.content_encoder.parameters()) +
                    list(self.decoder.parameters())
                ),
                'lr': content_lr,
            },
            {
                'params': (
                    list(self.wm_encoder.parameters()) +
                    list(self.fusion_layer.parameters()) +
                    list(self.extractor.parameters())
                ),
                'lr': g_lr,
            },
        ]
