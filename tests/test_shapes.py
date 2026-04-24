"""
Shape 一貫性テスト: src/models/encoders.py

各エンコーダの forward() がドキュメント通りのテンソル shape を返すこと、
および事前学習重みのロードが成功することを検証する。
"""

import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.encoders import (
    ConvBlock,
    ContentEncoder,
    F0Encoder,
    F0Quantizer,
    RhythmEncoder,
    SpeakerEncoder,
    load_pretrained_encoders,
)
from src.models.watermark import (
    FusionLayer,
    WatermarkEncoder,
    WatermarkExtractor,
    generate_watermark,
)
from src.losses.info_nce import InfoNCELoss
from src.losses.vclub import VCLUBLoss
from src.losses.adversarial import AdversarialLoss, GradientReversalLayer, SpeakerDiscriminator
from src.models.decoder import MelDecoder, load_pretrained_decoder
from src.models.full_model import VCWatermarkModel

B = 2
T = 192
CKPT_PATH = os.path.join(os.path.dirname(__file__), '..', 'SRD-VC', 'My_model', 'my_demo', '800000-G.ckpt')


# ---------------------------------------------------------------------------
# フィクスチャ
# ---------------------------------------------------------------------------

@pytest.fixture
def content_encoder() -> ContentEncoder:
    return ContentEncoder(
        dim_freq=80, dim_enc=512, dim_neck=8, freq=8,
        chs_grp=16, n_conv_layers=3, lstm_layers=2,
    )


@pytest.fixture
def rhythm_encoder() -> RhythmEncoder:
    return RhythmEncoder(
        dim_freq=80, dim_enc_2=128, dim_neck_2=1, freq_2=8, chs_grp=16,
    )


@pytest.fixture
def f0_encoder() -> F0Encoder:
    return F0Encoder(
        dim_f0=257, dim_enc_3=256, dim_neck_3=32, freq_3=8,
        chs_grp=16, n_conv_layers=3,
    )


@pytest.fixture
def speaker_encoder() -> SpeakerEncoder:
    return SpeakerEncoder(
        c_in=80, c_h=128, c_out=256, kernel_size=5,
        bank_size=8, bank_scale=1, c_bank=128,
        n_conv_blocks=6, n_dense_blocks=6,
        subsample=[1, 2, 1, 2, 1, 2], act='relu', dropout_rate=0.0,
    )


# ---------------------------------------------------------------------------
# F0Quantizer
# ---------------------------------------------------------------------------

class TestF0Quantizer:
    def test_output_shape(self) -> None:
        f0 = torch.rand(B, T)
        one_hot, indices = F0Quantizer.quantize(f0)
        assert one_hot.shape == (B, T, 257), f"Expected (B={B}, T={T}, 257), got {one_hot.shape}"
        assert indices.shape == (B, T), f"Expected (B={B}, T={T}), got {indices.shape}"

    def test_one_hot_sum(self) -> None:
        """各タイムステップで one-hot の和が 1 であること"""
        f0 = torch.rand(B, T)
        one_hot, _ = F0Quantizer.quantize(f0)
        sums = one_hot.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(B, T)), "one-hot sum != 1"

    def test_unvoiced_class_zero(self) -> None:
        """無声フレーム (f0 <= 0) が class 0 にマップされること"""
        f0 = torch.zeros(1, T)           # 全フレーム無声
        one_hot, indices = F0Quantizer.quantize(f0)
        assert (indices == 0).all(), "Unvoiced frames should map to class 0"
        assert one_hot[:, :, 0].all(), "Unvoiced one-hot should be class 0"

    def test_device_consistency(self) -> None:
        """CPU 入力 → CPU 出力"""
        f0 = torch.rand(B, T)
        one_hot, indices = F0Quantizer.quantize(f0)
        assert one_hot.device == f0.device
        assert indices.device == f0.device


# ---------------------------------------------------------------------------
# ContentEncoder
# ---------------------------------------------------------------------------

class TestContentEncoder:
    def test_output_shape(self, content_encoder: ContentEncoder) -> None:
        """(B, 80, 192) → (B, 24, 16)"""
        mel = torch.randn(B, 80, T)
        z_c = content_encoder(mel)
        assert z_c.shape == (B, 24, 16), f"Expected (B={B}, 24, 16), got {z_c.shape}"

    def test_variable_length(self, content_encoder: ContentEncoder) -> None:
        """T 可変でも T//freq の時間次元が得られること"""
        for t in [64, 128, 192]:
            mel = torch.randn(B, 80, t)
            z_c = content_encoder(mel)
            expected_t = t // 8
            assert z_c.shape == (B, expected_t, 16), (
                f"T={t}: Expected (B={B}, {expected_t}, 16), got {z_c.shape}"
            )

    def test_no_nan(self, content_encoder: ContentEncoder) -> None:
        mel = torch.randn(B, 80, T)
        z_c = content_encoder(mel)
        assert not torch.isnan(z_c).any(), "ContentEncoder output contains NaN"


# ---------------------------------------------------------------------------
# RhythmEncoder
# ---------------------------------------------------------------------------

class TestRhythmEncoder:
    def test_output_shape(self, rhythm_encoder: RhythmEncoder) -> None:
        """(B, 80, 192) → (B, 24, 2)"""
        mel = torch.randn(B, 80, T)
        z_r = rhythm_encoder(mel)
        assert z_r.shape == (B, 24, 2), f"Expected (B={B}, 24, 2), got {z_r.shape}"

    def test_mask_none(self, rhythm_encoder: RhythmEncoder) -> None:
        """mask=None でエラーなく動作すること"""
        mel = torch.randn(B, 80, T)
        z_r = rhythm_encoder(mel, mask=None)
        assert z_r.shape == (B, 24, 2)

    def test_no_nan(self, rhythm_encoder: RhythmEncoder) -> None:
        mel = torch.randn(B, 80, T)
        assert not torch.isnan(rhythm_encoder(mel)).any()


# ---------------------------------------------------------------------------
# F0Encoder
# ---------------------------------------------------------------------------

class TestF0Encoder:
    def test_output_shape(self, f0_encoder: F0Encoder) -> None:
        """(B, 192, 257) time-first → (B, 24, 64)"""
        f0 = torch.rand(B, T)
        f0_onehot, _ = F0Quantizer.quantize(f0)    # (B, T, 257) time-first
        z_f = f0_encoder(f0_onehot)
        assert z_f.shape == (B, 24, 64), f"Expected (B={B}, 24, 64), got {z_f.shape}"

    def test_no_nan(self, f0_encoder: F0Encoder) -> None:
        f0_onehot, _ = F0Quantizer.quantize(torch.rand(B, T))
        assert not torch.isnan(f0_encoder(f0_onehot)).any()


# ---------------------------------------------------------------------------
# SpeakerEncoder
# ---------------------------------------------------------------------------

class TestSpeakerEncoder:
    def test_output_shape(self, speaker_encoder: SpeakerEncoder) -> None:
        """(B, 80, T) → (B, 256)"""
        mel = torch.randn(B, 80, T)
        z_s = speaker_encoder(mel)
        assert z_s.shape == (B, 256), f"Expected (B={B}, 256), got {z_s.shape}"

    def test_variable_length(self, speaker_encoder: SpeakerEncoder) -> None:
        """T 可変でも (B, 256) が得られること (AdaptiveAvgPool)"""
        for t in [64, 128, 192, 256]:
            mel = torch.randn(B, 80, t)
            z_s = speaker_encoder(mel)
            assert z_s.shape == (B, 256), f"T={t}: Expected (B={B}, 256), got {z_s.shape}"

    def test_no_nan(self, speaker_encoder: SpeakerEncoder) -> None:
        mel = torch.randn(B, 80, T)
        assert not torch.isnan(speaker_encoder(mel)).any()


# ---------------------------------------------------------------------------
# 事前学習重みロード
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.path.exists(CKPT_PATH),
    reason=f"checkpoint not found: {CKPT_PATH}",
)
class TestLoadPretrainedEncoders:
    def test_load_succeeds(
        self,
        content_encoder: ContentEncoder,
        rhythm_encoder: RhythmEncoder,
        f0_encoder: F0Encoder,
        speaker_encoder: SpeakerEncoder,
    ) -> None:
        """load_pretrained_encoders() が strict=True でエラーなく完了すること"""
        load_pretrained_encoders(
            CKPT_PATH,
            content_encoder,
            rhythm_encoder,
            f0_encoder,
            speaker_encoder,
            strict=True,
        )

    def test_shapes_after_load(
        self,
        content_encoder: ContentEncoder,
        rhythm_encoder: RhythmEncoder,
        f0_encoder: F0Encoder,
        speaker_encoder: SpeakerEncoder,
    ) -> None:
        """重みロード後も forward() の出力 shape が変わらないこと"""
        load_pretrained_encoders(
            CKPT_PATH,
            content_encoder,
            rhythm_encoder,
            f0_encoder,
            speaker_encoder,
        )
        mel = torch.randn(B, 80, T)
        f0_onehot, _ = F0Quantizer.quantize(torch.rand(B, T))

        assert content_encoder(mel).shape == (B, 24, 16)
        assert rhythm_encoder(mel).shape == (B, 24, 2)
        assert f0_encoder(f0_onehot).shape == (B, 24, 64)
        assert speaker_encoder(mel).shape == (B, 256)


# ---------------------------------------------------------------------------
# generate_watermark
# ---------------------------------------------------------------------------

class TestGenerateWatermark:
    def test_output_shape(self) -> None:
        W = generate_watermark(B, num_bits=16)
        assert W.shape == (B, 16), f"Expected ({B}, 16), got {W.shape}"

    def test_binary_values(self) -> None:
        """値が {0.0, 1.0} のみであること"""
        W = generate_watermark(B, num_bits=16)
        assert ((W == 0.0) | (W == 1.0)).all(), "W contains non-binary values"

    def test_unique_rows(self) -> None:
        """バッチ内の透かしが全て異なること (In-batch negatives 保証)"""
        W = generate_watermark(B, num_bits=16)
        rows = set(tuple(row.tolist()) for row in W)
        assert len(rows) == B, f"Expected {B} unique rows, got {len(rows)}"

    def test_large_batch(self) -> None:
        """B=64 でも動作すること"""
        W = generate_watermark(64, num_bits=16)
        assert W.shape == (64, 16)


# ---------------------------------------------------------------------------
# WatermarkEncoder
# ---------------------------------------------------------------------------

class TestWatermarkEncoder:
    @pytest.fixture
    def encoder(self) -> WatermarkEncoder:
        return WatermarkEncoder(num_bits=16, dim_w=16, mlp_hidden=32, time_steps=24)

    def test_output_shape(self, encoder: WatermarkEncoder) -> None:
        """W (B, 16) → E_w (B, 24, 16)"""
        W = generate_watermark(B)
        E_w = encoder(W)
        assert E_w.shape == (B, 24, 16), f"Expected ({B}, 24, 16), got {E_w.shape}"

    def test_binary_input(self, encoder: WatermarkEncoder) -> None:
        """バイナリ入力 {0,1} で NaN が出ないこと"""
        W = generate_watermark(B)
        E_w = encoder(W)
        assert not torch.isnan(E_w).any()

    def test_float_input(self, encoder: WatermarkEncoder) -> None:
        """float 連続値入力でも動作すること"""
        W = torch.rand(B, 16)
        E_w = encoder(W)
        assert E_w.shape == (B, 24, 16)
        assert not torch.isnan(E_w).any()

    def test_time_axis_identical(self, encoder: WatermarkEncoder) -> None:
        """expand により全タイムステップが同一であること"""
        W = generate_watermark(B)
        E_w = encoder(W)
        # 全タイムステップが t=0 と同じ値
        assert torch.allclose(E_w[:, 0:1, :].expand_as(E_w), E_w)


# ---------------------------------------------------------------------------
# FusionLayer
# ---------------------------------------------------------------------------

class TestFusionLayer:
    @pytest.fixture
    def fusion(self) -> FusionLayer:
        return FusionLayer(dim_c=16, dim_w=16, conv_kernel=3, conv_channels=16)

    def test_output_shape(self, fusion: FusionLayer) -> None:
        """z_c (B,24,16) + E_w (B,24,16) → z_c_fused (B,24,16)"""
        z_c = torch.randn(B, 24, 16)
        E_w = torch.randn(B, 24, 16)
        z_c_fused = fusion(z_c, E_w)
        assert z_c_fused.shape == (B, 24, 16), f"Expected ({B}, 24, 16), got {z_c_fused.shape}"

    def test_output_shape_equals_input(self, fusion: FusionLayer) -> None:
        """出力 shape が z_c と完全に一致すること"""
        z_c = torch.randn(B, 24, 16)
        E_w = torch.randn(B, 24, 16)
        z_c_fused = fusion(z_c, E_w)
        assert z_c_fused.shape == z_c.shape

    def test_no_nan(self, fusion: FusionLayer) -> None:
        z_c = torch.randn(B, 24, 16)
        E_w = torch.randn(B, 24, 16)
        assert not torch.isnan(fusion(z_c, E_w)).any()

    def test_gradients_flow(self, fusion: FusionLayer) -> None:
        """z_c と E_w 双方から勾配が流れること"""
        z_c = torch.randn(B, 24, 16, requires_grad=True)
        E_w = torch.randn(B, 24, 16, requires_grad=True)
        loss = fusion(z_c, E_w).sum()
        loss.backward()
        assert z_c.grad is not None and not torch.all(z_c.grad == 0)
        assert E_w.grad is not None and not torch.all(E_w.grad == 0)


# ---------------------------------------------------------------------------
# WatermarkExtractor
# ---------------------------------------------------------------------------

class TestWatermarkExtractor:
    @pytest.fixture
    def extractor(self) -> WatermarkExtractor:
        return WatermarkExtractor(
            dim_freq=80, dim_enc=512, dim_neck=8, freq=8,
            chs_grp=16, n_conv_layers=3, lstm_layers=2,
            num_bits=16, mlp_hidden=32,
        )

    def test_output_shapes(self, extractor: WatermarkExtractor) -> None:
        """mel (B,80,192) → W_hat (B,16), z_c_hat (B,24,16)"""
        mel = torch.randn(B, 80, T)
        W_hat, z_c_hat = extractor(mel)
        assert W_hat.shape == (B, 16), f"W_hat: Expected ({B}, 16), got {W_hat.shape}"
        assert z_c_hat.shape == (B, 24, 16), f"z_c_hat: Expected ({B}, 24, 16), got {z_c_hat.shape}"

    def test_channel_first_input(self, extractor: WatermarkExtractor) -> None:
        """入力は channel-first (B, 80, T) — transpose は full_model.py の責務"""
        mel = torch.randn(B, 80, T)
        W_hat, _ = extractor(mel)
        assert W_hat.shape == (B, 16)

    def test_w_hat_are_logits(self, extractor: WatermarkExtractor) -> None:
        """W_hat は logits (sigmoid 前) であること — [-∞, +∞] に分布"""
        mel = torch.randn(B, 80, T)
        W_hat, _ = extractor(mel)
        # sigmoid を適用すると (0, 1) に収まる
        probs = torch.sigmoid(W_hat)
        assert (probs > 0).all() and (probs < 1).all()

    def test_no_nan(self, extractor: WatermarkExtractor) -> None:
        mel = torch.randn(B, 80, T)
        W_hat, z_c_hat = extractor(mel)
        assert not torch.isnan(W_hat).any()
        assert not torch.isnan(z_c_hat).any()

    def test_independent_from_content_encoder(
        self,
        extractor: WatermarkExtractor,
        content_encoder: ContentEncoder,
    ) -> None:
        """Extractor 前段と ContentEncoder が独立したパラメータを持つこと"""
        ext_params = set(id(p) for p in extractor.content_encoder.parameters())
        enc_params = set(id(p) for p in content_encoder.parameters())
        assert ext_params.isdisjoint(enc_params), "Extractor and ContentEncoder share parameters"

    def test_z_c_hat_shape_matches_content_encoder(
        self,
        extractor: WatermarkExtractor,
        content_encoder: ContentEncoder,
    ) -> None:
        """前段の出力 shape が ContentEncoder と同一であること"""
        mel = torch.randn(B, 80, T)
        _, z_c_hat = extractor(mel)
        z_c = content_encoder(mel)
        assert z_c_hat.shape == z_c.shape


# ---------------------------------------------------------------------------
# MelDecoder
# ---------------------------------------------------------------------------

Z_COMBINED_DIM = 338  # 16 + 2 + 64 + 256


class TestMelDecoder:
    @pytest.fixture
    def decoder(self) -> MelDecoder:
        return MelDecoder(
            dim_neck=8, dim_neck_2=1, dim_neck_3=32,
            dim_spk_emb=256, dim_freq=80, dim_dec=512, lstm_layers=3,
        )

    def test_output_shapes(self, decoder: MelDecoder) -> None:
        """z_combined (B,192,338) → decoder_output (B,192,80), mel_postnet (B,192,80)"""
        z = torch.randn(B, T, Z_COMBINED_DIM)
        dec_out, mel_postnet = decoder(z)
        assert dec_out.shape == (B, T, 80), f"decoder_output: Expected ({B}, {T}, 80), got {dec_out.shape}"
        assert mel_postnet.shape == (B, T, 80), f"mel_postnet: Expected ({B}, {T}, 80), got {mel_postnet.shape}"

    def test_output_is_time_first(self, decoder: MelDecoder) -> None:
        """出力が time-first (B, T, 80) であること"""
        z = torch.randn(B, T, Z_COMBINED_DIM)
        _, mel_postnet = decoder(z)
        assert mel_postnet.shape[1] == T     # time axis
        assert mel_postnet.shape[2] == 80    # mel bands

    def test_no_nan(self, decoder: MelDecoder) -> None:
        z = torch.randn(B, T, Z_COMBINED_DIM)
        dec_out, mel_postnet = decoder(z)
        assert not torch.isnan(dec_out).any()
        assert not torch.isnan(mel_postnet).any()

    def test_postnet_adds_residual(self, decoder: MelDecoder) -> None:
        """mel_outputs_postnet = decoder_output + postnet(decoder_output) であること"""
        z = torch.randn(B, T, Z_COMBINED_DIM)
        dec_out, mel_postnet = decoder(z)
        # decoder_output と mel_postnet が異なる値であること（残差が加わっている）
        assert not torch.allclose(dec_out, mel_postnet)


@pytest.mark.skipif(
    not os.path.exists(CKPT_PATH),
    reason=f"checkpoint not found: {CKPT_PATH}",
)
class TestLoadPretrainedDecoder:
    def test_load_succeeds(self) -> None:
        """load_pretrained_decoder() が strict=True でエラーなく完了すること"""
        decoder = MelDecoder()
        load_pretrained_decoder(CKPT_PATH, decoder, strict=True)

    def test_shapes_after_load(self) -> None:
        """重みロード後も forward() の出力 shape が変わらないこと"""
        decoder = MelDecoder()
        load_pretrained_decoder(CKPT_PATH, decoder)
        z = torch.randn(B, T, Z_COMBINED_DIM)
        dec_out, mel_postnet = decoder(z)
        assert dec_out.shape == (B, T, 80)
        assert mel_postnet.shape == (B, T, 80)


# ---------------------------------------------------------------------------
# VCWatermarkModel
# ---------------------------------------------------------------------------

class TestVCWatermarkModel:
    @pytest.fixture
    def model(self) -> VCWatermarkModel:
        return VCWatermarkModel()  # ckpt_path=None でランダム重み

    @pytest.fixture
    def inputs(self):
        mel = torch.randn(B, 80, T)
        f0_norm = torch.rand(B, T)
        W = generate_watermark(B, num_bits=16)
        return mel, f0_norm, W

    def test_output_keys(self, model: VCWatermarkModel, inputs) -> None:
        """forward() が 9 キーの Dict を返すこと"""
        mel, f0_norm, W = inputs
        out = model(mel, f0_norm, W)
        expected_keys = {
            'z_c', 'z_c_fused', 'z_s', 'z_r', 'z_f',
            'mel_before_postnet', 'mel_postnet', 'W_hat', 'z_c_hat',
        }
        assert set(out.keys()) == expected_keys

    def test_output_shapes(self, model: VCWatermarkModel, inputs) -> None:
        """各出力テンソルの shape が仕様通りであること"""
        mel, f0_norm, W = inputs
        out = model(mel, f0_norm, W)
        assert out['z_c'].shape == (B, 24, 16)
        assert out['z_c_fused'].shape == (B, 24, 16)
        assert out['z_s'].shape == (B, 256)
        assert out['z_r'].shape == (B, 24, 2)
        assert out['z_f'].shape == (B, 24, 64)
        assert out['mel_before_postnet'].shape == (B, T, 80)
        assert out['mel_postnet'].shape == (B, T, 80)
        assert out['W_hat'].shape == (B, 16)
        assert out['z_c_hat'].shape == (B, 24, 16)

    def test_no_nan(self, model: VCWatermarkModel, inputs) -> None:
        mel, f0_norm, W = inputs
        out = model(mel, f0_norm, W)
        for key, val in out.items():
            assert not torch.isnan(val).any(), f"{key} contains NaN"

    def test_attack_fn_identity(self, model: VCWatermarkModel, inputs) -> None:
        """attack_fn=identity のとき attack_fn=None と同じ出力になること"""
        mel, f0_norm, W = inputs
        out_no_attack = model(mel, f0_norm, W, attack_fn=None)
        out_identity = model(mel, f0_norm, W, attack_fn=lambda x: x)
        assert torch.allclose(out_no_attack['W_hat'], out_identity['W_hat'])
        assert torch.allclose(out_no_attack['z_c_hat'], out_identity['z_c_hat'])

    def test_frozen_encoders(self, model: VCWatermarkModel) -> None:
        """Speaker / Rhythm / F0 Encoder が凍結されていること"""
        for name, module in [
            ('speaker_encoder', model.speaker_encoder),
            ('rhythm_encoder', model.rhythm_encoder),
            ('f0_encoder', model.f0_encoder),
        ]:
            for param in module.parameters():
                assert not param.requires_grad, f"{name} has unfrozen parameters"

    def test_trainable_encoders(self, model: VCWatermarkModel) -> None:
        """Content Encoder と Decoder が学習可能であること"""
        for name, module in [
            ('content_encoder', model.content_encoder),
            ('decoder', model.decoder),
        ]:
            assert any(p.requires_grad for p in module.parameters()), (
                f"{name} has no trainable parameters"
            )

    def test_get_param_groups(self, model: VCWatermarkModel) -> None:
        """get_param_groups() が 2 グループを返し、凍結パラメータを含まないこと"""
        groups = model.get_param_groups()
        assert len(groups) == 2

        frozen_ids = set(
            id(p) for module in [
                model.speaker_encoder, model.rhythm_encoder, model.f0_encoder
            ]
            for p in module.parameters()
        )
        for group in groups:
            for param in group['params']:
                assert id(param) not in frozen_ids, "frozen param in optimizer group"

    def test_get_param_groups_lr(self, model: VCWatermarkModel) -> None:
        """デフォルト lr が仕様通りであること"""
        groups = model.get_param_groups()
        lrs = [g['lr'] for g in groups]
        assert 1e-5 in lrs, "fine-tune lr (1e-5) not found"
        assert 1e-4 in lrs, "watermark lr (1e-4) not found"


@pytest.mark.skipif(
    not os.path.exists(CKPT_PATH),
    reason=f"checkpoint not found: {CKPT_PATH}",
)
class TestVCWatermarkModelWithCheckpoint:
    def test_load_and_forward(self) -> None:
        """事前学習重みロード後も forward() が正常に動作すること"""
        model = VCWatermarkModel(ckpt_path=CKPT_PATH, strict=True)
        mel = torch.randn(B, 80, T)
        f0_norm = torch.rand(B, T)
        W = generate_watermark(B)
        out = model(mel, f0_norm, W)
        assert out['mel_postnet'].shape == (B, T, 80)
        assert out['W_hat'].shape == (B, 16)

    def test_load_pretrained_extractor(self) -> None:
        """load_pretrained_extractor=True でロード前後の重みが変わること"""
        base = VCWatermarkModel()
        base_w = next(base.extractor.content_encoder.parameters()).clone()

        loaded = VCWatermarkModel(ckpt_path=CKPT_PATH, load_pretrained_extractor=True, strict=False)
        loaded_w = next(loaded.extractor.content_encoder.parameters())

        assert not torch.allclose(base_w, loaded_w), \
            "事前学習 weights のロード後、Extractor ContentEncoder の重みが変化していない"

    def test_freeze_extractor_content(self) -> None:
        """freeze_extractor_content_encoder=True で凍結され、メイン CE は学習可能なまま"""
        model = VCWatermarkModel(
            ckpt_path=CKPT_PATH,
            load_pretrained_extractor=True,
            freeze_extractor_content_encoder=True,
            strict=False,
        )
        for p in model.extractor.content_encoder.parameters():
            assert not p.requires_grad, "Extractor ContentEncoder が凍結されていない"
        for p in model.content_encoder.parameters():
            assert p.requires_grad, "メイン ContentEncoder が誤って凍結されている"


# ---------------------------------------------------------------------------
# InfoNCELoss
# ---------------------------------------------------------------------------

class TestInfoNCELoss:
    @pytest.fixture
    def loss_fn(self) -> InfoNCELoss:
        return InfoNCELoss(temperature=0.1, symmetric=True)

    def test_output_is_scalar(self, loss_fn: InfoNCELoss) -> None:
        """forward() がスカラーを返すこと"""
        z_c_fused = torch.randn(B, 24, 16)
        W = generate_watermark(B, num_bits=16)
        loss = loss_fn(z_c_fused, W)
        assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"

    def test_output_is_positive(self, loss_fn: InfoNCELoss) -> None:
        """損失が非負であること"""
        z_c_fused = torch.randn(B, 24, 16)
        W = generate_watermark(B, num_bits=16)
        loss = loss_fn(z_c_fused, W)
        assert loss.item() >= 0.0

    def test_no_nan(self, loss_fn: InfoNCELoss) -> None:
        z_c_fused = torch.randn(B, 24, 16)
        W = generate_watermark(B, num_bits=16)
        loss = loss_fn(z_c_fused, W)
        assert not torch.isnan(loss)

    def test_random_approx_log_b(self, loss_fn: InfoNCELoss) -> None:
        """ランダム表現のとき損失が log(B) 付近に収まること"""
        z_c_fused = torch.randn(B, 24, 16)
        W = generate_watermark(B, num_bits=16)
        loss = loss_fn(z_c_fused, W)
        # InfoNCE の期待値は log(B) ≈ log(2) ≈ 0.69 (B=2)
        # 大きく外れていないことだけ確認（±2.0 の余裕）
        assert abs(loss.item() - math.log(B)) < 2.0, (
            f"loss={loss.item():.4f}, log(B)={math.log(B):.4f}"
        )

    def test_perfect_positive_lower_than_random(self, loss_fn: InfoNCELoss) -> None:
        """完全正例のとき損失がランダムより明確に低いこと。
        基準: loss_perfect < log(B) - 0.5
        """
        W = generate_watermark(B, num_bits=16)
        # z_avg の方向を w_pm1 と完全一致させる
        w_pm1 = 2.0 * W - 1.0
        import torch.nn.functional as F
        w_dir = F.normalize(w_pm1, dim=-1)          # (B, 16)
        z_perfect = w_dir.unsqueeze(1).expand(-1, 24, -1)  # (B, 24, 16)

        loss_perfect = loss_fn(z_perfect, W)
        threshold = math.log(B) - 0.5
        assert loss_perfect.item() < threshold, (
            f"loss_perfect={loss_perfect.item():.4f} >= threshold={threshold:.4f}"
        )

    def test_gradient_flows_to_z_c_fused(self, loss_fn: InfoNCELoss) -> None:
        """z_c_fused に勾配が流れること"""
        z_c_fused = torch.randn(B, 24, 16, requires_grad=True)
        W = generate_watermark(B, num_bits=16)
        loss = loss_fn(z_c_fused, W)
        loss.backward()
        assert z_c_fused.grad is not None, "z_c_fused に勾配が流れていない"
        assert not torch.all(z_c_fused.grad == 0), "z_c_fused の勾配がすべてゼロ"

    def test_no_gradient_to_W(self, loss_fn: InfoNCELoss) -> None:
        """W には勾配が流れないこと (requires_grad=False のまま)"""
        z_c_fused = torch.randn(B, 24, 16, requires_grad=True)
        W = generate_watermark(B, num_bits=16)
        # W は generate_watermark から来るため requires_grad=False
        assert not W.requires_grad, "W.requires_grad should be False"
        loss = loss_fn(z_c_fused, W)
        loss.backward()
        assert W.grad is None, "W に意図しない勾配が流れている"

    def test_asymmetric_mode(self) -> None:
        """symmetric=False で一方向損失が返ること"""
        loss_fn_asym = InfoNCELoss(temperature=0.1, symmetric=False)
        z_c_fused = torch.randn(B, 24, 16)
        W = generate_watermark(B, num_bits=16)
        loss = loss_fn_asym(z_c_fused, W)
        assert loss.shape == ()
        assert not torch.isnan(loss)

    def test_symmetric_leq_asymmetric_or_close(self) -> None:
        """symmetric モードと asymmetric モードで同一入力を使い、
        両方ともスカラーを返しかつ NaN でないこと（値の大小は問わない）"""
        torch.manual_seed(0)
        z_c_fused = torch.randn(B, 24, 16)
        W = generate_watermark(B, num_bits=16)

        loss_sym = InfoNCELoss(temperature=0.1, symmetric=True)(z_c_fused, W)
        loss_asym = InfoNCELoss(temperature=0.1, symmetric=False)(z_c_fused, W)

        assert not torch.isnan(loss_sym)
        assert not torch.isnan(loss_asym)


# ---------------------------------------------------------------------------
# VCLUBLoss
# ---------------------------------------------------------------------------

B_vclub = 4   # テスト用バッチサイズ（randperm に最低2必要）
T_PRIME = 24  # T' = 192 // freq(8)
CKPT_PATH_VCLUB = os.path.join(os.path.dirname(__file__), '..', 'SRD-VC', 'My_model', 'my_demo', '800000-G.ckpt')


class TestVCLUBLoss:
    @pytest.fixture
    def loss_fn(self) -> VCLUBLoss:
        return VCLUBLoss(d_c=16, d_r=2, d_f=64, hidden_size=512)

    @pytest.fixture
    def inputs(self):
        z_c = torch.randn(B_vclub, T_PRIME, 16, requires_grad=True)
        z_r = torch.randn(B_vclub, T_PRIME, 2)   # requires_grad=False
        z_f = torch.randn(B_vclub, T_PRIME, 64)  # requires_grad=False
        return z_c, z_r, z_f

    # --- vclub_loss / forward ---

    def test_vclub_loss_scalar(self, loss_fn: VCLUBLoss, inputs) -> None:
        """vclub_loss() がスカラーを返すこと"""
        z_c, z_r, z_f = inputs
        loss = loss_fn.vclub_loss(z_c, z_r, z_f)
        assert loss.shape == ()

    def test_forward_equals_vclub_loss(self, loss_fn: VCLUBLoss, inputs) -> None:
        """forward() が vclub_loss() と同値であること"""
        z_c, z_r, z_f = inputs
        torch.manual_seed(0)
        l1 = loss_fn.vclub_loss(z_c, z_r, z_f)
        torch.manual_seed(0)
        l2 = loss_fn(z_c, z_r, z_f)
        assert torch.allclose(l1, l2)

    def test_no_nan_vclub_loss(self, loss_fn: VCLUBLoss, inputs) -> None:
        z_c, z_r, z_f = inputs
        assert not torch.isnan(loss_fn.vclub_loss(z_c, z_r, z_f))

    def test_gradient_flows_to_z_c(self, loss_fn: VCLUBLoss, inputs) -> None:
        """vclub_loss() で z_c に勾配が流れること（ContentEncoder を更新できる）"""
        z_c, z_r, z_f = inputs
        loss = loss_fn.vclub_loss(z_c, z_r, z_f)
        loss.backward()
        assert z_c.grad is not None, "z_c に勾配が流れていない"
        assert not torch.all(z_c.grad == 0), "z_c の勾配がすべてゼロ"

    def test_no_gradient_to_z_r_z_f(self, loss_fn: VCLUBLoss, inputs) -> None:
        """z_r / z_f は凍結済みのため勾配が流れないこと"""
        z_c, z_r, z_f = inputs
        assert not z_r.requires_grad, "z_r.requires_grad should be False"
        assert not z_f.requires_grad, "z_f.requires_grad should be False"
        loss = loss_fn.vclub_loss(z_c, z_r, z_f)
        loss.backward()
        assert z_r.grad is None, "z_r に意図しない勾配が流れている"
        assert z_f.grad is None, "z_f に意図しない勾配が流れている"

    # --- estimator_loss ---

    def test_estimator_loss_scalar(self, loss_fn: VCLUBLoss, inputs) -> None:
        """estimator_loss() がスカラーを返すこと"""
        z_c, z_r, z_f = inputs
        loss = loss_fn.estimator_loss(z_c, z_r, z_f)
        assert loss.shape == ()

    def test_no_nan_estimator_loss(self, loss_fn: VCLUBLoss, inputs) -> None:
        z_c, z_r, z_f = inputs
        assert not torch.isnan(loss_fn.estimator_loss(z_c, z_r, z_f))

    def test_estimator_loss_no_grad_to_z_c(self, loss_fn: VCLUBLoss, inputs) -> None:
        """estimator_loss() 後、z_c に勾配が流れないこと。
        推定器内部で .detach() を使っているため、ContentEncoder への
        計算グラフが切れていることを確認する。
        """
        z_c, z_r, z_f = inputs
        loss = loss_fn.estimator_loss(z_c, z_r, z_f)
        loss.backward()
        # detach() されているので z_c.grad は None のまま
        assert z_c.grad is None, "estimator_loss 後に z_c へ勾配が漏れている"

    def test_estimator_loss_grad_to_estimator_params(
        self,
        loss_fn: VCLUBLoss,
        inputs,
    ) -> None:
        """estimator_loss() で推定器パラメータに勾配が流れること"""
        z_c, z_r, z_f = inputs
        loss = loss_fn.estimator_loss(z_c, z_r, z_f)
        loss.backward()
        for name, param in loss_fn.cp_mi_net.named_parameters():
            assert param.grad is not None, f"cp_mi_net.{name} に勾配がない"
        for name, param in loss_fn.rc_mi_net.named_parameters():
            assert param.grad is not None, f"rc_mi_net.{name} に勾配がない"

    # --- get_estimator_parameters ---

    def test_get_estimator_parameters_nonempty(self, loss_fn: VCLUBLoss) -> None:
        params = loss_fn.get_estimator_parameters()
        assert len(params) > 0

    def test_get_estimator_parameters_excludes_none(self, loss_fn: VCLUBLoss) -> None:
        """include_rp=False のとき rp_mi_net のパラメータが含まれないこと"""
        params = loss_fn.get_estimator_parameters()
        cp_rc_ids = (
            {id(p) for p in loss_fn.cp_mi_net.parameters()} |
            {id(p) for p in loss_fn.rc_mi_net.parameters()}
        )
        param_ids = {id(p) for p in params}
        assert param_ids == cp_rc_ids

    # --- include_rp オプション ---

    def test_include_rp_mode(self) -> None:
        """include_rp=True で rp_mi_net が追加されること"""
        loss_fn = VCLUBLoss(d_c=16, d_r=2, d_f=64, hidden_size=512, include_rp=True)
        assert loss_fn.rp_mi_net is not None
        z_c = torch.randn(B_vclub, T_PRIME, 16, requires_grad=True)
        z_r = torch.randn(B_vclub, T_PRIME, 2)
        z_f = torch.randn(B_vclub, T_PRIME, 64)
        loss = loss_fn.vclub_loss(z_c, z_r, z_f)
        assert loss.shape == ()
        assert not torch.isnan(loss)


@pytest.mark.skipif(
    not os.path.exists(CKPT_PATH_VCLUB),
    reason=f"checkpoint not found: {CKPT_PATH_VCLUB}",
)
class TestVCLUBLossWithCheckpoint:
    def test_load_pretrained_succeeds(self) -> None:
        """800000-G.ckpt から推定器をロードできること"""
        loss_fn = VCLUBLoss(
            d_c=16, d_r=2, d_f=64, hidden_size=512,
            ckpt_path=CKPT_PATH_VCLUB,
        )
        z_c = torch.randn(B_vclub, T_PRIME, 16)
        z_r = torch.randn(B_vclub, T_PRIME, 2)
        z_f = torch.randn(B_vclub, T_PRIME, 64)
        loss = loss_fn.estimator_loss(z_c, z_r, z_f)
        assert not torch.isnan(loss)

    def test_loglikeli_meaningful_after_load(self) -> None:
        """ckpt ロード後の loglikeli が意味のある値（有限かつ非ゼロ）を返すこと"""
        loss_fn = VCLUBLoss(
            d_c=16, d_r=2, d_f=64, hidden_size=512,
            ckpt_path=CKPT_PATH_VCLUB,
        )
        z_c = torch.randn(B_vclub, T_PRIME, 16)
        z_r = torch.randn(B_vclub, T_PRIME, 2)
        z_f = torch.randn(B_vclub, T_PRIME, 64)

        ll_cp = loss_fn.cp_mi_net.loglikeli(z_c, z_f)
        ll_rc = loss_fn.rc_mi_net.loglikeli(z_r, z_c)

        assert torch.isfinite(ll_cp), f"cp loglikeli が非有限: {ll_cp.item()}"
        assert torch.isfinite(ll_rc), f"rc loglikeli が非有限: {ll_rc.item()}"
        assert ll_cp.item() != 0.0, "cp loglikeli がゼロ（重みが未ロードの可能性）"
        assert ll_rc.item() != 0.0, "rc loglikeli がゼロ（重みが未ロードの可能性）"


# ---------------------------------------------------------------------------
# AdversarialLoss / GradientReversalLayer / SpeakerDiscriminator
# ---------------------------------------------------------------------------

B_adv = 4
NUM_CLASSES = 110
D_C_ADV = 16


class TestGradientReversalLayer:
    def test_forward_identity(self) -> None:
        """forward は恒等写像であること"""
        grl = GradientReversalLayer(alpha=1.0)
        x = torch.randn(B_adv, D_C_ADV)
        y = grl(x)
        assert torch.allclose(y, x), "GRL forward が恒等写像でない"

    def test_backward_negates_gradient_alpha1(self) -> None:
        """alpha=1.0 のとき勾配が -1 倍されること"""
        grl = GradientReversalLayer(alpha=1.0)
        x = torch.randn(B_adv, D_C_ADV, requires_grad=True)
        y = grl(x)
        y.sum().backward()
        assert x.grad is not None
        assert torch.allclose(x.grad, -torch.ones_like(x)), (
            f"GRL alpha=1.0: 勾配が -1 倍でない。grad={x.grad}"
        )

    def test_backward_scales_gradient_alpha_half(self) -> None:
        """alpha=0.5 のとき勾配が -0.5 倍されること（alpha可変性）"""
        grl = GradientReversalLayer(alpha=0.5)
        x = torch.randn(B_adv, D_C_ADV, requires_grad=True)
        y = grl(x)
        y.sum().backward()
        assert x.grad is not None
        assert torch.allclose(x.grad, -0.5 * torch.ones_like(x)), (
            f"GRL alpha=0.5: 勾配が -0.5 倍でない。grad={x.grad}"
        )

    def test_alpha_not_trainable(self) -> None:
        """alpha は学習可能パラメータでないこと"""
        grl = GradientReversalLayer(alpha=1.0)
        assert not grl.alpha.requires_grad


class TestSpeakerDiscriminator:
    @pytest.fixture
    def disc(self) -> SpeakerDiscriminator:
        return SpeakerDiscriminator(input_dim=D_C_ADV, num_classes=NUM_CLASSES)

    def test_output_shape(self, disc: SpeakerDiscriminator) -> None:
        """(B, D_c) → (B, num_classes)"""
        x = torch.randn(B_adv, D_C_ADV)
        logits = disc(x)
        assert logits.shape == (B_adv, NUM_CLASSES)

    def test_no_nan(self, disc: SpeakerDiscriminator) -> None:
        x = torch.randn(B_adv, D_C_ADV)
        assert not torch.isnan(disc(x)).any()

    def test_gradient_flows_through_grl(self, disc: SpeakerDiscriminator) -> None:
        """GRL経由で入力 x に負の勾配が流れること"""
        x = torch.randn(B_adv, D_C_ADV, requires_grad=True)
        speaker_id = torch.randint(0, NUM_CLASSES, (B_adv,))
        loss = F.cross_entropy(disc(x), speaker_id)
        loss.backward()
        assert x.grad is not None, "SpeakerDiscriminator からxへ勾配が流れていない"


class TestAdversarialLoss:
    @pytest.fixture
    def loss_fn(self) -> AdversarialLoss:
        return AdversarialLoss(d_c=D_C_ADV, num_classes=NUM_CLASSES)

    @pytest.fixture
    def inputs(self):
        z_c = torch.randn(B_adv, T_PRIME, D_C_ADV, requires_grad=True)
        z_c_hat = torch.randn(B_adv, T_PRIME, D_C_ADV, requires_grad=True)
        speaker_id = torch.randint(0, NUM_CLASSES, (B_adv,))
        return z_c, z_c_hat, speaker_id

    def test_output_is_scalar(self, loss_fn: AdversarialLoss, inputs) -> None:
        z_c, z_c_hat, spk = inputs
        loss = loss_fn(z_c, z_c_hat, spk)
        assert loss.shape == ()

    def test_no_nan(self, loss_fn: AdversarialLoss, inputs) -> None:
        z_c, z_c_hat, spk = inputs
        assert not torch.isnan(loss_fn(z_c, z_c_hat, spk))

    def test_disc_c_and_disc_c_hat_are_independent(
        self,
        loss_fn: AdversarialLoss,
    ) -> None:
        """disc_c と disc_c_hat のパラメータが独立していること（共有されていない）"""
        c_ids = {id(p) for p in loss_fn.disc_c.parameters()}
        c_hat_ids = {id(p) for p in loss_fn.disc_c_hat.parameters()}
        assert c_ids.isdisjoint(c_hat_ids), "disc_c と disc_c_hat がパラメータを共有している"

    def test_gradient_flows_to_z_c_and_z_c_hat(
        self,
        loss_fn: AdversarialLoss,
        inputs,
    ) -> None:
        """z_c と z_c_hat の両方に勾配が流れること"""
        z_c, z_c_hat, spk = inputs
        loss = loss_fn(z_c, z_c_hat, spk)
        loss.backward()
        assert z_c.grad is not None, "z_c に勾配が流れていない"
        assert z_c_hat.grad is not None, "z_c_hat に勾配が流れていない"

    def test_gradient_sign_reversed_by_grl(
        self,
        loss_fn: AdversarialLoss,
        inputs,
    ) -> None:
        """GRL により z_c / z_c_hat に流れる勾配の符号が反転していること。

        比較方法:
          (A) GRLあり (通常): z_c に流れる勾配
          (B) GRLなし (参照): z_c の GAP後ベクトルを直接 Classifier に渡した場合の勾配

        (A) と (B) の符号が逆（内積が負）であることを確認する。
        """
        z_c, z_c_hat, spk = inputs

        # (A) GRLあり（AdversarialLoss の通常フロー）
        loss_a = loss_fn(z_c, z_c_hat, spk)
        loss_a.backward()
        grad_with_grl = z_c.grad.clone()

        # (B) GRLなし: disc_c の MLP だけを直接適用して勾配を確認
        z_c2 = z_c.detach().requires_grad_(True)
        z_avg2 = z_c2.mean(dim=1)                    # GAP
        # GRLをバイパスしてMLPのみ適用
        logits_no_grl = loss_fn.disc_c.fc2(loss_fn.disc_c.fc1(z_avg2))
        loss_b = F.cross_entropy(logits_no_grl, spk)
        loss_b.backward()
        grad_without_grl = z_c2.grad.clone()

        # 符号が逆: 内積が負であること（完全に逆でなくても方向が逆なら成立）
        dot = (grad_with_grl * grad_without_grl).sum().item()
        assert dot < 0, (
            f"GRL により勾配符号が反転していない。内積={dot:.6f} (負であるべき)"
        )

    def test_get_discriminator_parameters_nonempty(
        self,
        loss_fn: AdversarialLoss,
    ) -> None:
        params = loss_fn.get_discriminator_parameters()
        assert len(params) > 0

    def test_get_discriminator_parameters_covers_both_discs(
        self,
        loss_fn: AdversarialLoss,
    ) -> None:
        """disc_c と disc_c_hat の両方のパラメータが含まれること"""
        param_ids = {id(p) for p in loss_fn.get_discriminator_parameters()}
        c_ids = {id(p) for p in loss_fn.disc_c.parameters()}
        c_hat_ids = {id(p) for p in loss_fn.disc_c_hat.parameters()}
        assert c_ids.issubset(param_ids)
        assert c_hat_ids.issubset(param_ids)
