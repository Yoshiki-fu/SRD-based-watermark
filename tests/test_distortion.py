"""
Shape / 動作テスト: src/attacks/distortion.py

DistortionLayer の shape 一貫性、カリキュラム動作（identity / 攻撃適用）、
eval mode identity、勾配伝搬を検証する。
"""

import os
import sys

import pytest
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.attacks.distortion import DistortionLayer

CFG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default.yaml')

B = 4
T = 192
F_MELS = 80


@pytest.fixture
def cfg():
    with open(CFG_PATH) as f:
        return yaml.safe_load(f)['distortion']


@pytest.fixture
def layer(cfg):
    return DistortionLayer(cfg).train()


@pytest.fixture
def mel():
    return torch.randn(B, F_MELS, T)


# ---------------------------------------------------------------------------
# Shape 一貫性
# ---------------------------------------------------------------------------

class TestDistortionLayerShape:
    def test_clean_epoch_shape(self, layer: DistortionLayer, mel: torch.Tensor) -> None:
        layer.set_epoch(1)
        assert layer(mel).shape == (B, F_MELS, T)

    def test_light_epoch_shape(self, layer: DistortionLayer, mel: torch.Tensor) -> None:
        layer.set_epoch(6)
        assert layer(mel).shape == (B, F_MELS, T)

    def test_heavy_epoch_shape(self, layer: DistortionLayer, mel: torch.Tensor) -> None:
        layer.set_epoch(16)
        assert layer(mel).shape == (B, F_MELS, T)

    def test_boundary_epochs_shape(self, layer: DistortionLayer, mel: torch.Tensor) -> None:
        for ep in [5, 6, 15, 16]:
            layer.set_epoch(ep)
            out = layer(mel)
            assert out.shape == (B, F_MELS, T), f"epoch {ep}: shape mismatch"

    def test_individual_attacks_preserve_shape(
        self, layer: DistortionLayer, mel: torch.Tensor
    ) -> None:
        """各攻撃メソッドが単独で shape を保持すること"""
        for name, fn in [
            ('gaussian_noise', layer._gaussian_noise),
            ('dropout',        layer._dropout),
            ('time_mask',      layer._time_mask),
            ('freq_mask',      layer._freq_mask),
            ('resample',       layer._resample),
        ]:
            out = fn(mel)
            assert out.shape == (B, F_MELS, T), f"{name}: shape mismatch"


# ---------------------------------------------------------------------------
# カリキュラム動作
# ---------------------------------------------------------------------------

class TestDistortionLayerCurriculum:
    def test_set_epoch_0_is_identity(self, layer: DistortionLayer, mel: torch.Tensor) -> None:
        """set_epoch(0) は identity を返すこと（数値的に一致）"""
        layer.set_epoch(0)
        assert torch.equal(layer(mel), mel), "epoch=0 は identity であるべき"

    def test_clean_phase_is_identity(self, layer: DistortionLayer, mel: torch.Tensor) -> None:
        """clean フェーズ（epoch <= clean_end=5）は identity を返すこと"""
        for ep in [1, 3, 5]:
            layer.set_epoch(ep)
            assert torch.equal(layer(mel), mel), f"epoch={ep} は identity であるべき"

    def test_light_phase_applies_attack(
        self, layer: DistortionLayer, mel: torch.Tensor
    ) -> None:
        """light フェーズ（epoch 6-15）で少なくとも 1 回攻撃が適用されること。

        apply_prob=0.8 の確率的適用のため複数回試行し、いずれかで変化を確認する。
        """
        layer.set_epoch(6)
        changed = any(not torch.equal(layer(mel), mel) for _ in range(20))
        assert changed, "epoch=6 で 20 回試行しても攻撃が適用されなかった"

    def test_heavy_phase_applies_attack(
        self, layer: DistortionLayer, mel: torch.Tensor
    ) -> None:
        """heavy フェーズ（epoch 16+）で少なくとも 1 回攻撃が適用されること。"""
        layer.set_epoch(16)
        changed = any(not torch.equal(layer(mel), mel) for _ in range(10))
        assert changed, "epoch=16 で 10 回試行しても攻撃が適用されなかった"


# ---------------------------------------------------------------------------
# eval mode
# ---------------------------------------------------------------------------

class TestDistortionLayerEvalMode:
    def test_eval_epoch1_is_identity(self, layer: DistortionLayer, mel: torch.Tensor) -> None:
        layer.eval()
        layer.set_epoch(1)
        assert torch.equal(layer(mel), mel)

    def test_eval_epoch16_is_identity(self, layer: DistortionLayer, mel: torch.Tensor) -> None:
        """eval mode + epoch 16 でも identity を返すこと"""
        layer.eval()
        layer.set_epoch(16)
        assert torch.equal(layer(mel), mel), "eval mode は常に identity であるべき"

    def test_train_mode_restores_attack(
        self, layer: DistortionLayer, mel: torch.Tensor
    ) -> None:
        """eval→train の切り替えで攻撃が復活すること"""
        layer.set_epoch(16)
        layer.eval()
        assert torch.equal(layer(mel), mel), "eval mode で identity でない"
        layer.train()
        changed = any(not torch.equal(layer(mel), mel) for _ in range(10))
        assert changed, "train mode に戻っても攻撃が適用されない"


# ---------------------------------------------------------------------------
# 勾配伝搬
# ---------------------------------------------------------------------------

class TestDistortionLayerGradient:
    def test_gradient_flows_through_noise(self, layer: DistortionLayer, mel: torch.Tensor) -> None:
        mel_req = mel.clone().requires_grad_(True)
        out = layer._gaussian_noise(mel_req)
        out.sum().backward()
        assert mel_req.grad is not None
        assert not torch.all(mel_req.grad == 0)

    def test_gradient_flows_through_time_mask(
        self, layer: DistortionLayer, mel: torch.Tensor
    ) -> None:
        mel_req = mel.clone().requires_grad_(True)
        out = layer._time_mask(mel_req)
        out.sum().backward()
        assert mel_req.grad is not None
        assert not torch.all(mel_req.grad == 0)

    def test_gradient_flows_through_freq_mask(
        self, layer: DistortionLayer, mel: torch.Tensor
    ) -> None:
        mel_req = mel.clone().requires_grad_(True)
        out = layer._freq_mask(mel_req)
        out.sum().backward()
        assert mel_req.grad is not None
        assert not torch.all(mel_req.grad == 0)

    def test_gradient_flows_through_resample(
        self, layer: DistortionLayer, mel: torch.Tensor
    ) -> None:
        mel_req = mel.clone().requires_grad_(True)
        out = layer._resample(mel_req)
        out.sum().backward()
        assert mel_req.grad is not None
        assert not torch.all(mel_req.grad == 0)

    def test_no_nan_gradient_heavy_phase(
        self, layer: DistortionLayer, mel: torch.Tensor
    ) -> None:
        """heavy フェーズで全攻撃を適用したとき勾配に NaN が含まれないこと"""
        mel_req = mel.clone().requires_grad_(True)
        layer.set_epoch(16)
        layer.apply_prob = 1.0
        out = layer(mel_req)
        out.sum().backward()
        if mel_req.grad is not None:
            assert not torch.isnan(mel_req.grad).any()


# ---------------------------------------------------------------------------
# NaN チェック
# ---------------------------------------------------------------------------

class TestDistortionLayerNoNan:
    def test_no_nan_clean(self, layer: DistortionLayer, mel: torch.Tensor) -> None:
        layer.set_epoch(1)
        assert not torch.isnan(layer(mel)).any()

    def test_no_nan_light(self, layer: DistortionLayer, mel: torch.Tensor) -> None:
        layer.set_epoch(6)
        assert not torch.isnan(layer(mel)).any()

    def test_no_nan_heavy(self, layer: DistortionLayer, mel: torch.Tensor) -> None:
        layer.set_epoch(16)
        assert not torch.isnan(layer(mel)).any()
