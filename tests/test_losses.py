"""
Shape・挙動テスト: src/losses/reconstruction.py & watermark_loss.py
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.losses.reconstruction import ReconstructionLoss
from src.losses.watermark_loss import WatermarkLoss

B = 2
T = 192
MEL = 80
N_BITS = 16


# ---------------------------------------------------------------------------
# ReconstructionLoss
# ---------------------------------------------------------------------------

class TestReconstructionLoss:
    @pytest.fixture
    def loss_fn(self) -> ReconstructionLoss:
        return ReconstructionLoss(aux_weight=1.0)

    def test_zero_loss_on_identical_inputs(self, loss_fn: ReconstructionLoss) -> None:
        """同一テンソルを渡したとき loss = 0 であること"""
        mel = torch.randn(B, T, MEL)
        loss = loss_fn(mel, mel, mel)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_loss_on_different_inputs(self, loss_fn: ReconstructionLoss) -> None:
        """異なるテンソルを渡したとき loss > 0 であること"""
        target = torch.randn(B, T, MEL)
        postnet = torch.randn(B, T, MEL)
        before = torch.randn(B, T, MEL)
        loss = loss_fn(target, postnet, before)
        assert loss.item() > 0.0

    def test_aux_weight_zero(self) -> None:
        """aux_weight=0.0 のとき postnet の L1 のみが残ること"""
        loss_fn = ReconstructionLoss(aux_weight=0.0)
        target = torch.zeros(B, T, MEL)
        postnet = torch.ones(B, T, MEL)
        before = torch.full((B, T, MEL), 100.0)  # 大きな値だが無視されるはず

        loss = loss_fn(target, postnet, before)
        expected = torch.ones(B, T, MEL).mean()  # L1(zeros, ones) = 1.0
        assert loss.item() == pytest.approx(expected.item(), rel=1e-5)

    def test_aux_weight_scaling(self) -> None:
        """aux_weight が補助 loss のスケールを正しく変えること"""
        target = torch.zeros(B, T, MEL)
        postnet = torch.zeros(B, T, MEL)       # postnet loss = 0
        before = torch.ones(B, T, MEL)         # before loss = 1.0

        loss_w1 = ReconstructionLoss(aux_weight=1.0)(target, postnet, before)
        loss_w2 = ReconstructionLoss(aux_weight=2.0)(target, postnet, before)
        assert loss_w2.item() == pytest.approx(2.0 * loss_w1.item(), rel=1e-5)

    def test_gradient_flows_to_postnet(self, loss_fn: ReconstructionLoss) -> None:
        """mel_postnet から勾配が流れること"""
        target = torch.randn(B, T, MEL)
        postnet = torch.randn(B, T, MEL, requires_grad=True)
        before = torch.randn(B, T, MEL)
        loss = loss_fn(target, postnet, before)
        loss.backward()
        assert postnet.grad is not None and not torch.all(postnet.grad == 0)

    def test_gradient_flows_to_before_postnet(self, loss_fn: ReconstructionLoss) -> None:
        """mel_before_postnet から勾配が流れること"""
        target = torch.randn(B, T, MEL)
        postnet = torch.randn(B, T, MEL)
        before = torch.randn(B, T, MEL, requires_grad=True)
        loss = loss_fn(target, postnet, before)
        loss.backward()
        assert before.grad is not None and not torch.all(before.grad == 0)

    def test_output_is_scalar(self, loss_fn: ReconstructionLoss) -> None:
        target = torch.randn(B, T, MEL)
        loss = loss_fn(target, torch.randn(B, T, MEL), torch.randn(B, T, MEL))
        assert loss.shape == torch.Size([])


# ---------------------------------------------------------------------------
# WatermarkLoss
# ---------------------------------------------------------------------------

class TestWatermarkLoss:
    @pytest.fixture
    def loss_fn(self) -> WatermarkLoss:
        return WatermarkLoss()

    def _perfect_logits(self, W: torch.Tensor) -> torch.Tensor:
        """W={0,1} に対して sigmoid(logit) がほぼ W になる大きな logit を返す"""
        return (W * 2 - 1) * 10.0  # W=1 → logit=+10, W=0 → logit=-10

    def test_output_types(self, loss_fn: WatermarkLoss) -> None:
        """(loss, ber) のタプルを返すこと"""
        W_hat = torch.randn(B, N_BITS)
        W = (torch.rand(B, N_BITS) > 0.5).float()
        result = loss_fn(W_hat, W)
        assert isinstance(result, tuple) and len(result) == 2

    def test_loss_positive(self, loss_fn: WatermarkLoss) -> None:
        """BCE loss > 0 であること"""
        W_hat = torch.randn(B, N_BITS)
        W = (torch.rand(B, N_BITS) > 0.5).float()
        loss, _ = loss_fn(W_hat, W)
        assert loss.item() > 0.0

    def test_ber_range(self, loss_fn: WatermarkLoss) -> None:
        """BER が [0, 1] に収まること"""
        W_hat = torch.randn(B, N_BITS)
        W = (torch.rand(B, N_BITS) > 0.5).float()
        _, ber = loss_fn(W_hat, W)
        assert 0.0 <= ber.item() <= 1.0

    def test_ber_zero_on_perfect_prediction(self, loss_fn: WatermarkLoss) -> None:
        """完全に正しい予測のとき BER = 0.0 であること"""
        W = (torch.rand(B, N_BITS) > 0.5).float()
        W_hat = self._perfect_logits(W)
        _, ber = loss_fn(W_hat, W)
        assert ber.item() == pytest.approx(0.0, abs=1e-6)

    def test_ber_one_on_all_wrong_prediction(self, loss_fn: WatermarkLoss) -> None:
        """全ビット誤りのとき BER = 1.0 であること"""
        W = (torch.rand(B, N_BITS) > 0.5).float()
        W_inverted = 1.0 - W
        W_hat = self._perfect_logits(W_inverted)
        _, ber = loss_fn(W_hat, W)
        assert ber.item() == pytest.approx(1.0, abs=1e-6)

    def test_loss_has_gradient(self, loss_fn: WatermarkLoss) -> None:
        """loss に勾配が流れること"""
        W_hat = torch.randn(B, N_BITS, requires_grad=True)
        W = (torch.rand(B, N_BITS) > 0.5).float()
        loss, _ = loss_fn(W_hat, W)
        loss.backward()
        assert W_hat.grad is not None and not torch.all(W_hat.grad == 0)

    def test_ber_has_no_gradient(self, loss_fn: WatermarkLoss) -> None:
        """BER に勾配が流れないこと（評価専用）"""
        W_hat = torch.randn(B, N_BITS, requires_grad=True)
        W = (torch.rand(B, N_BITS) > 0.5).float()
        _, ber = loss_fn(W_hat, W)
        assert not ber.requires_grad

    def test_output_scalars(self, loss_fn: WatermarkLoss) -> None:
        """loss と ber がどちらもスカラーであること"""
        W_hat = torch.randn(B, N_BITS)
        W = (torch.rand(B, N_BITS) > 0.5).float()
        loss, ber = loss_fn(W_hat, W)
        assert loss.shape == torch.Size([])
        assert ber.shape == torch.Size([])

    def test_logits_accepted(self, loss_fn: WatermarkLoss) -> None:
        """sigmoid 前の logits を受け取っても NaN が出ないこと"""
        W_hat = torch.randn(B, N_BITS) * 10  # 大きな logits
        W = (torch.rand(B, N_BITS) > 0.5).float()
        loss, ber = loss_fn(W_hat, W)
        assert not torch.isnan(loss)
        assert not torch.isnan(ber)


# ---------------------------------------------------------------------------
# __init__.py から import できること
# ---------------------------------------------------------------------------

class TestPackageInit:
    def test_imports(self) -> None:
        from src.losses import ReconstructionLoss, WatermarkLoss
        assert ReconstructionLoss is not None
        assert WatermarkLoss is not None
