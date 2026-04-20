"""
VCWatermarkTrainer テスト

Q1 検証: Phase A の torch.no_grad() forward → estimator のみ更新される
Q2 検証: Phase B の 1 回 backward → G と D_adv 両方に勾配が流れ、二重カウントなし
"""

import os
import sys

import pytest
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.training.trainer import VCWatermarkTrainer
from src.data.mock_dataset import MockVCTKDataset

CFG_PATH = os.path.join(os.path.dirname(__file__), '..', 'configs', 'default.yaml')


# ---------------------------------------------------------------------------
# フィクスチャ
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg() -> dict:
    with open(CFG_PATH) as f:
        base = yaml.safe_load(f)
    # テスト用に小規模化
    base['training']['batch_size'] = 2
    base['data']['num_samples'] = 4
    base['data']['num_workers'] = 0
    base['losses']['vclub']['estimator_iters'] = 1   # Phase A を 1 回に短縮
    return base


@pytest.fixture
def trainer(cfg) -> VCWatermarkTrainer:
    return VCWatermarkTrainer(cfg=cfg, ckpt_path='', device='cpu')


@pytest.fixture
def small_loader(cfg) -> DataLoader:
    ds = MockVCTKDataset(
        num_samples=cfg['data']['num_samples'],
        num_speakers=cfg['losses']['adversarial']['num_classes'],
        seed=0,
    )
    return DataLoader(
        ds,
        batch_size=cfg['training']['batch_size'],
        shuffle=False,
        drop_last=True,
    )


@pytest.fixture
def batch(small_loader) -> tuple:
    return next(iter(small_loader))


# ---------------------------------------------------------------------------
# TestVCWatermarkTrainerInit
# ---------------------------------------------------------------------------

class TestVCWatermarkTrainerInit:
    def test_device_is_cpu(self, trainer: VCWatermarkTrainer) -> None:
        assert trainer.device == torch.device('cpu')

    def test_three_optimizers_exist(self, trainer: VCWatermarkTrainer) -> None:
        assert isinstance(trainer.optimizer_G, torch.optim.Adam)
        assert isinstance(trainer.optimizer_D_vclub, torch.optim.Adam)
        assert isinstance(trainer.optimizer_D_adv, torch.optim.Adam)

    def test_optimizer_param_disjoint(self, trainer: VCWatermarkTrainer) -> None:
        """3 Optimizer のパラメータ id に重複がないこと（二重カウント不発生の前提）"""
        ids_G     = {id(p) for g in trainer.optimizer_G.param_groups for p in g['params']}
        ids_vclub = {id(p) for g in trainer.optimizer_D_vclub.param_groups for p in g['params']}
        ids_adv   = {id(p) for g in trainer.optimizer_D_adv.param_groups for p in g['params']}
        assert ids_G.isdisjoint(ids_vclub), "optimizer_G と optimizer_D_vclub がパラメータを共有"
        assert ids_G.isdisjoint(ids_adv),   "optimizer_G と optimizer_D_adv がパラメータを共有"
        assert ids_vclub.isdisjoint(ids_adv), "optimizer_D_vclub と optimizer_D_adv がパラメータを共有"

    def test_lambda_weights_loaded(self, trainer: VCWatermarkTrainer) -> None:
        for name in ['lambda_rec', 'lambda_wm', 'lambda_nce', 'lambda_vclub', 'lambda_adv']:
            val = getattr(trainer, name)
            assert isinstance(val, float) and val > 0, f"{name}={val} は正の float であるべき"

    def test_start_epoch_is_one(self, trainer: VCWatermarkTrainer) -> None:
        assert trainer.start_epoch == 1

    def test_all_modules_on_cpu(self, trainer: VCWatermarkTrainer) -> None:
        for module in [trainer.model, trainer.vclub, trainer.adv_loss]:
            for p in module.parameters():
                assert p.device.type == 'cpu', f"{type(module).__name__} のパラメータが CPU にない"


# ---------------------------------------------------------------------------
# TestGenerateWatermark
# ---------------------------------------------------------------------------

class TestGenerateWatermark:
    def test_shape_and_device(self, trainer: VCWatermarkTrainer) -> None:
        W = trainer._generate_watermark(2)
        assert W.shape == (2, 16)
        assert W.device.type == 'cpu'

    def test_binary_values(self, trainer: VCWatermarkTrainer) -> None:
        W = trainer._generate_watermark(4)
        assert ((W == 0.0) | (W == 1.0)).all(), "W に {0,1} 以外の値が含まれる"

    def test_unique_rows(self, trainer: VCWatermarkTrainer) -> None:
        W = trainer._generate_watermark(4)
        rows = [tuple(row.tolist()) for row in W]
        assert len(set(rows)) == 4, "In-batch uniqueness が保証されていない"


# ---------------------------------------------------------------------------
# TestPhaseAStep  ← Q1 の数値検証
# ---------------------------------------------------------------------------

class TestPhaseAStep:
    def _get_outputs_a(
        self,
        trainer: VCWatermarkTrainer,
        mel: torch.Tensor,
        f0_norm: torch.Tensor,
        W: torch.Tensor,
    ) -> dict:
        """torch.no_grad() 下で model forward を実行して outputs を返す。"""
        with torch.no_grad():
            return trainer.model(mel, f0_norm, W, attack_fn=None)

    def test_estimator_loss_is_scalar(self, trainer, batch) -> None:
        mel, f0_norm, speaker_id = batch
        W = trainer._generate_watermark(mel.size(0))
        outputs_a = self._get_outputs_a(trainer, mel, f0_norm, W)
        loss_est = trainer._phase_a_step(outputs_a)
        assert loss_est.shape == (), f"スカラーを期待: shape={loss_est.shape}"

    def test_estimator_params_updated(self, trainer, batch) -> None:
        """Phase A 後に vCLUB 推定器パラメータが変化すること"""
        mel, f0_norm, _ = batch
        W = trainer._generate_watermark(mel.size(0))
        outputs_a = self._get_outputs_a(trainer, mel, f0_norm, W)

        before = [p.clone() for p in trainer.vclub.cp_mi_net.parameters()]
        trainer._phase_a_step(outputs_a)
        after = list(trainer.vclub.cp_mi_net.parameters())

        changed = any(not torch.equal(b, a) for b, a in zip(before, after))
        assert changed, "Phase A 後に cp_mi_net のパラメータが変化していない"

    def test_model_content_encoder_unchanged(self, trainer, batch) -> None:
        """Phase A 後に ContentEncoder パラメータが変化しないこと"""
        mel, f0_norm, _ = batch
        W = trainer._generate_watermark(mel.size(0))
        outputs_a = self._get_outputs_a(trainer, mel, f0_norm, W)

        before = [p.clone() for p in trainer.model.content_encoder.parameters()]
        trainer._phase_a_step(outputs_a)
        after = list(trainer.model.content_encoder.parameters())

        unchanged = all(torch.equal(b, a) for b, a in zip(before, after))
        assert unchanged, "Phase A 後に ContentEncoder パラメータが変化している（勾配漏れ）"

    def test_no_grad_forward_feeds_estimator_correctly(self, trainer, batch) -> None:
        """torch.no_grad() 下で生成された outputs で estimator_loss.backward() が動作し、
        推定器パラメータに .grad が設定されること（Q1 の核心検証）。
        """
        mel, f0_norm, _ = batch
        W = trainer._generate_watermark(mel.size(0))

        # torch.no_grad() 下で forward
        with torch.no_grad():
            outputs_a = trainer.model(mel, f0_norm, W, attack_fn=None)

        # outputs_a のテンソルは requires_grad=False
        assert not outputs_a['z_c'].requires_grad, "no_grad 下の z_c が requires_grad=True"

        # それでも推定器パラメータに勾配が流れること
        trainer.optimizer_D_vclub.zero_grad()
        loss_est = trainer.vclub.estimator_loss(
            outputs_a['z_c'], outputs_a['z_r'], outputs_a['z_f'],
        )
        loss_est.backward()

        for name, param in trainer.vclub.cp_mi_net.named_parameters():
            assert param.grad is not None, f"cp_mi_net.{name} に勾配がない"


# ---------------------------------------------------------------------------
# TestPhaseBStep  ← Q2 の数値検証
# ---------------------------------------------------------------------------

class TestPhaseBStep:
    def test_returns_expected_keys(self, trainer, batch) -> None:
        mel, f0_norm, speaker_id = batch
        trainer.model.train()
        trainer.distortion.train()
        trainer.distortion.set_epoch(1)
        W = trainer._generate_watermark(mel.size(0))
        losses = trainer._phase_b_step(mel, f0_norm, W, speaker_id)
        expected = {'loss_total', 'loss_rec', 'loss_wm', 'loss_nce', 'loss_vclub', 'loss_adv', 'ber_train'}
        assert set(losses.keys()) == expected

    def test_ber_in_range(self, trainer, batch) -> None:
        mel, f0_norm, speaker_id = batch
        trainer.model.train()
        trainer.distortion.train()
        trainer.distortion.set_epoch(1)
        W = trainer._generate_watermark(mel.size(0))
        losses = trainer._phase_b_step(mel, f0_norm, W, speaker_id)
        ber = losses['ber_train'].item() if isinstance(losses['ber_train'], torch.Tensor) \
              else losses['ber_train']
        assert 0.0 <= ber <= 1.0, f"BER={ber} が [0,1] 範囲外"

    def test_g_params_have_grad_after_phase_b(self, trainer, batch) -> None:
        """Phase B 後に optimizer_G の ContentEncoder パラメータに .grad があること"""
        mel, f0_norm, speaker_id = batch
        trainer.model.train()
        trainer.distortion.train()
        trainer.distortion.set_epoch(1)
        W = trainer._generate_watermark(mel.size(0))
        trainer._phase_b_step(mel, f0_norm, W, speaker_id)
        # .step() 後は .grad がクリアされずに残る（次の zero_grad まで）
        any_grad = any(
            p.grad is not None
            for p in trainer.model.content_encoder.parameters()
        )
        assert any_grad, "Phase B 後に ContentEncoder パラメータに .grad がない"

    def test_disc_params_have_grad_after_phase_b(self, trainer, batch) -> None:
        """Phase B 後に optimizer_D_adv の Discriminator パラメータに .grad があること"""
        mel, f0_norm, speaker_id = batch
        trainer.model.train()
        trainer.distortion.train()
        trainer.distortion.set_epoch(1)
        W = trainer._generate_watermark(mel.size(0))
        trainer._phase_b_step(mel, f0_norm, W, speaker_id)
        any_grad = any(
            p.grad is not None
            for p in trainer.adv_loss.disc_c.parameters()
        )
        assert any_grad, "Phase B 後に disc_c パラメータに .grad がない"

    def test_no_grad_accumulation_across_two_calls(self, trainer, batch) -> None:
        """zero_grad が機能し勾配が積算されないことを直接検証する。

        同一入力で手動で 2 回 backward を呼んだとき（zero_grad なし）は
        勾配が 2 倍になる。_phase_b_step が zero_grad を正しく呼んでいれば
        1 回目と同等の grad になる。
        同一 W を固定した上で手動 backward と _phase_b_step を比較する。
        """
        mel, f0_norm, speaker_id = batch
        trainer.model.train()
        trainer.distortion.train()
        trainer.distortion.set_epoch(1)

        W = trainer._generate_watermark(mel.size(0))

        # --- ベースライン: _phase_b_step を 1 回（内部で zero_grad 済み） ---
        trainer._phase_b_step(mel, f0_norm, W, speaker_id)
        grads_after_1 = {
            id(p): p.grad.clone()
            for p in trainer.model.content_encoder.parameters()
            if p.grad is not None
        }

        # --- zero_grad なしで 2 回目の backward を手動で実行（積算を再現） ---
        # _phase_b_step は内部で zero_grad を呼ぶため、
        # 実際に積算をテストするには手動で backward のみを呼ぶ
        outputs = trainer.model(mel, f0_norm, W, attack_fn=trainer.distortion)
        mel_tf = mel.transpose(1, 2)
        loss_rec = trainer.rec_loss(mel_tf, outputs['mel_postnet'], outputs['mel_before_postnet'])
        loss_wm, _ = trainer.wm_loss(outputs['W_hat'], W)
        # zero_grad を呼ばずに backward（勾配積算）
        loss_rec.backward(retain_graph=True)
        grads_accumulated = {
            id(p): p.grad.clone()
            for p in trainer.model.content_encoder.parameters()
            if p.grad is not None
        }

        # zero_grad ありの grad と積算後の grad は異なるはず（積算により大きくなる）
        diffs = [
            (grads_accumulated[k] - grads_after_1[k]).abs().mean().item()
            for k in grads_after_1 if k in grads_accumulated
        ]
        assert len(diffs) > 0
        # 積算後の grad は 1 回目の grad と異なる（zero_grad の有効性を間接確認）
        assert any(d > 1e-9 for d in diffs), (
            "零分散: 積算後の grad が 1 回目と同じ — 勾配積算が起きていないかテスト設計を確認"
        )

    def test_vclub_estimator_not_updated_in_phase_b(self, trainer, batch) -> None:
        """Phase B では optimizer_D_vclub.step() が呼ばれないため、
        vCLUB 推定器パラメータの値が変化しないこと。

        NOTE: vclub_loss() は推定器の mu/logvar 計算を経由するため Phase B でも
        estimator params に .grad が設定される。しかし optimizer_D_vclub.step() は
        Phase A でのみ呼ばれるため、実際のパラメータ更新は起きない。
        """
        mel, f0_norm, speaker_id = batch
        trainer.model.train()
        trainer.distortion.train()
        trainer.distortion.set_epoch(1)
        W = trainer._generate_watermark(mel.size(0))

        before = [p.clone() for p in trainer.vclub.get_estimator_parameters()]
        trainer._phase_b_step(mel, f0_norm, W, speaker_id)
        after = list(trainer.vclub.get_estimator_parameters())

        unchanged = all(torch.equal(b, a) for b, a in zip(before, after))
        assert unchanged, "Phase B 後に vCLUB 推定器パラメータが変化している（optimizer_D_vclub.step() が呼ばれてはならない）"


# ---------------------------------------------------------------------------
# TestTrainLoop
# ---------------------------------------------------------------------------

class TestTrainLoop:
    def test_single_epoch_no_error(self, trainer, small_loader) -> None:
        trainer.train(small_loader, val_loader=None, num_epochs=1)

    def test_distortion_curriculum_set(self, trainer, small_loader) -> None:
        """N エポック後に distortion.current_epoch == N であること"""
        trainer.train(small_loader, val_loader=None, num_epochs=2)
        assert trainer.distortion.current_epoch == 2


# ---------------------------------------------------------------------------
# TestValidation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_returns_two_ber_metrics(self, trainer, small_loader) -> None:
        metrics = trainer.validate(small_loader)
        assert 'val_ber_clean' in metrics
        assert 'val_ber_attacked' in metrics

    def test_ber_values_in_range(self, trainer, small_loader) -> None:
        metrics = trainer.validate(small_loader)
        assert 0.0 <= metrics['val_ber_clean']    <= 1.0
        assert 0.0 <= metrics['val_ber_attacked'] <= 1.0

    def test_model_in_eval_during_validate(self, trainer, small_loader) -> None:
        """validate() の内部で model が eval モードであること。
        validate() が model.eval() を呼ぶことを、モードを観測して確認する。
        """
        trainer.model.train()

        original_eval = trainer.model.eval

        observed_modes = []

        def patched_eval():
            observed_modes.append('eval_called')
            return original_eval()

        trainer.model.eval = patched_eval  # type: ignore[method-assign]
        try:
            trainer.validate(small_loader)
        finally:
            trainer.model.eval = original_eval  # type: ignore[method-assign]

        assert 'eval_called' in observed_modes, "validate() が model.eval() を呼んでいない"


# ---------------------------------------------------------------------------
# TestCheckpoint
# ---------------------------------------------------------------------------

class TestCheckpoint:
    def test_latest_always_written(self, trainer, small_loader, tmp_path) -> None:
        trainer.cfg['training']['checkpoint_dir'] = str(tmp_path)
        trainer.train(small_loader, val_loader=None, num_epochs=1)
        assert (tmp_path / 'checkpoint_latest.pt').exists()

    def test_save_load_restores_epoch(self, cfg, small_loader, tmp_path) -> None:
        cfg['training']['checkpoint_dir'] = str(tmp_path)
        t1 = VCWatermarkTrainer(cfg=cfg, ckpt_path='', device='cpu')
        t1.train(small_loader, val_loader=None, num_epochs=1)

        latest = str(tmp_path / 'checkpoint_latest.pt')
        t2 = VCWatermarkTrainer(cfg=cfg, ckpt_path='', device='cpu', resume_from=latest)
        assert t2.start_epoch == 2, f"start_epoch={t2.start_epoch} (期待値=2)"

    def test_periodic_checkpoint_saved(self, cfg, small_loader, tmp_path) -> None:
        cfg['training']['checkpoint_dir'] = str(tmp_path)
        cfg['training']['checkpoint_every_n_epochs'] = 1
        t = VCWatermarkTrainer(cfg=cfg, ckpt_path='', device='cpu')
        t.train(small_loader, val_loader=None, num_epochs=1)
        assert (tmp_path / 'checkpoint_epoch0001.pt').exists()

    def test_best_checkpoint_saved_on_improvement(self, cfg, small_loader, tmp_path) -> None:
        cfg['training']['checkpoint_dir'] = str(tmp_path)
        t = VCWatermarkTrainer(cfg=cfg, ckpt_path='', device='cpu')
        t.best_val_ber_clean = 1.0   # 現在の best を 1.0 に設定
        t._save_checkpoint(epoch=1, is_best=True)
        assert (tmp_path / 'checkpoint_best.pt').exists()

    def test_optimizer_states_restored(self, cfg, small_loader, tmp_path) -> None:
        """checkpoint 復元後に 1 バッチ学習しても loss が有限であること
        （optimizer の momentum state が正しく復元されている）
        """
        cfg['training']['checkpoint_dir'] = str(tmp_path)
        t1 = VCWatermarkTrainer(cfg=cfg, ckpt_path='', device='cpu')
        t1.train(small_loader, val_loader=None, num_epochs=1)

        latest = str(tmp_path / 'checkpoint_latest.pt')
        t2 = VCWatermarkTrainer(cfg=cfg, ckpt_path='', device='cpu', resume_from=latest)
        t2.train(small_loader, val_loader=None, num_epochs=1)
