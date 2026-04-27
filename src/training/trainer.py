"""
VC-Robust Neural Audio Watermarking — Trainer

3 Optimizer (Optimizer_G / Optimizer_D_vclub / Optimizer_D_adv) を用いた
2フェーズ学習ループを実装する。

Phase A: vCLUB MI 推定器の先行更新（ContentEncoder への勾配なし）
Phase B: Generator + Adversarial Discriminator の同時更新（1回の backward）

更新順序の根拠は以下の通り:
  - vCLUB と Adversarial は内部の勾配力学が異なるため Optimizer を分離
  - Phase A で torch.no_grad() によりメモリを節約しつつ推定器のみ更新
  - Phase B で 1回の loss_total.backward() が GRL の符号分配を含む全勾配を計算し、
    disjoint な 2 Optimizer が各自の .grad を読んで更新（二重カウントなし）
"""

import logging
import os
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.models.full_model import VCWatermarkModel
from src.models.watermark import BypassExtractor
from src.losses.reconstruction import ReconstructionLoss
from src.losses.watermark_loss import WatermarkLoss
from src.losses.info_nce import InfoNCELoss
from src.losses.vclub import VCLUBLoss
from src.losses.adversarial import AdversarialLoss
from src.attacks.distortion import DistortionLayer


def _resolve_device(device_arg: Optional[str]) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _resolve_ckpt_path(cfg: dict, explicit: Optional[str]) -> Optional[str]:
    """ckpt_path の解決。優先順位: 明示的引数 > cfg > None。
    空文字列は None として扱う（テスト用）。
    """
    if explicit is not None:
        return explicit if explicit != '' else None
    path = cfg.get('training', {}).get('pretrained_ckpt', None)
    return path if path else None


class VCWatermarkTrainer:
    """Phase 1 全モジュール統合トレーナー。

    Args:
        cfg:         default.yaml を yaml.safe_load した dict
        ckpt_path:   800000-G.ckpt への明示的パス。
                     None → cfg['training']['pretrained_ckpt'] を使用。
                     '' (空文字) → ランダム初期化（テスト用）。
        resume_from: checkpoint ファイルパス。指定時は学習を再開。
        device:      'cuda' / 'cpu' / 'cuda:0' 等。None なら自動検出。
    """

    def __init__(
        self,
        cfg: dict,
        ckpt_path: Optional[str] = None,
        resume_from: Optional[str] = None,
        device: Optional[str] = None,
        fixed_watermark_seed: Optional[int] = None,
        bypass_decoder: bool = False,
        boost_watermark_lr: Optional[float] = None,
        disable_vclub_adv: bool = False,
        load_pretrained_extractor: bool = False,
        freeze_extractor_content_encoder: bool = False,
    ) -> None:
        self.cfg = cfg
        self.device = _resolve_device(device)
        self._fixed_watermark_seed = fixed_watermark_seed
        self.bypass_decoder = bypass_decoder
        self.disable_vclub_adv = disable_vclub_adv
        self.load_pretrained_extractor = load_pretrained_extractor
        self.freeze_extractor_content_encoder = freeze_extractor_content_encoder
        self._setup_logger()

        resolved_ckpt = _resolve_ckpt_path(cfg, ckpt_path)

        # モデル構築
        self.model = self._build_model(resolved_ckpt)

        # 診断用 BypassExtractor（--bypass_decoder 時のみ）
        if bypass_decoder:
            enc = cfg.get('encoder', {})
            wm  = cfg.get('watermark', {})
            self.bypass_extractor = BypassExtractor(
                dim_c=enc.get('dim_neck', 8) * 2,
                num_bits=wm.get('num_bits', 16),
                mlp_hidden=wm.get('extractor_mlp_hidden', 32),
            ).to(self.device)

        # Loss 構築
        self._build_losses(resolved_ckpt)

        # 3 Optimizer 構築（boost_watermark_lr を反映）
        self._build_optimizers(boost_watermark_lr=boost_watermark_lr)

        # 学習再開カウンタ
        self.start_epoch: int = 1
        self.best_val_ber_clean: float = float('inf')

        # val 用専用攻撃レイヤー（常に heavy phase, train モード固定）
        # これにより validate() 内でのモード切り替えトグルを避ける
        self.val_distortion = DistortionLayer(cfg.get('distortion', {})).to(self.device)
        self.val_distortion.set_epoch(999)   # heavy phase 固定
        self.val_distortion.train()

        # Lambda 重みを instance variable に格納（hot loop での dict lookup を避ける）
        lc = cfg.get('losses', {})
        self.lambda_rec   = lc.get('reconstruction', {}).get('weight', 1.0)
        self.lambda_wm    = lc.get('watermark', {}).get('weight', 1.0)
        self.lambda_nce   = lc.get('info_nce', {}).get('weight', 1.0)
        self.lambda_vclub = lc.get('vclub', {}).get('weight', 0.01)
        self.lambda_adv   = lc.get('adversarial', {}).get('weight', 0.1)

        if resume_from is not None:
            self._load_checkpoint(resume_from)

        self.logger.info(
            f"Trainer initialized on {self.device}. "
            f"start_epoch={self.start_epoch}"
        )

    # ------------------------------------------------------------------
    # 初期化ヘルパー
    # ------------------------------------------------------------------

    def _setup_logger(self) -> None:
        self.logger = logging.getLogger('VCWatermarkTrainer')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '[%(asctime)s][%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
            ))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _build_model(self, ckpt_path: Optional[str]) -> VCWatermarkModel:
        enc = self.cfg.get('encoder', {})
        spk = self.cfg.get('speaker_encoder', {})
        dec = self.cfg.get('decoder', {})
        wm  = self.cfg.get('watermark', {})
        model = VCWatermarkModel(
            dim_freq=enc.get('dim_freq', 80),
            dim_f0=enc.get('dim_f0', 257),
            dim_spk_emb=enc.get('dim_spk_emb', 256),
            max_len_pad=enc.get('max_len_pad', 192),
            chs_grp=enc.get('chs_grp', 16),
            dim_enc=enc.get('dim_enc', 512),
            dim_neck=enc.get('dim_neck', 8),
            freq=enc.get('freq', 8),
            content_n_conv=enc.get('content_n_conv', 3),
            content_lstm_layers=enc.get('content_lstm_layers', 2),
            dim_enc_2=enc.get('dim_enc_2', 128),
            dim_neck_2=enc.get('dim_neck_2', 1),
            freq_2=enc.get('freq_2', 8),
            dim_enc_3=enc.get('dim_enc_3', 256),
            dim_neck_3=enc.get('dim_neck_3', 32),
            freq_3=enc.get('freq_3', 8),
            f0_n_conv=enc.get('f0_n_conv', 3),
            f0_num_bins=enc.get('f0_num_bins', 256),
            c_h=spk.get('c_h', 128),
            kernel_size=spk.get('kernel_size', 5),
            bank_size=spk.get('bank_size', 8),
            bank_scale=spk.get('bank_scale', 1),
            c_bank=spk.get('c_bank', 128),
            n_conv_blocks=spk.get('n_conv_blocks', 6),
            n_dense_blocks=spk.get('n_dense_blocks', 6),
            subsample=spk.get('subsample', [1, 2, 1, 2, 1, 2]),
            act=spk.get('act', 'relu'),
            dropout_rate=spk.get('dropout_rate', 0.0),
            dim_dec=dec.get('dim_dec', 512),
            lstm_layers=dec.get('lstm_layers', 3),
            num_bits=wm.get('num_bits', 16),
            dim_w=wm.get('dim_w', 16),
            encoder_mlp_hidden=wm.get('encoder_mlp_hidden', 32),
            extractor_mlp_hidden=wm.get('extractor_mlp_hidden', 32),
            fusion_conv_kernel=wm.get('fusion_conv_kernel', 3),
            fusion_conv_channels=wm.get('fusion_conv_channels', 16),
            ckpt_path=ckpt_path,
            strict=False,  # 部分ロードを許容（Watermark 系は随時追加）
            load_pretrained_extractor=self.load_pretrained_extractor,
            freeze_extractor_content_encoder=self.freeze_extractor_content_encoder,
        ).to(self.device)
        return model

    def _build_losses(self, ckpt_path: Optional[str]) -> None:
        lc = self.cfg.get('losses', {})
        enc = self.cfg.get('encoder', {})
        wm  = self.cfg.get('watermark', {})

        self.rec_loss = ReconstructionLoss(
            aux_weight=lc.get('reconstruction', {}).get('aux_weight', 1.0),
        ).to(self.device)

        self.wm_loss = WatermarkLoss().to(self.device)

        nce_cfg = lc.get('info_nce', {})
        self.nce_loss = InfoNCELoss(
            temperature=nce_cfg.get('temperature', 0.1),
            symmetric=nce_cfg.get('symmetric', True),
        ).to(self.device)

        vc_cfg = lc.get('vclub', {})
        self.vclub = VCLUBLoss(
            d_c=enc.get('dim_neck', 8) * 2,
            d_r=enc.get('dim_neck_2', 1) * 2,
            d_f=enc.get('dim_neck_3', 32) * 2,
            hidden_size=vc_cfg.get('hidden_size', 512),
            include_rp=vc_cfg.get('include_rp', False),
            ckpt_path=ckpt_path,
            load_pretrained=vc_cfg.get('load_pretrained', True),
        ).to(self.device)

        adv_cfg = lc.get('adversarial', {})
        self.adv_loss = AdversarialLoss(
            d_c=enc.get('dim_neck', 8) * 2,
            num_classes=adv_cfg.get('num_classes', 110),
            hidden_dim=adv_cfg.get('hidden_dim', 256),
            alpha=adv_cfg.get('alpha', 1.0),
        ).to(self.device)

        dcfg = self.cfg.get('distortion', {})
        self.distortion = DistortionLayer(dcfg).to(self.device)

    def _build_optimizers(self, boost_watermark_lr: Optional[float] = None) -> None:
        tc = self.cfg.get('training', {})
        lc = self.cfg.get('losses', {})

        beta1 = tc.get('beta1', 0.9)
        beta2 = tc.get('beta2', 0.999)

        base_g_lr = tc.get('g_lr', 1e-4)
        wm_lr = base_g_lr * boost_watermark_lr if boost_watermark_lr is not None else base_g_lr

        param_groups = self.model.get_param_groups(
            g_lr=wm_lr,
            content_lr=tc.get('content_lr', 1e-5),
            decoder_lr=tc.get('decoder_lr', 1e-5),
        )

        # bypass_extractor のパラメータを WM モジュールグループ (index 1) に追加
        if self.bypass_decoder:
            bypass_params = list(self.bypass_extractor.parameters())
            param_groups[1]['params'] = param_groups[1]['params'] + bypass_params
            self.logger.info(
                f"[bypass] bypass_extractor params added to optimizer_G group[1]: "
                f"{sum(p.numel() for p in bypass_params)} params, lr={wm_lr}"
            )

        if boost_watermark_lr is not None:
            self.logger.info(
                f"[boost] WM module lr = {base_g_lr} × {boost_watermark_lr} = {wm_lr}"
            )

        self.optimizer_G = optim.Adam(param_groups, betas=(beta1, beta2))

        vclub_lr = lc.get('vclub', {}).get('lr', 3e-4)
        self.optimizer_D_vclub = optim.Adam(
            self.vclub.get_estimator_parameters(),
            lr=vclub_lr,
            betas=(beta1, beta2),
        )

        adv_lr = lc.get('adversarial', {}).get('lr', 1e-4)
        self.optimizer_D_adv = optim.Adam(
            self.adv_loss.get_discriminator_parameters(),
            lr=adv_lr,
            betas=(beta1, beta2),
        )

    # ------------------------------------------------------------------
    # 透かし生成
    # ------------------------------------------------------------------

    def _generate_watermark(self, batch_size: int) -> torch.Tensor:
        """バッチ内全サンプルが異なる 16-bit バイナリ透かしを生成する。

        torch.randperm(2**16) で 65536 個のユニーク整数をシャッフルし
        先頭 batch_size 個を選ぶ。これにより In-batch negatives の
        False Negative（同一透かしが複数存在する状況）を完全に回避する。

        fixed_watermark_seed が指定された場合は毎回同じ透かしを返す
        （overfit 検証用）。

        Args:
            batch_size: バッチサイズ B (最大 65536)
        Returns:
            (B, 16) float {0.0, 1.0}
        """
        num_bits = self.cfg.get('watermark', {}).get('num_bits', 16)
        if self._fixed_watermark_seed is not None:
            gen = torch.Generator()
            gen.manual_seed(self._fixed_watermark_seed)
            indices = torch.randperm(2 ** num_bits, generator=gen)[:batch_size]
        else:
            indices = torch.randperm(2 ** num_bits)[:batch_size]
        bits = ((indices.unsqueeze(1) >> torch.arange(num_bits)) & 1).float()
        return bits.to(self.device)

    # ------------------------------------------------------------------
    # Phase A: vCLUB 推定器の先行更新
    # ------------------------------------------------------------------

    def _phase_a_step(
        self,
        outputs_a: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """vCLUB MI 推定器のみを更新する。ContentEncoder には勾配を流さない。

        outputs_a は torch.no_grad() 下で生成されたテンソルのため
        requires_grad=False。estimator_loss() は内部でも .detach() する
        ため、勾配は推定器パラメータのみに流れる（二重安全策）。

        ContentEncoder / Decoder に Dropout が存在しないため forward は
        決定論的。同一 outputs_a を estimator_iters 回使い回すことが
        SRD-VC 準拠の「各 iter で forward を呼ぶ」実装と数値的に等価かつ
        ~estimator_iters 倍効率的（中間活性化の再計算が不要）。

        Args:
            outputs_a: torch.no_grad() 下の model.forward() 出力 dict
        Returns:
            estimator_loss の平均値（ログ用スカラー）
        """
        estimator_iters = self.cfg.get('losses', {}).get('vclub', {}).get('estimator_iters', 5)
        total_est_loss = torch.tensor(0.0, device=self.device)

        for _ in range(estimator_iters):
            self.optimizer_D_vclub.zero_grad()
            loss_est = self.vclub.estimator_loss(
                outputs_a['z_c'], outputs_a['z_r'], outputs_a['z_f'],
            )
            loss_est.backward()
            self.optimizer_D_vclub.step()
            total_est_loss = total_est_loss + loss_est.detach()

        return total_est_loss / estimator_iters

    # ------------------------------------------------------------------
    # Phase B: Generator + Adversarial Discriminator の同時更新
    # ------------------------------------------------------------------

    def _phase_b_step(
        self,
        mel: torch.Tensor,
        f0_norm: torch.Tensor,
        W: torch.Tensor,
        speaker_id: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """メインネットと Adversarial Discriminator を 1 回の backward で更新する。

        勾配二重カウントが発生しない理由:
          - optimizer_G と optimizer_D_adv のパラメータセットは完全に disjoint
          - loss_total.backward() が .grad を 1 回だけ書き込む
          - GRL は backward 中に ContentEncoder への勾配を符号反転するため
            optimizer_G.step() は「すでに反転済み」の .grad を読む
          - 2 つの .step() 呼び出しは各自の disjoint なパラメータに対して
            それぞれ 1 回だけ更新を行う

        Args:
            mel:        (B, 80, T) channel-first
            f0_norm:    (B, T)
            W:          (B, num_bits) binary float
            speaker_id: (B,) int64
        Returns:
            各 loss とメトリクスの dict（全て .detach() 済み、ログ用）
        """
        zero = torch.tensor(0.0, device=self.device)

        # ------------------------------------------------------------------
        # bypass_decoder モード: Decoder/Attack/Extractor をすべてスキップ
        # optimizer_G のみ更新（D_vclub / D_adv は呼ばない）
        # ------------------------------------------------------------------
        if self.bypass_decoder:
            outputs = self.model.forward_bypass(mel, f0_norm, W)
            W_hat = self.bypass_extractor(outputs['z_c_fused'])
            loss_wm, ber_train = self.wm_loss(W_hat, W)
            loss_nce = self.nce_loss(outputs['z_c_fused'], W)
            loss_total = self.lambda_wm * loss_wm + self.lambda_nce * loss_nce
            self.optimizer_G.zero_grad()
            loss_total.backward()
            self.optimizer_G.step()
            return {
                'loss_total': loss_total.detach(),
                'loss_rec':   zero,
                'loss_wm':    loss_wm.detach(),
                'loss_nce':   loss_nce.detach(),
                'loss_vclub': zero,
                'loss_adv':   zero,
                'ber_train':  ber_train,
            }

        # ------------------------------------------------------------------
        # 通常 forward（Decoder + Attack + Extractor を使う）
        # ------------------------------------------------------------------
        outputs = self.model(mel, f0_norm, W, attack_fn=self.distortion)

        # mel は channel-first (B,80,T); Decoder 出力は time-first (B,T,80)
        mel_tf = mel.transpose(1, 2)
        loss_rec = self.rec_loss(mel_tf, outputs['mel_postnet'], outputs['mel_before_postnet'])
        loss_wm, ber_train = self.wm_loss(outputs['W_hat'], W)
        loss_nce = self.nce_loss(outputs['z_c_fused'], W)

        # ------------------------------------------------------------------
        # disable_vclub_adv モード: Rec + WM + InfoNCE のみ
        # optimizer_G のみ更新（D_adv は呼ばない）
        # ------------------------------------------------------------------
        if self.disable_vclub_adv:
            loss_total = (
                self.lambda_rec * loss_rec +
                self.lambda_wm  * loss_wm  +
                self.lambda_nce * loss_nce
            )
            self.optimizer_G.zero_grad()
            loss_total.backward()
            self.optimizer_G.step()
            return {
                'loss_total': loss_total.detach(),
                'loss_rec':   loss_rec.detach(),
                'loss_wm':    loss_wm.detach(),
                'loss_nce':   loss_nce.detach(),
                'loss_vclub': zero,
                'loss_adv':   zero,
                'ber_train':  ber_train,
            }

        # ------------------------------------------------------------------
        # フル学習: vCLUB + Adversarial を含む全 Loss
        # 1 回の backward: GRL 込みで全パラメータの .grad を書き込む
        # zero_grad を backward の前に呼ぶことが二重カウント回避の必須条件
        # ------------------------------------------------------------------
        loss_vclub = self.vclub.vclub_loss(outputs['z_c'], outputs['z_r'], outputs['z_f'])
        loss_adv   = self.adv_loss(outputs['z_c'], outputs['z_c_hat'], speaker_id)
        loss_total = (
            self.lambda_rec   * loss_rec   +
            self.lambda_wm    * loss_wm    +
            self.lambda_nce   * loss_nce   #+
            #self.lambda_vclub * loss_vclub +
            #self.lambda_adv   * loss_adv
        )
        self.optimizer_G.zero_grad()
        self.optimizer_D_adv.zero_grad()
        loss_total.backward()
        self.optimizer_G.step()
        self.optimizer_D_adv.step()

        return {
            'loss_total': loss_total.detach(),
            'loss_rec':   loss_rec.detach(),
            'loss_wm':    loss_wm.detach(),
            'loss_nce':   loss_nce.detach(),
            'loss_vclub': loss_vclub.detach(),
            'loss_adv':   loss_adv.detach(),
            'ber_train':  ber_train,
        }

    # ------------------------------------------------------------------
    # エポック学習
    # ------------------------------------------------------------------

    def _train_epoch(
        self,
        loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """1 エポック分の学習ループ。

        Returns:
            各 loss と ber_train の平均値 dict
        """
        self.model.train()
        self.distortion.train()
        self.distortion.set_epoch(epoch)

        tc = self.cfg.get('training', {})
        log_every = tc.get('log_every_n_iters', 50)
        n_iters = len(loader)

        accum: Dict[str, float] = {
            k: 0.0 for k in [
                'loss_total', 'loss_rec', 'loss_wm',
                'loss_nce', 'loss_vclub', 'loss_adv',
                'ber_train', 'loss_est',
            ]
        }
        count = 0

        for i, (mel, f0_norm, speaker_id) in enumerate(loader, start=1):
            mel        = mel.to(self.device)
            f0_norm    = f0_norm.to(self.device)
            speaker_id = speaker_id.to(self.device)
            W          = self._generate_watermark(mel.size(0))

            # Phase A: vCLUB 推定器の先行更新
            # bypass_decoder / disable_vclub_adv のいずれかが True の場合はスキップ。
            # スキップ時は optimizer_D_vclub を一切呼ばない。
            if not self.disable_vclub_adv and not self.bypass_decoder:
                with torch.no_grad():
                    outputs_a = self.model(mel, f0_norm, W, attack_fn=None)
                loss_est_avg = self._phase_a_step(outputs_a)
            else:
                loss_est_avg = torch.tensor(0.0, device=self.device)

            # Phase B: メインネット + Adversarial Discriminator 同時更新
            losses = self._phase_b_step(mel, f0_norm, W, speaker_id)

            for k, v in losses.items():
                accum[k] += v.item() if isinstance(v, torch.Tensor) else float(v)
            accum['loss_est'] += loss_est_avg.item()
            count += 1

            if i % log_every == 0:
                self.logger.info(
                    f"  iter {i:04d}/{n_iters:04d} | "
                    f"loss_total={losses['loss_total'].item():.4f} "
                    f"ber_train={losses['ber_train'].item():.4f} "
                    f"loss_est={loss_est_avg.item():.4f}"
                )

        return {k: v / count for k, v in accum.items()} if count > 0 else accum

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """クリーン BER と攻撃後 BER の 2 メトリクスを計算する。

        clean BER:   attack_fn=None（再構築のみ）
        attacked BER: self.val_distortion（常に heavy phase、train モード固定）
                      を使用。val_distortion は __init__ で別インスタンスとして
                      生成されるため、self.distortion のモード切り替えが不要。

        Returns:
            {'val_ber_clean': float, 'val_ber_attacked': float}
        """
        self.model.eval()

        bers_clean: List[float] = []
        bers_att: List[float] = []

        with torch.no_grad():
            for mel, f0_norm, speaker_id in val_loader:
                mel     = mel.to(self.device)
                f0_norm = f0_norm.to(self.device)
                W       = self._generate_watermark(mel.size(0))

                if self.bypass_decoder:
                    # bypass モード: attack なし、BypassExtractor で BER を計測
                    out = self.model.forward_bypass(mel, f0_norm, W)
                    W_hat = self.bypass_extractor(out['z_c_fused'])
                    _, ber = self.wm_loss(W_hat, W)
                    bers_clean.append(ber.item())
                    bers_att.append(ber.item())   # attack なしのため同値
                else:
                    out_clean = self.model(mel, f0_norm, W, attack_fn=None)
                    _, ber_clean = self.wm_loss(out_clean['W_hat'], W)
                    bers_clean.append(ber_clean.item())

                    out_att = self.model(mel, f0_norm, W, attack_fn=self.val_distortion)
                    _, ber_att = self.wm_loss(out_att['W_hat'], W)
                    bers_att.append(ber_att.item())

        n = len(bers_clean)
        return {
            'val_ber_clean':    sum(bers_clean) / n if n > 0 else float('nan'),
            'val_ber_attacked': sum(bers_att)   / n if n > 0 else float('nan'),
        }

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """checkpoint を保存する。

        保存ファイル:
          checkpoint_latest.pt          毎エポック末に上書き（resume 用）
          checkpoint_epoch{N:04d}.pt    checkpoint_every_n_epochs 毎に追記
          checkpoint_best.pt            val_ber_clean が最良時に上書き

        保存内容:
          model / vclub / adv_loss の state_dict（rec_loss/wm_loss/nce_loss は
          学習パラメータなしのため除外）、3 Optimizer の state_dict、
          epoch、best_val_ber_clean
        """
        tc = self.cfg.get('training', {})
        ckpt_dir = tc.get('checkpoint_dir', 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)

        state = {
            'epoch':              epoch,
            'model':              self.model.state_dict(),
            'vclub':              self.vclub.state_dict(),
            'adv_loss':           self.adv_loss.state_dict(),
            'optimizer_G':        self.optimizer_G.state_dict(),
            'optimizer_D_vclub':  self.optimizer_D_vclub.state_dict(),
            'optimizer_D_adv':    self.optimizer_D_adv.state_dict(),
            'best_val_ber_clean': self.best_val_ber_clean,
        }
        if self.bypass_decoder:
            state['bypass_extractor'] = self.bypass_extractor.state_dict()

        latest = os.path.join(ckpt_dir, 'checkpoint_latest.pt')
        torch.save(state, latest)

        every_n = tc.get('checkpoint_every_n_epochs', 10)
        if epoch % every_n == 0:
            periodic = os.path.join(ckpt_dir, f'checkpoint_epoch{epoch:04d}.pt')
            torch.save(state, periodic)

        if is_best:
            best = os.path.join(ckpt_dir, 'checkpoint_best.pt')
            torch.save(state, best)

    def _load_checkpoint(self, path: str) -> None:
        """checkpoint を復元する。

        map_location=self.device により CUDA/CPU 間のデバイス差異を吸収する。
        """
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state['model'])
        self.vclub.load_state_dict(state['vclub'])
        self.adv_loss.load_state_dict(state['adv_loss'])
        self.optimizer_G.load_state_dict(state['optimizer_G'])
        self.optimizer_D_vclub.load_state_dict(state['optimizer_D_vclub'])
        self.optimizer_D_adv.load_state_dict(state['optimizer_D_adv'])
        if self.bypass_decoder and 'bypass_extractor' in state:
            self.bypass_extractor.load_state_dict(state['bypass_extractor'])
        self.start_epoch = state['epoch'] + 1
        self.best_val_ber_clean = state.get('best_val_ber_clean', float('inf'))
        self.logger.info(f"Resumed from epoch {state['epoch']}: {path}")

    # ------------------------------------------------------------------
    # メインループ
    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: Optional[int] = None,
    ) -> None:
        """学習ループのエントリポイント。

        Args:
            train_loader: 学習用 DataLoader
            val_loader:   検証用 DataLoader。None の場合は val をスキップ。
            num_epochs:   学習エポック数。None の場合は cfg から取得。
        """
        tc = self.cfg.get('training', {})
        if num_epochs is None:
            num_epochs = tc.get('num_epochs', 100)
        val_every = tc.get('val_every_n_epochs', 5)

        for epoch in range(self.start_epoch, num_epochs + 1):
            epoch_start = time.time()

            train_metrics = self._train_epoch(train_loader, epoch)

            val_metrics: Dict[str, float] = {}
            if val_loader is not None and epoch % val_every == 0:
                val_metrics = self.validate(val_loader)
                self.model.train()  # validate() が eval() にするため戻す

            elapsed = time.time() - epoch_start

            # エポック末ログ
            log_parts = [
                f"Epoch {epoch:03d}/{num_epochs:03d}",
                f"loss_total={train_metrics.get('loss_total', 0):.4f}",
                f"loss_rec={train_metrics.get('loss_rec', 0):.4f}",
                f"loss_wm={train_metrics.get('loss_wm', 0):.4f}",
                f"loss_nce={train_metrics.get('loss_nce', 0):.4f}",
                f"loss_vclub={train_metrics.get('loss_vclub', 0):.4f}",
                f"loss_adv={train_metrics.get('loss_adv', 0):.4f}",
                f"ber_train={train_metrics.get('ber_train', 0):.4f}",
            ]
            if val_metrics:
                log_parts.append(
                    f"val_ber_clean={val_metrics.get('val_ber_clean', float('nan')):.4f}"
                )
                log_parts.append(
                    f"val_ber_attacked={val_metrics.get('val_ber_attacked', float('nan')):.4f}"
                )
            log_parts.append(f"{elapsed:.1f}s")
            self.logger.info(" | ".join(log_parts))

            # Checkpoint 保存
            is_best = False
            if val_metrics:
                val_ber = val_metrics.get('val_ber_clean', float('inf'))
                if val_ber < self.best_val_ber_clean:
                    self.best_val_ber_clean = val_ber
                    is_best = True

            self._save_checkpoint(epoch, is_best=is_best)
