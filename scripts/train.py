"""
VC-Robust Neural Audio Watermarking — 学習スクリプト

Usage:
  # Mock データ（デフォルト）
  python scripts/train.py --config configs/default.yaml

  # 実 VCTK データ
  python scripts/train.py --config configs/default.yaml \
      --data_dir /workspace/vctk_preprocessed

  # overfit 検証（4 サンプル、固定透かし、200 エポック）
  python scripts/train.py --config configs/overfit.yaml \
      --data_dir /workspace/vctk_preprocessed \
      --overfit 4 \
      --fixed_watermark \
      --device cuda

  # checkpoint 再開
  python scripts/train.py --config configs/default.yaml \
      --resume checkpoints/checkpoint_latest.pt

  # ランダム初期化（テスト用）
  python scripts/train.py --config configs/default.yaml --no-pretrained
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader, Subset

from src.data.mock_dataset import MockVCTKDataset
from src.data.vctk_dataset import VCTKDataset
from src.training.trainer import VCWatermarkTrainer


def _build_datasets(args: argparse.Namespace, cfg: dict):
    """データセットを構築する。

    data_dir が指定された場合は VCTKDataset、そうでなければ MockVCTKDataset を返す。
    VCTKDataset 使用時は metadata.json から実際の話者数を取得して
    cfg['losses']['adversarial']['num_classes'] を上書きする（trainer 構築前に呼ぶこと）。

    Returns:
        (train_dataset, val_dataset)
    """
    if args.data_dir is not None:
        # 話者数を metadata.json から取得して cfg を更新
        meta_path = os.path.join(args.data_dir, "metadata.json")
        with open(meta_path) as f:
            metadata = json.load(f)
        unique_ids = {e["speaker_id"] for e in metadata.get("train", [])}
        num_speakers = len(unique_ids)
        cfg.setdefault("losses", {}).setdefault("adversarial", {})["num_classes"] = num_speakers

        train_dataset = VCTKDataset(args.data_dir, "train", cfg)
        val_dataset = VCTKDataset(args.data_dir, "val", cfg)
    else:
        num_speakers = cfg.get("losses", {}).get("adversarial", {}).get("num_classes", 110)
        num_samples = cfg.get("data", {}).get("num_samples", 100)
        train_dataset = MockVCTKDataset(
            num_samples=num_samples,
            num_speakers=num_speakers,
            seed=42,
        )
        val_dataset = MockVCTKDataset(
            num_samples=max(16, num_samples // 10),
            num_speakers=num_speakers,
            seed=999,
        )

    if args.overfit is not None:
        n_train = min(args.overfit, len(train_dataset))
        n_val = min(args.overfit, len(val_dataset))
        train_dataset = Subset(train_dataset, list(range(n_train)))
        val_dataset = Subset(val_dataset, list(range(n_val)))

    return train_dataset, val_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description='VC-Robust Audio Watermark Training')
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--resume', default=None, help='checkpoint ファイルパス（再開用）')
    parser.add_argument('--device', default=None, help='cuda / cpu / cuda:0')
    parser.add_argument(
        '--no-pretrained',
        action='store_true',
        help='事前学習 weights を使わずランダム初期化（テスト・デバッグ用）',
    )
    parser.add_argument(
        '--data_dir',
        default=None,
        help='前処理済み VCTK データのルートディレクトリ（例: /workspace/vctk_preprocessed）'
             '。省略時は MockVCTKDataset を使用。',
    )
    parser.add_argument(
        '--overfit',
        type=int,
        default=None,
        metavar='N',
        help='train/val データセットを先頭 N サンプルに制限（overfit 検証用）',
    )
    parser.add_argument(
        '--fixed_watermark',
        action='store_true',
        help='毎 iteration 同じ透かし W を使用（overfit 検証用）',
    )
    parser.add_argument(
        '--bypass_decoder',
        action='store_true',
        help='診断モード: Decoder/Attack/Extractor をスキップし '
             'z_c_fused → BypassExtractor → W_hat で直接復元する',
    )
    parser.add_argument(
        '--boost_watermark_lr',
        type=float,
        default=None,
        metavar='N',
        help='WM Encoder / FusionLayer / Extractor の学習率を base_lr × N 倍にする '
             '（例: --boost_watermark_lr 10 → 1e-3 × 10 = 1e-2）',
    )
    parser.add_argument(
        '--disable_vclub_adv',
        action='store_true',
        help='vCLUB と Adversarial Loss を無効化し '
             'Reconstruction + Watermark + InfoNCE のみで学習する',
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # データセット構築（VCTKDataset 使用時は cfg の num_classes を更新）
    train_dataset, val_dataset = _build_datasets(args, cfg)

    num_workers = cfg.get('data', {}).get('num_workers', 0)
    batch_size = cfg.get('training', {}).get('batch_size', 16)
    pin_memory = args.device is not None and 'cuda' in args.device

    # num_workers > 0 の場合の乱数シード固定（SRD-VC 準拠）
    worker_init_fn = (
        lambda x: np.random.seed(torch.initial_seed() % (2 ** 32))
        if num_workers > 0 else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=pin_memory,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=pin_memory,
    )

    # --no-pretrained 指定時は空文字を渡してランダム初期化
    ckpt_path = '' if args.no_pretrained else None

    trainer = VCWatermarkTrainer(
        cfg=cfg,
        ckpt_path=ckpt_path,
        resume_from=args.resume,
        device=args.device,
        fixed_watermark_seed=0 if args.fixed_watermark else None,
        bypass_decoder=args.bypass_decoder,
        boost_watermark_lr=args.boost_watermark_lr,
        disable_vclub_adv=args.disable_vclub_adv,
    )
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=cfg.get('training', {}).get('num_epochs', 100),
    )


if __name__ == '__main__':
    main()
