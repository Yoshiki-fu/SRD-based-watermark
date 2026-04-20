"""
VC-Robust Neural Audio Watermarking — 学習スクリプト

Usage:
  python scripts/train.py --config configs/default.yaml
  python scripts/train.py --config configs/default.yaml --resume checkpoints/checkpoint_latest.pt
  python scripts/train.py --config configs/default.yaml --device cuda
  python scripts/train.py --config configs/default.yaml --no-pretrained  # ランダム初期化（テスト用）
"""

import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader

from src.data.mock_dataset import MockVCTKDataset
from src.training.trainer import VCWatermarkTrainer


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
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    num_speakers = cfg.get('losses', {}).get('adversarial', {}).get('num_classes', 110)
    num_samples  = cfg.get('data', {}).get('num_samples', 100)
    num_workers  = cfg.get('data', {}).get('num_workers', 0)
    batch_size   = cfg.get('training', {}).get('batch_size', 16)

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=(args.device is not None and 'cuda' in args.device),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    # --no-pretrained 指定時は空文字を渡してランダム初期化
    ckpt_path = '' if args.no_pretrained else None

    trainer = VCWatermarkTrainer(
        cfg=cfg,
        ckpt_path=ckpt_path,
        resume_from=args.resume,
        device=args.device,
    )
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=cfg.get('training', {}).get('num_epochs', 100),
    )


if __name__ == '__main__':
    main()
