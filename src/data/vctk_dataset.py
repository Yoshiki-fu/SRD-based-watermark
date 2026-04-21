"""
VC-Robust Neural Audio Watermarking — VCTK Dataset

前処理済み VCTK データ（/workspace/vctk_preprocessed/）を読み込む
torch.utils.data.Dataset 実装。MockVCTKDataset と同一インターフェース。

クロップ/パディング実装: SRD-VC/My_model/data_loader.py (MyCollator) を参考に実装。
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class VCTKDataset(Dataset):
    """前処理済み VCTK データセット。

    metadata.json を __init__ 時に全件メモリロードし、
    __getitem__ では mel/f0 .npy を逐次読み込む。
    train: ランダムクロップ + 右パディング (len_crop ∈ [min_len_seq, max_len_seq])
    val/test: 中央固定クロップ + 右パディング (len_crop = min(max_len_seq, T))

    Args:
        root_dir:    前処理済みデータのルートディレクトリ
                     （metadata.json、mel/、f0/ を含む）
        split:       "train" | "val" | "test"
        config:      default.yaml 全体を読み込んだ dict
        min_len_seq: クロップ最小フレーム数 (default 64)
        max_len_seq: クロップ最大フレーム数 (default 128)
    """

    def __init__(
        self,
        root_dir: str,
        split: str,
        config: Dict,
        min_len_seq: int = 64,
        max_len_seq: int = 128,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split は 'train'/'val'/'test' のいずれか。got: '{split}'")

        self.root_dir = Path(root_dir)
        self.split = split
        self.is_train = split == "train"
        self.min_len_seq = min_len_seq
        self.max_len_seq = max_len_seq
        self.max_len_pad: int = config["encoder"]["max_len_pad"]  # 192

        with open(self.root_dir / "metadata.json", "r") as f:
            metadata = json.load(f)

        raw_entries: List[Dict] = metadata[split]

        self.entries: List[Dict] = []
        skipped = 0
        for entry in raw_entries:
            mel_path = self.root_dir / entry["mel"]
            f0_path = self.root_dir / entry["f0"]
            if not mel_path.exists() or not f0_path.exists():
                skipped += 1
                continue
            # mmap で先頭 shape のみ確認（全データをロードしない）
            mel_arr = np.load(str(mel_path), mmap_mode="r")
            if mel_arr.shape[0] < min_len_seq:
                skipped += 1
                continue
            self.entries.append(entry)

        if skipped > 0:
            logger.warning(
                "VCTKDataset[%s]: %d 件をスキップ（ファイル欠損 or T < %d フレーム）",
                split,
                skipped,
                min_len_seq,
            )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            idx: サンプルインデックス

        Returns:
            mel:        (80, 192) float32  channel-first メルスペクトログラム
            f0_norm:    (192,)    float32  voiced ∈ (-∞, 1]、unvoiced = -1e10
            speaker_id: scalar   int64
        """
        entry = self.entries[idx]
        mel_np: np.ndarray = np.load(str(self.root_dir / entry["mel"]))  # (T, 80)
        f0_np: np.ndarray = np.load(str(self.root_dir / entry["f0"]))    # (T,)

        T = mel_np.shape[0]

        if self.is_train:
            # SRD-VC MyCollator 準拠: ランダム長・ランダム開始位置クロップ
            len_crop = int(np.random.randint(self.min_len_seq, self.max_len_seq + 1))
            left = int(np.random.randint(0, T - len_crop + 1))
        else:
            # val/test: 中央固定クロップ（再現性のため）
            len_crop = min(self.max_len_seq, T)
            center = T // 2
            left = max(0, center - len_crop // 2)
            left = min(left, T - len_crop)

        mel_crop = mel_np[left : left + len_crop, :]  # (len_crop, 80)
        f0_crop = f0_np[left : left + len_crop]       # (len_crop,)

        pad_len = self.max_len_pad - len_crop
        # 右パディング: mel → 0.0、f0 → -1e10 (SRD-VC MyCollator 準拠)
        mel_pad = np.pad(
            mel_crop,
            ((0, pad_len), (0, 0)),
            mode="constant",
            constant_values=0.0,
        )  # (192, 80)
        f0_pad = np.pad(
            f0_crop,
            (0, pad_len),
            mode="constant",
            constant_values=-1e10,
        )  # (192,)

        # (192, 80) → transpose → (80, 192)、非連続メモリを解消するため .copy()
        mel_tensor = torch.from_numpy(mel_pad.T.copy().astype(np.float32))
        f0_tensor = torch.from_numpy(f0_pad.astype(np.float32))
        speaker_id = torch.tensor(entry["speaker_id"], dtype=torch.long)

        return mel_tensor, f0_tensor, speaker_id
