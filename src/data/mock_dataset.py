"""
VC-Robust Neural Audio Watermarking — Mock Dataset

trainer.py 動作確認用のランダム生成データセット。
実 VCTK DataLoader (Phase 2) と同一 interface を提供する。
"""

from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset


class MockVCTKDataset(Dataset):
    """ランダム生成によるダミー VCTK データセット。

    __getitem__ は per-index Generator で決定論的に生成するため、
    seed を指定すると同じ idx では常に同じデータが返る。
    DataLoader(dataset, batch_size=B) で default collate_fn が使用可能。

    Args:
        num_samples:  データセットサイズ (default 100)
        num_speakers: 話者数 (default 110、VCTK 準拠)
        seed:         Optional[int]。再現性のための乱数シード
        mel_bands:    メルバンド数 (default 80)
        seq_len:      系列長 T (default 192)
    """

    def __init__(
        self,
        num_samples: int = 100,
        num_speakers: int = 110,
        seed: Optional[int] = None,
        mel_bands: int = 80,
        seq_len: int = 192,
    ) -> None:
        self.num_samples = num_samples
        self.num_speakers = num_speakers
        self.seed = seed
        self.mel_bands = mel_bands
        self.seq_len = seq_len

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            idx: サンプルインデックス

        Returns:
            mel:        (mel_bands, seq_len) = (80, 192) float32
            f0_norm:    (seq_len,) = (192,)  float32、有声フレーム (0,1]、無声フレーム =0
            speaker_id: scalar int64 tensor
        """
        if self.seed is not None:
            gen = torch.Generator()
            gen.manual_seed(self.seed + idx)
        else:
            gen = None

        mel = torch.randn(self.mel_bands, self.seq_len, generator=gen)

        # 無声フレームを約20%含む F0 系列（実 VCTK の典型的な特性）
        f0_norm = torch.rand(self.seq_len, generator=gen)
        voiced_mask = torch.rand(self.seq_len, generator=gen) > 0.2
        f0_norm = f0_norm * voiced_mask.float()

        speaker_id_val = int(
            torch.randint(0, self.num_speakers, (1,), generator=gen).item()
        )
        speaker_id = torch.tensor(speaker_id_val, dtype=torch.long)

        return mel, f0_norm, speaker_id
