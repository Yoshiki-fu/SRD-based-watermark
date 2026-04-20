"""
src/data/mock_dataset.py の shape・dtype・再現性・DataLoader 統合テスト。
"""

import torch
from torch.utils.data import DataLoader

from src.data import MockVCTKDataset


# ── fixtures ──────────────────────────────────────────────────────────────────

def _make_dataset(**kwargs) -> MockVCTKDataset:
    defaults = dict(num_samples=100, num_speakers=110, seed=42)
    defaults.update(kwargs)
    return MockVCTKDataset(**defaults)


# ── shape / dtype ─────────────────────────────────────────────────────────────

def test_len() -> None:
    ds = MockVCTKDataset(num_samples=37)
    assert len(ds) == 37


def test_getitem_shapes() -> None:
    ds = _make_dataset()
    mel, f0_norm, speaker_id = ds[0]

    assert mel.shape == (80, 192), f"mel shape: {mel.shape}"
    assert f0_norm.shape == (192,), f"f0_norm shape: {f0_norm.shape}"
    assert speaker_id.shape == (), f"speaker_id should be scalar, got shape {speaker_id.shape}"


def test_getitem_dtypes() -> None:
    ds = _make_dataset()
    mel, f0_norm, speaker_id = ds[0]

    assert mel.dtype == torch.float32, f"mel dtype: {mel.dtype}"
    assert f0_norm.dtype == torch.float32, f"f0_norm dtype: {f0_norm.dtype}"
    assert speaker_id.dtype == torch.int64, f"speaker_id dtype: {speaker_id.dtype}"


# ── value ranges ──────────────────────────────────────────────────────────────

def test_value_ranges() -> None:
    ds = _make_dataset(seed=0)
    for idx in range(len(ds)):
        _, f0_norm, speaker_id = ds[idx]
        assert float(f0_norm.min()) >= 0.0, "f0_norm contains negative values"
        assert float(f0_norm.max()) <= 1.0, "f0_norm exceeds 1.0"
        assert 0 <= int(speaker_id) < 110, f"speaker_id out of range: {speaker_id}"


def test_f0_has_unvoiced_frames() -> None:
    # 1000 サンプル生成して無声フレーム（f0 == 0）が1つ以上存在することを確認
    ds = MockVCTKDataset(num_samples=1000, seed=0)
    found_unvoiced = False
    for idx in range(len(ds)):
        _, f0_norm, _ = ds[idx]
        if (f0_norm == 0.0).any():
            found_unvoiced = True
            break
    assert found_unvoiced, "No unvoiced frames (f0==0) found in 1000 samples"


def test_f0_has_voiced_frames() -> None:
    # 1000 サンプル生成して有声フレーム（f0 > 0）が1つ以上存在することを確認
    ds = MockVCTKDataset(num_samples=1000, seed=0)
    found_voiced = False
    for idx in range(len(ds)):
        _, f0_norm, _ = ds[idx]
        if (f0_norm > 0.0).any():
            found_voiced = True
            break
    assert found_voiced, "No voiced frames (f0>0) found in 1000 samples"


# ── reproducibility ───────────────────────────────────────────────────────────

def test_seed_reproducibility() -> None:
    ds = _make_dataset(seed=99)
    mel_a, f0_a, sid_a = ds[7]
    mel_b, f0_b, sid_b = ds[7]

    assert torch.allclose(mel_a, mel_b), "mel not reproducible with same seed+idx"
    assert torch.allclose(f0_a, f0_b), "f0_norm not reproducible with same seed+idx"
    assert sid_a == sid_b, "speaker_id not reproducible with same seed+idx"


def test_different_indices_differ() -> None:
    # 異なる idx では（ほぼ）異なるデータが返ること
    ds = _make_dataset(seed=42)
    mel_0, _, _ = ds[0]
    mel_1, _, _ = ds[1]
    assert not torch.allclose(mel_0, mel_1), "Different indices returned identical mel"


# ── DataLoader integration ────────────────────────────────────────────────────

def test_dataloader_batch_shapes() -> None:
    ds = _make_dataset()
    dl = DataLoader(ds, batch_size=16, num_workers=0)
    mel, f0_norm, speaker_id = next(iter(dl))

    assert mel.shape == (16, 80, 192), f"batched mel shape: {mel.shape}"
    assert f0_norm.shape == (16, 192), f"batched f0_norm shape: {f0_norm.shape}"
    assert speaker_id.shape == (16,), f"batched speaker_id shape: {speaker_id.shape}"
    assert speaker_id.dtype == torch.int64, f"batched speaker_id dtype: {speaker_id.dtype}"
