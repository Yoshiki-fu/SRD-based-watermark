#!/usr/bin/env python3
"""
VCTK前処理スクリプト: メルスペクトログラムと正規化済みF0を事前計算して保存。

SRD-VC (Yang et al., Interspeech 2022) の前処理と完全互換。
800000-G.ckpt との互換性を保つためメル・F0の計算仕様はSRD-VCと一致。

参考元:
    SRD-VC/My_model/make_spect_f0.py — メル・F0計算ロジック
    SRD-VC/My_model/utils.py       — pySTFT, butter_highpass, speaker_normalization

使用例:
    # 1話者のみでテスト
    python scripts/preprocess_vctk.py --speakers p225

    # フル処理 (8並列)
    python scripts/preprocess_vctk.py --num_workers 8
"""
import argparse
import json
import logging
import multiprocessing
import os
import pickle
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from scipy import signal
from scipy.signal import get_window

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ワーカープロセスで共有する定数 (_worker_init で設定)
_mel_basis: Optional[np.ndarray] = None
_b: Optional[np.ndarray] = None
_a: Optional[np.ndarray] = None
_min_level: float = float(np.exp(-100 / 20 * np.log(10)))
_min_frames: int = 64
_librosa: Any = None
_sptk: Any = None


# ---------------------------------------------------------------------------
# SRD-VC準拠のDSP関数
# ---------------------------------------------------------------------------

def butter_highpass(
    cutoff: float,
    fs: int,
    order: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Butterworth ハイパスフィルタ係数を計算する。

    SRD-VC/My_model/utils.py:10-14 準拠。

    Args:
        cutoff: カットオフ周波数 [Hz]
        fs: サンプリング周波数 [Hz]
        order: フィルタ次数
    Returns:
        b, a: フィルタ係数
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype="high", analog=False)
    return b, a


def py_stft(
    x: np.ndarray,
    fft_length: int = 1024,
    hop_length: int = 256,
) -> np.ndarray:
    """
    カスタム STFT (stride_tricks 実装)。reflect padding 込み。

    SRD-VC/My_model/utils.py:18-31 準拠。

    Args:
        x: (N,) 音声波形
        fft_length: FFT サイズ
        hop_length: ホップ長
    Returns:
        magnitude: (n_fft//2+1, T) 絶対値スペクトログラム
    """
    x = np.pad(x, int(fft_length // 2), mode="reflect")
    noverlap = fft_length - hop_length
    shape = x.shape[:-1] + ((x.shape[-1] - noverlap) // hop_length, fft_length)
    strides = x.strides[:-1] + (hop_length * x.strides[-1], x.strides[-1])
    result = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    fft_window = get_window("hann", fft_length, fftbins=True)
    result = np.fft.rfft(fft_window * result, n=fft_length).T
    return np.abs(result)


def speaker_normalization(
    f0: np.ndarray,
    index_nonzero: np.ndarray,
    mean_f0: float,
    std_f0: float,
) -> np.ndarray:
    """
    F0 の発話単位正規化 (voiced フレームのみ)。

    SRD-VC/My_model/utils.py:35-42 完全準拠。
    voiced: z-score ÷ 4 → clip[-1, 1] → rescale [0, 1]
    unvoiced: -1e10 のまま (F0Quantizer が ≤0 を unvoiced class に割当)

    Args:
        f0: (T,) log F0 値列。unvoiced フレームは -1e10
        index_nonzero: (T,) voiced フレームの bool マスク
        mean_f0: voiced フレームの F0 平均
        std_f0: voiced フレームの F0 標準偏差
    Returns:
        f0_norm: (T,) 正規化済み。voiced ∈ [0, 1]、unvoiced = -1e10
    """
    f0 = f0.astype(float).copy()
    f0[index_nonzero] = (f0[index_nonzero] - mean_f0) / std_f0 / 4.0
    f0[index_nonzero] = np.clip(f0[index_nonzero], -1, 1)
    f0[index_nonzero] = (f0[index_nonzero] + 1) / 2.0
    return f0


# ---------------------------------------------------------------------------
# ワーカープロセス: 初期化 & 1ファイル処理
# ---------------------------------------------------------------------------

def _worker_init(
    mel_basis: np.ndarray,
    b: np.ndarray,
    a: np.ndarray,
    min_frames: int,
) -> None:
    """Pool initializer: 各ワーカープロセスで一度だけ実行してDSP定数を設定。"""
    global _mel_basis, _b, _a, _min_frames, _librosa, _sptk
    import librosa as _lib
    from pysptk import sptk as _s
    _mel_basis = mel_basis
    _b = b
    _a = a
    _min_frames = min_frames
    _librosa = _lib
    _sptk = _s


def _process_one_file(
    task: Tuple[str, str, str, float, float],
) -> Dict[str, Any]:
    """
    1ファイルを処理してメル + F0 を保存する。

    SRD-VC/My_model/make_spect_f0.py:47-76 準拠。

    Args:
        task: (wav_path, mel_out, f0_out, lo, hi)
            wav_path: 入力 WAV ファイルパス (48kHz VCTK)
            mel_out: 出力メル .npy パス
            f0_out: 出力 F0 .npy パス
            lo: RAPT 最小 F0 [Hz] (話者性別による)
            hi: RAPT 最大 F0 [Hz] (話者性別による)
    Returns:
        result: {'ok', 'wav_path', 'frames', 'error', 'short'}
    """
    wav_path, mel_out, f0_out, lo, hi = task
    result: Dict[str, Any] = {
        "ok": False,
        "wav_path": wav_path,
        "frames": 0,
        "error": "",
        "short": False,
    }

    try:
        # 1. 16kHz ロード & リサンプリング (VCTK は 48kHz)
        wav, fs = _librosa.load(wav_path, sr=16000, mono=True)
        assert fs == 16000

        # 2. len % 256 == 0 の場合に tiny noise を付加 (make_spect_f0.py:54-55)
        if wav.shape[0] % 256 == 0:
            wav = np.concatenate((wav, np.array([1e-6], dtype=np.float32)))

        # 3. Butterworth ハイパスフィルタ (make_spect_f0.py:56)
        y = signal.filtfilt(_b, _a, wav)

        # 4. メルスペクトログラム (make_spect_f0.py:59-63)
        D = py_stft(y).T                          # (T, n_fft//2+1)
        D_mel = np.dot(D, _mel_basis)             # (T, 80)
        D_db = 20 * np.log10(np.maximum(_min_level, D_mel)) - 16
        S = ((D_db + 100) / 100).astype(np.float32)  # [0, 1]

        # 5. RAPT F0 抽出 (make_spect_f0.py:66)
        f0_rapt = _sptk.rapt(
            y.astype(np.float32) * 32768,
            16000,
            256,
            min=lo,
            max=hi,
            otype=2,  # log F0
        )

        # 6. 発話単位正規化 (make_spect_f0.py:67-69)
        index_nonzero = f0_rapt != -1e10
        if index_nonzero.sum() > 0:
            mean_f0 = float(np.mean(f0_rapt[index_nonzero]))
            std_f0 = float(np.std(f0_rapt[index_nonzero]))
            if std_f0 < 1e-6:
                std_f0 = 1.0
            f0_norm = speaker_normalization(f0_rapt, index_nonzero, mean_f0, std_f0)
        else:
            # 全フレーム無声: -1e10 のまま
            f0_norm = f0_rapt.copy()

        f0_norm = f0_norm.astype(np.float32)

        # フレーム長一致チェック (make_spect_f0.py:71)
        assert len(S) == len(f0_norm), (
            f"Frame mismatch: mel={len(S)}, f0={len(f0_norm)}"
        )

        # 7. .npy 保存
        os.makedirs(os.path.dirname(mel_out), exist_ok=True)
        os.makedirs(os.path.dirname(f0_out), exist_ok=True)
        np.save(mel_out, S, allow_pickle=False)
        np.save(f0_out, f0_norm, allow_pickle=False)

        result["ok"] = True
        result["frames"] = len(S)
        result["short"] = len(S) < _min_frames

    except Exception:
        result["error"] = traceback.format_exc()

    return result


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------

def load_split_speaker(
    split_speaker_path: Path,
) -> Tuple[List[str], List[str]]:
    """
    SRD-VC/VCTK/split_speaker.txt を読み込む。

    ファイル形式:
        train:
        p299 p262 p259 ...
        test:
        p236 p232 ...

    Args:
        split_speaker_path: split_speaker.txt のパス
    Returns:
        train_speakers: 順序保持リスト (100話者)
        test_speakers: 順序保持リスト (9話者)
    """
    text = split_speaker_path.read_text()
    train_speakers: List[str] = []
    test_speakers: List[str] = []
    current: Optional[str] = None
    for line in text.strip().splitlines():
        line = line.strip()
        if line.startswith("train"):
            current = "train"
        elif line.startswith("test"):
            current = "test"
        elif current == "train" and line:
            train_speakers = line.split()
        elif current == "test" and line:
            test_speakers = line.split()
    return train_speakers, test_speakers


def build_tasks(
    vctk_root: Path,
    output_dir: Path,
    speakers: Optional[List[str]],
    train_speakers: List[str],
    test_speakers: List[str],
    spk2gen: Dict[str, str],
    cfg_pre: Dict[str, Any],
) -> Tuple[List[Tuple], Dict[str, Dict], List[str]]:
    """
    処理タスクリストと話者情報を構築する。

    Args:
        vctk_root: wav48_silence_trimmed ディレクトリ
        output_dir: 出力先
        speakers: 絞り込み話者リスト (None = 全話者)
        train_speakers: split_speaker.txt から読んだ train 話者 (順序保持)
        test_speakers: split_speaker.txt から読んだ test 話者
        spk2gen: {speaker_id: 'M'|'F'} マッピング
        cfg_pre: preprocess 設定 dict
    Returns:
        tasks: [(wav_path, mel_out, f0_out, lo, hi), ...]
        speaker_info: {speaker: {speaker_id, gender, split, lo, hi, files}}
        unknown_speakers: split_speaker.txt に無い話者 ID リスト
    """
    f0_min_m = cfg_pre["f0_min_male"]
    f0_max_m = cfg_pre["f0_max_male"]
    f0_min_f = cfg_pre["f0_min_female"]
    f0_max_f = cfg_pre["f0_max_female"]

    all_known = set(train_speakers) | set(test_speakers)
    test_set = set(test_speakers)
    # train 話者の speaker_id は split_speaker.txt の記載順で 0-99 付番
    speaker_id_map = {spk: idx for idx, spk in enumerate(train_speakers)}

    target = set(speakers) if speakers else None

    tasks: List[Tuple] = []
    speaker_info: Dict[str, Dict] = {}
    unknown_speakers: List[str] = []

    for spk_dir in sorted(vctk_root.iterdir()):
        if not spk_dir.is_dir():
            continue
        spk = spk_dir.name

        if target is not None and spk not in target:
            continue
        if spk not in all_known:
            if target is None or spk in target:
                unknown_speakers.append(spk)
            continue

        gender = spk2gen.get(spk, "F")
        lo = f0_min_m if gender == "M" else f0_min_f
        hi = f0_max_m if gender == "M" else f0_max_f
        split = "test" if spk in test_set else "train"
        spk_id = speaker_id_map.get(spk, -1)

        wav_files = sorted(spk_dir.glob("*.wav"))
        if not wav_files:
            wav_files = sorted(spk_dir.glob("**/*.wav"))

        speaker_info[spk] = {
            "speaker_id": spk_id,
            "gender": gender,
            "split": split,
            "lo": lo,
            "hi": hi,
            "files": [p.stem for p in wav_files],
        }

        for wav_path in wav_files:
            stem = wav_path.stem
            mel_out = str(output_dir / "mel" / spk / f"{stem}.npy")
            f0_out = str(output_dir / "f0" / spk / f"{stem}.npy")
            tasks.append((str(wav_path), mel_out, f0_out, lo, hi))

    return tasks, speaker_info, unknown_speakers


def build_metadata(
    speaker_info: Dict[str, Dict],
    output_dir: Path,
    train_speakers: List[str],
) -> Dict[str, Any]:
    """
    metadata.json を構築する。

    train 話者: sorted(files)[:-5] → train、最後5発話 → val
    test 話者: 全発話 → test
    speaker_id_map は split_speaker.txt の順序 (0-99) を記録。

    Args:
        speaker_info: build_tasks が返す話者情報 dict
        output_dir: 出力先 (mel .npy の存在確認に使用)
        train_speakers: split_speaker.txt 記載順の train 話者リスト
    Returns:
        metadata dict
    """
    train_entries: List[Dict] = []
    val_entries: List[Dict] = []
    test_entries: List[Dict] = []

    for spk, info in sorted(speaker_info.items()):
        spk_id = info["speaker_id"]
        split = info["split"]
        stems = sorted(info["files"])
        total = len(stems)

        for idx, stem in enumerate(stems):
            mel_abs = output_dir / "mel" / spk / f"{stem}.npy"
            if not mel_abs.exists():
                continue  # 処理失敗ファイルはスキップ

            entry = {
                "speaker": spk,
                "speaker_id": spk_id,
                "mel": str(Path("mel") / spk / f"{stem}.npy"),
                "f0": str(Path("f0") / spk / f"{stem}.npy"),
            }

            if split == "test":
                test_entries.append(entry)
            elif total > 5 and idx >= total - 5:
                val_entries.append(entry)
            else:
                train_entries.append(entry)

    # speaker_id_map: id → speaker_name (split_speaker.txt の順序を保持)
    speaker_id_map = {
        idx: spk
        for idx, spk in enumerate(train_speakers)
        if spk in speaker_info
    }

    return {
        "train": train_entries,
        "val": val_entries,
        "test": test_entries,
        "speaker_id_map": speaker_id_map,
    }


def build_speaker_stats(
    results: List[Dict[str, Any]],
    speaker_info: Dict[str, Dict],
) -> Dict[str, Any]:
    """
    speaker_stats.json 用の統計情報を構築する。

    Args:
        results: _process_one_file の戻り値リスト
        speaker_info: build_tasks が返す話者情報 dict
    Returns:
        {speaker: {gender, speaker_id, split, f0_min, f0_max,
                   num_files, mean_frames, short_files}}
    """
    spk_results: Dict[str, List[Dict]] = {}
    for r in results:
        if not r["ok"]:
            continue
        spk = Path(r["wav_path"]).parent.name
        spk_results.setdefault(spk, []).append(r)

    stats: Dict[str, Any] = {}
    for spk, info in speaker_info.items():
        rlist = spk_results.get(spk, [])
        frames = [r["frames"] for r in rlist]
        stats[spk] = {
            "gender": info["gender"],
            "speaker_id": info["speaker_id"],
            "split": info["split"],
            "f0_min": info["lo"],
            "f0_max": info["hi"],
            "num_files": len(rlist),
            "mean_frames": float(np.mean(frames)) if frames else 0.0,
            "short_files": sum(1 for r in rlist if r["short"]),
        }
    return stats


# ---------------------------------------------------------------------------
# 環境チェック
# ---------------------------------------------------------------------------

def check_environment(
    split_speaker_path: Path,
    spk2gen_path: Path,
    vctk_root: Path,
) -> None:
    """起動時に依存ライブラリとファイルの存在を確認する。失敗時は即座に終了。"""
    errors: List[str] = []

    for pkg in ("pysptk", "librosa", "tqdm"):
        try:
            __import__(pkg)
        except ImportError:
            errors.append(f"{pkg} がインストールされていません: pip install {pkg}")

    for path, label in [
        (split_speaker_path, "SRD-VC/VCTK/split_speaker.txt"),
        (spk2gen_path, "SRD-VC/My_model/assets/spk2gen.pkl"),
        (vctk_root, "--vctk_root"),
    ]:
        if not path.exists():
            errors.append(f"{label} が見つかりません: {path}")

    if errors:
        for e in errors:
            logger.error(e)
        sys.exit(1)

    logger.info("環境チェック OK")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VCTK前処理: mel + F0 を事前計算して保存 (SRD-VC準拠)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--vctk_root",
        type=Path,
        default=Path("/workspace/wav48_silence_trimmed"),
        help="VCTK wav48_silence_trimmed ディレクトリ",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/workspace/vctk_preprocessed"),
        help="前処理済みデータの出力先",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="ハイパーパラメータ設定ファイル",
    )
    parser.add_argument(
        "--srd_vc_root",
        type=Path,
        default=Path("SRD-VC"),
        help="SRD-VC ディレクトリのルート (読み取り専用)",
    )
    parser.add_argument(
        "--speakers",
        nargs="+",
        default=None,
        metavar="SPEAKER_ID",
        help="処理する話者を絞る (例: --speakers p225 p226)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="並列処理ワーカー数 (未指定時は configs から取得)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # --- 設定読み込み ---
    cfg = yaml.safe_load(args.config.read_text())
    cfg_audio = cfg["audio"]
    cfg_pre = cfg.get("preprocess", {})
    cfg_pre.setdefault("num_workers", 8)
    cfg_pre.setdefault("min_frames", 64)
    cfg_pre.setdefault("f0_min_male", 50)
    cfg_pre.setdefault("f0_max_male", 250)
    cfg_pre.setdefault("f0_min_female", 100)
    cfg_pre.setdefault("f0_max_female", 600)
    cfg_pre.setdefault("highpass_cutoff", 30)
    cfg_pre.setdefault("highpass_order", 5)

    num_workers = args.num_workers if args.num_workers is not None else cfg_pre["num_workers"]

    srd_vc_root = args.srd_vc_root
    split_speaker_path = srd_vc_root / "VCTK" / "split_speaker.txt"
    spk2gen_path = srd_vc_root / "My_model" / "assets" / "spk2gen.pkl"

    # --- 環境チェック ---
    check_environment(split_speaker_path, spk2gen_path, args.vctk_root)

    # --- SRD-VC メタデータ読み込み ---
    train_speakers, test_speakers = load_split_speaker(split_speaker_path)
    with open(spk2gen_path, "rb") as f:
        spk2gen: Dict[str, str] = pickle.load(f)

    logger.info(
        f"Train 話者: {len(train_speakers)} / Test 話者: {len(test_speakers)} "
        f"/ spk2gen エントリ数: {len(spk2gen)}"
    )

    # --- 出力ディレクトリ作成 ---
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- タスク構築 ---
    tasks, speaker_info, unknown = build_tasks(
        vctk_root=args.vctk_root,
        output_dir=args.output_dir,
        speakers=args.speakers,
        train_speakers=train_speakers,
        test_speakers=test_speakers,
        spk2gen=spk2gen,
        cfg_pre=cfg_pre,
    )

    if unknown:
        logger.warning(f"split_speaker.txt に無い話者をスキップ: {unknown}")

    logger.info(f"処理対象: {len(tasks)} ファイル / {len(speaker_info)} 話者")
    if not tasks:
        logger.error("処理対象ファイルが見つかりません。--vctk_root を確認してください。")
        sys.exit(1)

    # --- DSP定数を事前計算 (メインプロセスで一度だけ) ---
    from librosa.filters import mel as librosa_mel

    mel_basis = librosa_mel(
        sr=cfg_audio["sample_rate"],
        n_fft=cfg_audio["n_fft"],
        fmin=cfg_audio["fmin"],
        fmax=cfg_audio["fmax"],
        n_mels=cfg_audio["n_mels"],
    ).T  # (n_fft//2+1, n_mels) = (513, 80)

    b, a = butter_highpass(
        cfg_pre["highpass_cutoff"],
        cfg_audio["sample_rate"],
        cfg_pre["highpass_order"],
    )

    # --- 並列処理 ---
    from tqdm import tqdm

    logger.info(f"並列処理開始 (workers={num_workers})")
    results: List[Dict[str, Any]] = []

    with multiprocessing.Pool(
        processes=num_workers,
        initializer=_worker_init,
        initargs=(mel_basis, b, a, cfg_pre["min_frames"]),
    ) as pool:
        for result in tqdm(
            pool.imap_unordered(_process_one_file, tasks),
            total=len(tasks),
            desc="前処理",
            unit="files",
        ):
            results.append(result)

    # --- 失敗ファイルを保存 ---
    failed = [r for r in results if not r["ok"]]
    if failed:
        failed_path = args.output_dir / "failed_files.txt"
        with open(failed_path, "w") as f:
            for r in failed:
                f.write(f"{r['wav_path']}\n  ERROR: {r['error'][:300]}\n\n")
        logger.warning(f"失敗ファイルリスト: {failed_path}")

    # --- metadata.json 生成 ---
    metadata = build_metadata(speaker_info, args.output_dir, train_speakers)
    metadata_path = args.output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"metadata.json 保存: {metadata_path}")

    # --- speaker_stats.json 生成 ---
    stats = build_speaker_stats(results, speaker_info)
    stats_path = args.output_dir / "speaker_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info(f"speaker_stats.json 保存: {stats_path}")

    # --- 最終サマリー ---
    n_ok = sum(1 for r in results if r["ok"])
    n_fail = len(failed)
    n_short = sum(1 for r in results if r["ok"] and r["short"])
    min_frames = cfg_pre["min_frames"]

    spk_counts: Dict[str, int] = {}
    for r in results:
        if r["ok"]:
            spk = Path(r["wav_path"]).parent.name
            spk_counts[spk] = spk_counts.get(spk, 0) + 1

    print("\n" + "=" * 60)
    print("前処理完了サマリー")
    print("=" * 60)
    print(f"  総ファイル数       : {len(tasks)}")
    print(f"  成功               : {n_ok}")
    print(f"  失敗               : {n_fail}")
    print(f"  短い (<{min_frames}fr)    : {n_short}")
    print(f"  train              : {len(metadata['train'])}")
    print(f"  val                : {len(metadata['val'])}")
    print(f"  test               : {len(metadata['test'])}")
    print()
    print("話者ごとの成功ファイル数:")
    for spk in sorted(spk_counts):
        info = speaker_info[spk]
        gender = info["gender"]
        spk_id = info["speaker_id"]
        cnt = spk_counts[spk]
        short = stats[spk]["short_files"]
        short_str = f" ({short} short)" if short else ""
        print(f"  {spk} (id={spk_id:3d}, {gender}): {cnt} files{short_str}")
    print("=" * 60)


if __name__ == "__main__":
    main()
