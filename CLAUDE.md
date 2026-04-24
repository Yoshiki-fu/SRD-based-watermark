# VC-Robust Neural Audio Watermarking (Phase 1: PoC)

## プロジェクト概要
Self Voice Conversion（自己声質変換）攻撃に耐性を持つ
音声電子透かしのPoC実装。透かし情報をVCが保持する
Content潜在表現に交絡（Entangle）させるアプローチ。

## 研究上の位置づけ
- ベース: SRDVC (Interspeech 2022, Yang et al.)の分離アーキテクチャ
- 攻撃モデル: StarGANv2-VC（Phase 1評価時のみ。学習ループには入れない）
- 新規性: InfoNCEによるMI最大化 + vCLUB/GRLによるMI最小化で
  透かしをContent空間にのみ交絡させるLoss設計

## 技術スタック
- Python 3.8
- PyTorch >= 2.0
- torchaudio（メルスペクトログラム計算、音声I/O）
- データ: VCTK corpus, /workspace/wav48_silence_trimmed にある

## ディレクトリ構成
````
vc-watermark/
├── CLAUDE.md
├── SRD-VC/                # 参照専用：SRDVCオリジナル実装
│    └── My_model/
│         └── my_demo/
│              ├── 640000-P.ckpt                      # pre-trained pitch decoder
│              ├── 800000-G.ckpt                      # pre-trained SRD-VC (Generator)
│              └── checkpoint_step001000000_ema.pth   # pre-trained vocoder
├── configs/
│   └── default.yaml       # ハイパーパラメータ
├── src/
│   ├── models/
│   │   ├── encoders.py     # Content/Speaker/Rhythm/F0 Encoder + F0Quantizer
│   │   ├── watermark.py    # WM Encoder, Extractor, Fusion Layer
│   │   ├── decoder.py      # Mel Decoder（SRD-VCオリジナル流用）
│   │   └── full_model.py   # 統合モデル
│   ├── losses/
│   │   ├── reconstruction.py
│   │   ├── watermark_loss.py
│   │   ├── info_nce.py     # MI最大化
│   │   ├── vclub.py        # MI最小化（vCLUB）
│   │   └── adversarial.py  # GRL + Speaker Discriminator
│   ├── attacks/
│   │   └── distortion.py   # 微分可能な攻撃レイヤー
│   ├── training/
│   │   └── trainer.py      # training_step, eval
│   └── utils/
└── tests/
    └── test_shapes.py      # shape一貫性テスト
````

## SRD-VC リファレンス実装の扱い
- `SRD-VC/` 以下にはSRDVCのオリジナル実装（Yang et al., Interspeech 2022）が
  配置されている
- Content/Speaker/Rhythm/F0 Encoder、Decoder、vCLUB、GRLの実装方針は
  まずこのリファレンスを参照すること
- ただし以下を厳守:
  - `SRD-VC/` 以下のファイルは**読み取り専用**（編集・移動禁止）
  - コードをコピーする場合も、`src/` 以下に新規ファイルとして実装し、
    どのファイルを参考にしたかをdocstringに記載
  - リファレンスのコードをそのまま流用するのではなく、本プロジェクトの
    型ヒント・docstring規約に従ってリライトする

## コーディング規約
- 型ヒント必須（Tensor shape情報はdocstringに記載）
- 型ヒントはPython 3.8互換で記述:
  - `Union[int, str]` (NOT `int | str`)
  - `List[int]` from typing (NOT `list[int]`)
  - `Optional[X]` from typing
````python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: (B, 80, T) melスペクトログラム
    Returns:
        z_c: (B, T', D_c) Content特徴量
    """
````
- Lossクラスは1ファイル1クラス、各自forward()でスカラーを返す
- ハイパーパラメータはハードコードせずconfigs/から読む
- 日本語コメント可

## 設計上の重要な制約

### 事前学習weightsの利用
- `SRD-VC/My_model/my_demo/800000-G.ckpt` から以下をロード:
  - Content Encoder: 学習可能（低学習率 lr=1e-5）
  - Speaker Encoder: 凍結 (requires_grad=False)
  - Rhythm Encoder: 凍結 (requires_grad=False)
  - F0 Encoder: 凍結 (requires_grad=False)
  - Decoder: 学習可能（低学習率 lr=1e-5、fine-tuning）
- 凍結の理由: クリーン音声入力のため透かしをフィルタリングする
  学習が不可能。事前学習済み「正解抽出能力」をアンカーとする
- `640000-P.ckpt` (pitch decoder), `checkpoint_step001000000_ema.pth` (vocoder)
  はPhase 1では使用しない

### Phase 1のスコープ
- Decoderはメルスペクトログラム (B, 80, 192) を出力
  （vocoderによる波形生成は行わない）
- 微分可能な攻撃レイヤー、Watermark Extractorは
  すべてメルスペクトログラム上で動作
- Phase 2以降でvocoderと波形ベース攻撃（kNN-VC等）への拡張を検討

## Phase 2前処理の方針
- SRD-VCの前処理を参考にする
- メル/F0の事前計算を scripts/preprocess_vctk.py で実装
- 出力形式は .npy または .pt で /workspace/ 配下に保存
- VCTK全44,455ファイル（110話者）が対象

### 攻撃カリキュラム（微分可能な攻撃レイヤー）
- Epoch 1-5: 攻撃なし（クリーンな再構築と抽出）
- Epoch 6-15: 軽微なガウシアンノイズ、Dropout
- Epoch 16-: SpecAugment（Time/Frequency masking）、
  ダウン・アップサンプリングによる情報ボトルネック
- 注: StarGANv2-VC等の重いVCモデルは学習ループに入れず、
  Phase 1評価時のみ適用

### データローダー仕様
- 返却形式: `(mel, f0_norm, speaker_id)`
- mel: (80, 192) チャンネルファースト、pre-computed
- f0_norm: (192,) 話者正規化済み、pre-computed
- speaker_id: int、Speaker Discriminator の正解ラベル用
- F0とメルは事前計算（オフラインまたはデータローダー側）。
  学習ループ内で毎回の抽出は行わない

## ビルドとテスト
- テスト実行: `python -m pytest tests/`（実装後に追加）
- 学習開始: 未実装（trainer完成後に追加予定）

## 現在の進捗
- [x] エンコーダ群（Content/Speaker/Rhythm/F0）+ 事前学習weightsロード
- [x] Watermark Encoder + Fusion Layer + Watermark Extractor
- [x] Decoder（SRD-VCオリジナル流用、事前学習weightsロード）
- [×] full_model.py（統合モデル）
- [×] 微分可能な攻撃レイヤー（カリキュラム対応）
- [x] 基本Loss（Reconstruction L1、Watermark BCE+BER）
- [x] InfoNCE Loss
- [x] vCLUB Loss（cp_mi_net + rc_mi_net、推定器weightsロード済み）
- [×] Adversarial Loss（GRL + Speaker Discriminator）
- [x] MockVCTKDataset（Phase 1動作確認用、無声20%含む）
- [×] training_step（2 Optimizer交互更新、中程度スコープ、checkpoint機能付き）
- [x] VCTK前処理スクリプト（SRD-VC完全準拠、per-utterance F0正規化、可変長保存）
- [x] 前処理実行完了（87,532ファイル：train 79,242 / val 500 / test 7,790）
- [ ] 実VCTK DataLoader
- [ ] overfit検証（1サンプル）
- [ ] フル学習- [ ] VCTK前処理スクリプト（mel/f0事前計算、SRD-VC準拠）
- [ ] 実VCTK DataLoader
- [ ] overfit検証（1サンプル）
- [ ] フル学習

