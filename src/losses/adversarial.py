"""
VC-Robust Neural Audio Watermarking — Adversarial Loss

L_MI_min: GRL + SpeakerDiscriminator で z_c / z_c_hat から話者情報を排除する。

1回の loss.backward() で以下が同時に正しく更新される:
  - SpeakerDiscriminator (正の勾配 → 話者分類精度向上)
  - ContentEncoder / WatermarkExtractor前段 (GRL経由の負の勾配 → 話者情報排除)

参照:
  SRD-VC/My_model/pytorch_revgrad/functional.py — RevGrad Function (line 1-18)
  SRD-VC/My_model/pytorch_revgrad/module.py    — RevGrad Module (line 1-17)
  SRD-VC/My_model/AdversarialClassifier.py     — LinearNorm + AdversarialClassifier (line 11-36)
"""

from typing import Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Gradient Reversal Layer
# ---------------------------------------------------------------------------

class _RevGradFunction(torch.autograd.Function):
    """GRL の autograd.Function 実装。
    参照: SRD-VC/My_model/pytorch_revgrad/functional.py (line 3-16)
    """

    @staticmethod
    def forward(ctx: Any, input_: torch.Tensor, alpha_: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(alpha_)
        return input_.clone()

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):
        alpha_, = ctx.saved_tensors
        return -grad_output * alpha_, None


class GradientReversalLayer(nn.Module):
    """GRL: 順伝播は恒等写像、逆伝播で勾配を -alpha 倍する。

    参照: SRD-VC/My_model/pytorch_revgrad/module.py (line 5-17)

    Args:
        alpha: 勾配反転係数 λ (default 1.0、固定値)
    """

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_: (B, D) 任意の特徴量
        Returns:
            (B, D) 恒等写像（逆伝播時のみ勾配を -alpha 倍）
        """
        return _RevGradFunction.apply(input_, self.alpha)


# ---------------------------------------------------------------------------
# Speaker Discriminator
# ---------------------------------------------------------------------------

class SpeakerDiscriminator(nn.Module):
    """GRL + 2層MLP による話者分類器。

    参照: SRD-VC/My_model/AdversarialClassifier.py LinearNorm + AdversarialClassifier (line 11-36)

    SRD-VCオリジナルとの差異:
      - GRLを独立クラスとして外部化（条件分岐なし、常にGRL適用）
      - 入力は GAP 済みの 2D テンソル (B, input_dim) を受け取る
      - 活性化関数なし・Dropoutなし（SRD-VC準拠）

    Args:
        input_dim:   入力特徴次元 (default 16 = D_c)
        num_classes: 話者クラス数 (default 110 = VCTK話者数)
        hidden_dim:  中間層次元 (default 256、SRD-VC準拠)
        alpha:       GRL 勾配反転係数 (default 1.0)
    """

    def __init__(
        self,
        input_dim: int = 16,
        num_classes: int = 110,
        hidden_dim: int = 256,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.grl = GradientReversalLayer(alpha)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

        # Xavier uniform 初期化（SRD-VC LinearNorm準拠）
        nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('linear'))
        nn.init.xavier_uniform_(self.fc2.weight, gain=nn.init.calculate_gain('linear'))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) GAP済みの特徴量
        Returns:
            logits: (B, num_classes) sigmoid/softmax 前のlogits
        """
        x = self.grl(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


# ---------------------------------------------------------------------------
# Adversarial Loss
# ---------------------------------------------------------------------------

class AdversarialLoss(nn.Module):
    """L_MI_min: z_c / z_c_hat に独立した GRL+SpeakerDiscriminator を適用する敵対的損失。

    z_c (ContentEncoder出力) と z_c_hat (WatermarkExtractor前段出力) は
    生成元が異なるため、別々のDiscriminatorで独立して話者情報を排除する。

    事前学習weights:
      800000-G.ckpt の adversarial1 は input_dim=82 / num_classes=100 で
      本プロジェクトの次元設定 (D_c=16, num_classes=110) と不一致のため
      ランダム初期化で開始する。

    Args:
        d_c:         Content特徴次元 (default 16 = dim_neck*2)
        num_classes: 話者クラス数 (default 110 = VCTKの実際の話者数)
                     ※ /workspace/wav48_silence_trimmed のサブディレクトリ数で最終確認。
                        DataLoader実装時に不一致があれば調整すること。
        hidden_dim:  SpeakerDiscriminator MLP 中間層次元 (default 256、SRD-VC準拠)
        alpha:       GRL 勾配反転係数 λ (default 1.0、SRD-VC RevGrad default準拠)
    """

    def __init__(
        self,
        d_c: int = 16,
        num_classes: int = 110,
        hidden_dim: int = 256,
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        self.disc_c = SpeakerDiscriminator(d_c, num_classes, hidden_dim, alpha)
        self.disc_c_hat = SpeakerDiscriminator(d_c, num_classes, hidden_dim, alpha)

    def forward(
        self,
        z_c: torch.Tensor,
        z_c_hat: torch.Tensor,
        speaker_id: torch.Tensor,
    ) -> torch.Tensor:
        """1回のloss.backward()でDiscriminatorとEncoder/Extractorを同時に更新する。

        Discriminatorは正の勾配（CrossEntropy最小化 → 話者分類精度向上）、
        ContentEncoder/WatermarkExtractorはGRL経由の負の勾配（話者情報排除）を受け取る。

        Args:
            z_c:        (B, T', D_c) full_model.forward()['z_c']
            z_c_hat:    (B, T', D_c) full_model.forward()['z_c_hat']
            speaker_id: (B,) int64 DataLoader提供の正解話者ラベル
        Returns:
            loss: scalar（trainer側で weight を乗算して g_loss に加算）
        """
        # Global Average Pooling: (B, T', D_c) → (B, D_c)
        z_c_avg = z_c.mean(dim=1)
        z_c_hat_avg = z_c_hat.mean(dim=1)

        # GRL → MLP → (B, num_classes)
        logits_c = self.disc_c(z_c_avg)
        logits_c_hat = self.disc_c_hat(z_c_hat_avg)

        loss = (
            F.cross_entropy(logits_c, speaker_id) +
            F.cross_entropy(logits_c_hat, speaker_id)
        )
        return loss

    def get_discriminator_parameters(self) -> List[nn.Parameter]:
        """Optimizer_D に渡す SpeakerDiscriminator のパラメータリスト。

        GRLのalphaはrequires_grad=Falseのため含まれない。
        """
        return list(self.disc_c.parameters()) + list(self.disc_c_hat.parameters())
