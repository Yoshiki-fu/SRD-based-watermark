"""
VC-Robust Neural Audio Watermarking — vCLUB Loss

L_disentangle: vCLUB で Content(z_c)・Rhythm(z_r)・F0(z_f) の
ペア間相互情報量を最小化し、ContentEncoder の分離性を維持する。

2段階更新が必要:
  1. Optimizer_D: estimator_loss() で変分近似ネットワークを更新
  2. Optimizer_G: vclub_loss() で ContentEncoder を分離方向に更新

参照:
  SRD-VC/My_model/mi_estimators.py — CLUBSample_reshape (line 102)
  SRD-VC/My_model/solver.py — mi_first_forward / mi_est 呼び出し (line 145, 313)
"""

from typing import List, Optional

import torch
import torch.nn as nn


class _CLUBSampleReshape(nn.Module):
    """変分近似ネットワーク q_θ(y|x)。3D入力 (B, T, dim) に対応する vCLUB 推定器。

    参照: SRD-VC/My_model/mi_estimators.py CLUBSample_reshape (line 102–147)

    Args:
        x_dim:       入力 x の特徴次元
        y_dim:       入力 y の特徴次元（推定対象）
        hidden_size: MLP 隠れ層次元（各層は hidden_size // 2）
    """

    def __init__(self, x_dim: int, y_dim: int, hidden_size: int) -> None:
        super().__init__()
        h = hidden_size // 2
        # q(y|x) の平均を出力するブランチ
        self.p_mu = nn.Sequential(
            nn.Linear(x_dim, h), nn.ReLU(),
            nn.Linear(h, h),     nn.ReLU(),
            nn.Linear(h, h),     nn.ReLU(),
            nn.Linear(h, y_dim),
        )
        # q(y|x) の log 分散を出力するブランチ
        self.p_logvar = nn.Sequential(
            nn.Linear(x_dim, h), nn.ReLU(),
            nn.Linear(h, h),     nn.ReLU(),
            nn.Linear(h, h),     nn.ReLU(),
            nn.Linear(h, y_dim), nn.Tanh(),
        )

    def get_mu_logvar(
        self,
        x_samples: torch.Tensor,
    ) -> tuple:
        """
        Args:
            x_samples: (B, T, x_dim)
        Returns:
            mu:     (B, T, y_dim)
            logvar: (B, T, y_dim)
        """
        return self.p_mu(x_samples), self.p_logvar(x_samples)

    def loglikeli(
        self,
        x_samples: torch.Tensor,
        y_samples: torch.Tensor,
    ) -> torch.Tensor:
        """変分近似の非正規化対数尤度（推定器更新用）。

        Args:
            x_samples: (B, T, x_dim)
            y_samples: (B, T, y_dim)
        Returns:
            scalar  (最大化する → -loglikeli を Optimizer_D で最小化)
        """
        mu, logvar = self.get_mu_logvar(x_samples)  # (B, T, y_dim)
        # (B, T, y_dim) → (B*T, y_dim)
        mu = mu.reshape(-1, mu.shape[-1])
        logvar = logvar.reshape(-1, logvar.shape[-1])
        y_samples = y_samples.reshape(-1, y_samples.shape[-1])
        return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0)

    def mi_est(
        self,
        x_samples: torch.Tensor,
        y_samples: torch.Tensor,
    ) -> torch.Tensor:
        """vCLUB 上界（メインネット更新用）。

        負例はバッチ次元のランダム置換で構成する（時間軸を超えない）。
        置換後に (B, T, dim) → (B*T, dim) へ flatten する。

        Args:
            x_samples: (B, T, x_dim)
            y_samples: (B, T, y_dim)
        Returns:
            scalar  (最小化により I(x; y) を間接的に低下させる)
        """
        mu, logvar = self.get_mu_logvar(x_samples)  # (B, T, y_dim)

        # バッチ次元でシャッフル（時間軸は保持）
        B = mu.shape[0]
        random_index = torch.randperm(B, device=x_samples.device).long()
        y_shuffle = y_samples[random_index]  # (B, T, y_dim)

        # シャッフル後に flatten
        mu = mu.reshape(-1, mu.shape[-1])            # (B*T, y_dim)
        logvar = logvar.reshape(-1, logvar.shape[-1])
        y_samples = y_samples.reshape(-1, y_samples.shape[-1])
        y_shuffle = y_shuffle.reshape(-1, y_shuffle.shape[-1])

        positive = -(mu - y_samples) ** 2 / logvar.exp()
        negative = -(mu - y_shuffle) ** 2 / logvar.exp()
        upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        return upper_bound / 2.0


class VCLUBLoss(nn.Module):
    """L_disentangle: vCLUB による Content/Rhythm/F0 ペア間 MI 最小化。

    実装する推定器:
      - cp_mi_net: I(z_c; z_f) 推定。x=z_c, y=z_f
      - rc_mi_net: I(z_r; z_c) 推定。x=z_r, y=z_c

    rp_mi_net (z_r, z_f) は Phase 1 では省略する。
    z_r と z_f は両方凍結されており、vclub_loss() でメインネットへの
    勾配が一切流れないため。include_rp=True で将来拡張可能。

    事前学習 Weights:
      800000-G.ckpt 内の cp_mi_net / rc_mi_net は本プロジェクトの
      次元設定と完全互換のため、ckpt_path を指定するとロードする。

    Args:
        d_c:              Content 特徴次元 (default 16)
        d_r:              Rhythm 特徴次元 (default 2)
        d_f:              F0 特徴次元 (default 64)
        hidden_size:      MLP 隠れ層次元 (default 512, SRD-VC 準拠)
        include_rp:       rp_mi_net (z_r, z_f) を含めるか (default False)
        ckpt_path:        800000-G.ckpt へのパス。None のとき重みロードをスキップ
        load_pretrained:  ckpt_path 指定時に重みをロードするか (default True)
    """

    def __init__(
        self,
        d_c: int = 16,
        d_r: int = 2,
        d_f: int = 64,
        hidden_size: int = 512,
        include_rp: bool = False,
        ckpt_path: Optional[str] = None,
        load_pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.cp_mi_net = _CLUBSampleReshape(d_c, d_f, hidden_size)
        self.rc_mi_net = _CLUBSampleReshape(d_r, d_c, hidden_size)

        self.include_rp = include_rp
        if include_rp:
            self.rp_mi_net: Optional[_CLUBSampleReshape] = _CLUBSampleReshape(
                d_r, d_f, hidden_size
            )
        else:
            self.rp_mi_net = None

        if ckpt_path is not None and load_pretrained:
            _load_pretrained_vclub(ckpt_path, self.cp_mi_net, self.rc_mi_net)

    # ------------------------------------------------------------------
    # 推定器パラメータ
    # ------------------------------------------------------------------

    def get_estimator_parameters(self) -> List[nn.Parameter]:
        """Optimizer_D に渡す MI 推定器のパラメータリスト。"""
        params = list(self.cp_mi_net.parameters()) + list(self.rc_mi_net.parameters())
        if self.include_rp and self.rp_mi_net is not None:
            params += list(self.rp_mi_net.parameters())
        return params

    # ------------------------------------------------------------------
    # 2段階更新メソッド
    # ------------------------------------------------------------------

    def estimator_loss(
        self,
        z_c: torch.Tensor,
        z_r: torch.Tensor,
        z_f: torch.Tensor,
    ) -> torch.Tensor:
        """Optimizer_D 用: -loglikeli の合算（SRD-VC mi_first_forward 準拠）。

        全入力を .detach() して使用し、推定器更新時に
        ContentEncoder への不要な勾配計算グラフ構築を防ぐ。

        Args:
            z_c: (B, T', D_c)  透かし混入前 Content codes
            z_r: (B, T', D_r)  Rhythm codes（凍結）
            z_f: (B, T', D_f)  F0 codes（凍結）
        Returns:
            scalar（最小化 → 推定器の変分近似精度が向上）
        """
        z_c_ = z_c.detach()
        z_r_ = z_r.detach()
        z_f_ = z_f.detach()

        lld_cp = -self.cp_mi_net.loglikeli(z_c_, z_f_)
        lld_rc = -self.rc_mi_net.loglikeli(z_r_, z_c_)
        loss = lld_cp + lld_rc

        if self.include_rp and self.rp_mi_net is not None:
            loss = loss + (-self.rp_mi_net.loglikeli(z_r_, z_f_))

        return loss

    def vclub_loss(
        self,
        z_c: torch.Tensor,
        z_r: torch.Tensor,
        z_f: torch.Tensor,
    ) -> torch.Tensor:
        """Optimizer_G 用: vCLUB 上界の合算（SRD-VC g_loss 加算部分準拠）。

        z_c には勾配が流れ ContentEncoder を分離方向に学習する。
        z_r / z_f は凍結済み (requires_grad=False) のため detach 不要。

        Args:
            z_c: (B, T', D_c)  透かし混入前 Content codes（学習可能）
            z_r: (B, T', D_r)  Rhythm codes（凍結）
            z_f: (B, T', D_f)  F0 codes（凍結）
        Returns:
            scalar（最小化 → I(z_c, z_f) と I(z_r, z_c) を下げる）
        """
        mi_cp = self.cp_mi_net.mi_est(z_c, z_f)
        mi_rc = self.rc_mi_net.mi_est(z_r, z_c)
        loss = mi_cp + mi_rc

        if self.include_rp and self.rp_mi_net is not None:
            loss = loss + self.rp_mi_net.mi_est(z_r, z_f)

        return loss

    def forward(
        self,
        z_c: torch.Tensor,
        z_r: torch.Tensor,
        z_f: torch.Tensor,
    ) -> torch.Tensor:
        """vclub_loss() のエイリアス（Optimizer_G フロー用）。

        Args:
            z_c: (B, T', D_c)
            z_r: (B, T', D_r)
            z_f: (B, T', D_f)
        Returns:
            scalar
        """
        return self.vclub_loss(z_c, z_r, z_f)


def _load_pretrained_vclub(
    ckpt_path: str,
    cp_mi_net: _CLUBSampleReshape,
    rc_mi_net: _CLUBSampleReshape,
    strict: bool = True,
) -> None:
    """800000-G.ckpt から cp_mi_net / rc_mi_net の重みをロードする。

    Args:
        ckpt_path: チェックポイントファイルのパス
        cp_mi_net: cp_mi_net インスタンス
        rc_mi_net: rc_mi_net インスタンス
        strict:    load_state_dict の strict フラグ
    """
    ckpt = torch.load(ckpt_path, map_location='cpu')
    cp_mi_net.load_state_dict(ckpt['cp_mi_net'], strict=strict)
    rc_mi_net.load_state_dict(ckpt['rc_mi_net'], strict=strict)
