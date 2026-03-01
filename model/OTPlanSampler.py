# ot_sampler.py
import math
from functools import partial
from typing import Optional, Tuple

import numpy as np
import ot as pot
import torch


class OTPlanSampler:
    """OTPlanSampler implements sampling coordinates according to a squared L2 OT plan
    with different implementations of the plan calculation.
    Methods:
      - exact:     EMD (linear program / Hungarian), high accuracy, slower on big batches
      - sinkhorn:  Entropic regularized OT, faster, needs reg
      - unbalanced: Unbalanced Sinkhorn (mass not preserved), needs reg, reg_m
      - partial:    Partial OT, allows unmatched mass, needs reg
    """

    def __init__(
        self,
        method: str = "sinkhorn",
        reg: float = 0.05,
        reg_m: float = 1.0,
        normalize_cost: bool = True,
        **kwargs,
    ):
        if method == "exact":
            self.ot_fn = pot.emd
        elif method == "sinkhorn":
            self.ot_fn = partial(pot.sinkhorn, reg=reg)
        elif method == "unbalanced":
            self.ot_fn = partial(pot.unbalanced.sinkhorn_knopp_unbalanced, reg=reg, reg_m=reg_m)
        elif method == "partial":
            self.ot_fn = partial(pot.partial.entropic_partial_wasserstein, reg=reg)
        else:
            raise ValueError(f"Unknown method: {method}")
        self.method = method
        self.reg = reg
        self.reg_m = reg_m
        self.normalize_cost = normalize_cost
        self.kwargs = kwargs

    @staticmethod
    def _flatten_if_needed(x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            return x.reshape(x.shape[0], -1)
        return x

    def get_map(self, x0: torch.Tensor, x1: torch.Tensor) -> np.ndarray:
        """Return the OT plan matrix π (shape [B, B]) on CPU (numpy)."""
        # Uniform marginals
        a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
        # Flatten to [B, D]
        X0 = self._flatten_if_needed(x0)
        X1 = self._flatten_if_needed(x1)
        # Cost matrix (squared L2)
        M = torch.cdist(X0, X1) ** 2  # [B,B]
        if self.normalize_cost:
            M = M / (M.max() + 1e-12)

        # Call POT (expects numpy arrays)
        pi = self.ot_fn(a, b, M.detach().cpu().numpy(), **self.kwargs)
        if not np.all(np.isfinite(pi)):
            print("ERROR: OT plan contains non-finite values.")
            print("Cost mean, max", float(M.mean()), float(M.max()))
        return pi

    @staticmethod
    def sample_map(pi: np.ndarray, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample (i,j) pairs from the OT plan π."""
        p = pi.reshape(-1)
        p = p / (p.sum() + 1e-12)
        choices = np.random.choice(pi.size, p=p, size=batch_size)
        i, j = np.divmod(choices, pi.shape[1])
        return i, j

    def sample_plan(self, x0: torch.Tensor, x1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return paired samples (x0[i], x1[j]) according to plan π."""
        pi = self.get_map(x0, x1)
        i, j = self.sample_map(pi, x0.shape[0])
        return x0[i], x1[j]

    def sample_plan_indices(self, x0: torch.Tensor, x1: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Return paired indices (i, j) according to plan π (more efficient for GPU)."""
        pi = self.get_map(x0, x1)
        i, j = self.sample_map(pi, x0.shape[0])
        return i, j


def wasserstein(
    x0: torch.Tensor,
    x1: torch.Tensor,
    method: Optional[str] = None,
    reg: float = 0.05,
    power: int = 2,
    **kwargs,
) -> float:
    """Compute empirical Wasserstein distance between two batches."""
    assert power in (1, 2)
    if method == "exact" or method is None:
        ot_fn = pot.emd2
    elif method == "sinkhorn":
        ot_fn = partial(pot.sinkhorn2, reg=reg)
    else:
        raise ValueError(f"Unknown method: {method}")

    X0 = x0.reshape(x0.shape[0], -1) if x0.dim() > 2 else x0
    X1 = x1.reshape(x1.shape[0], -1) if x1.dim() > 2 else x1

    a, b = pot.unif(X0.shape[0]), pot.unif(X1.shape[0])
    M = torch.cdist(X0, X1)
    if power == 2:
        M = M ** 2
    ret = ot_fn(a, b, M.detach().cpu().numpy(), numItermax=10_000)
    if power == 2:
        ret = math.sqrt(ret)
    return float(ret)