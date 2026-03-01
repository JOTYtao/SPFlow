
import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)
import torch
import torch.nn as nn
import numpy as np
import torch.cuda
from torch.autograd import Function
from tslearn.metrics.soft_dtw_loss_pytorch import _SoftDTWLossPyTorch
from numba import njit, prange

DBL_MAX = np.finfo("double").max
@njit(fastmath=True)
def _njit_softmin3(a, b, c, gamma):
    a /= -gamma
    b /= -gamma
    c /= -gamma
    max_val = max(max(a, b), c)
    tmp = 0
    tmp += np.exp(a - max_val)
    tmp += np.exp(b - max_val)
    tmp += np.exp(c - max_val)
    softmin_value = -gamma * (np.log(tmp) + max_val)
    return softmin_value


@njit(parallel=True)
def _njit_soft_dtw_batch(D, R, gamma):
    for i_sample in prange(D.shape[0]):
        _njit_soft_dtw(D[i_sample, :, :], R[i_sample, :, :], gamma)


@njit(fastmath=True)
def _njit_soft_dtw(D, R, gamma):
    m = D.shape[0]
    n = D.shape[1]
    # Initialization.
    R[: m + 1, 0] = DBL_MAX
    R[0, : n + 1] = DBL_MAX
    R[0, 0] = 0
    # DP recursion.
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # D is indexed starting from 0.
            R[i, j] = D[i - 1, j - 1] + _njit_softmin3(
                R[i - 1, j], R[i - 1, j - 1], R[i, j - 1], gamma
            )


@njit(parallel=True)
def _njit_soft_dtw_grad_batch(D, R, E, gamma):
    for i_sample in prange(D.shape[0]):
        _njit_soft_dtw_grad(D[i_sample, :, :], R[i_sample, :, :], E[i_sample, :, :], gamma)

@njit(fastmath=True)
def _njit_soft_dtw_grad(D, R, E, gamma):
    m = D.shape[0] - 1
    n = D.shape[1] - 1
    # Initialization.
    D[:m, n] = 0
    D[m, :n] = 0
    R[1 : m + 1, n + 1] = -DBL_MAX
    R[m + 1, 1 : n + 1] = -DBL_MAX

    E[m + 1, n + 1] = 1
    R[m + 1, n + 1] = R[m, n]
    D[m, n] = 0
    for j in range(n, 0, -1):  # ranges from n to 1
        for i in range(m, 0, -1):  # ranges from m to 1
            a = np.exp((R[i + 1, j] - R[i, j] - D[i, j - 1]) / gamma)
            b = np.exp((R[i, j + 1] - R[i, j] - D[i - 1, j]) / gamma)
            c = np.exp((R[i + 1, j + 1] - R[i, j] - D[i, j]) / gamma)
            E[i, j] = E[i + 1, j] * a + E[i, j + 1] * b + E[i + 1, j + 1] * c

class _SoftDTWLossPyTorch(Function):
    @staticmethod
    def forward(ctx, D, gamma):
            dev = D.device
            dtype = D.dtype
            b, m, n = torch.Tensor.size(D)
            D_ = D.detach().cpu().numpy()
            R_ = np.zeros((b, m + 2, n + 2), dtype=np.float64)
            _njit_soft_dtw_batch(D_, R_, gamma)
            gamma_tensor = torch.Tensor([gamma]).to(dev).type(dtype)
            R = torch.Tensor(R_).to(dev).type(dtype)
            ctx.save_for_backward(D, R, gamma_tensor)
            return R[:, -2, -2]

    @staticmethod
    def backward(ctx, grad_output):
            dev = grad_output.device
            dtype = grad_output.dtype
            D, R, gamma_tensor = ctx.saved_tensors
            b, m, n = torch.Tensor.size(D)
            D_ = D.detach().cpu().numpy()
            R_ = R.detach().cpu().numpy()
            E_ = np.zeros((b, m + 2, n + 2), dtype=np.float64)
            gamma = gamma_tensor.item()
            _njit_soft_dtw_grad_batch(D_, R_, E_, gamma)
            E = torch.Tensor(E_[:, 1 : m + 1, 1 : n + 1]).to(dev).type(dtype)
            return grad_output.view(-1, 1, 1).expand_as(E) * E, None, None
class SoftDTWLossPyTorch(nn.Module):
    def __init__(self, gamma=0.001, normalize=True, dist_func=None):
            super(SoftDTWLossPyTorch, self).__init__()
            self.gamma = gamma
            self.normalize = normalize
            if dist_func is not None:
                self.dist_func = dist_func
            else:
                self.dist_func = SoftDTWLossPyTorch._euclidean_squared_dist

    @staticmethod
    def _euclidean_squared_dist(x, y):
            m = x.size(1)
            n = y.size(1)
            d = x.size(2)
            x = x.unsqueeze(2).expand(-1, m, n, d)
            y = y.unsqueeze(1).expand(-1, m, n, d)
            return torch.pow(x - y, 2).sum(3)

    def forward(self, x, y):
            bx, lx, dx = x.shape
            by, ly, dy = y.shape
            assert bx == by
            assert dx == dy
            d_xy = self.dist_func(x, y)
            loss_xy = _SoftDTWLossPyTorch.apply(d_xy, self.gamma)
            if self.normalize:
                d_xx = self.dist_func(x, x)
                d_yy = self.dist_func(y, y)
                loss_xx = _SoftDTWLossPyTorch.apply(d_xx, self.gamma)
                loss_yy = _SoftDTWLossPyTorch.apply(d_yy, self.gamma)
                total_loss = loss_xy - 0.5 * (loss_xx + loss_yy)
            else:
                total_loss = loss_xy
            mean_loss = total_loss.mean()
            return mean_loss