import torch

class Interpolant:

    def __init__(self, sigma_coef: float = 1.0, beta_fn: str = "t"):
        self.sigma_coef = float(sigma_coef)
        assert beta_fn in ["t", "t^2"]
        self.beta_fn = beta_fn

    def wide(self, t: torch.Tensor, x: torch.Tensor):
        if t.dim() == 1:
            shape = [x.size(0)] + [1] * (x.dim() - 1)
            return t.view(-1, *([1] * (x.dim() - 1)))
        else:
            return t.view(x.size(0), *([1] * (x.dim() - 1)))

    def alpha(self, t, x):
        # α_t = 1 - t
        return self.wide(1.0 - t, x)

    def alpha_dot(self, t, x):
        # d/dt α_t = -1
        return self.wide(-torch.ones_like(t), x)

    def beta(self, t, x):
        # β_t = t or t^2
        if self.beta_fn == "t^2":
            val = t.pow(2)
        else:
            val = t
        return self.wide(val, x)

    def beta_dot(self, t, x):
        # d/dt β_t
        if self.beta_fn == "t^2":
            val = 2.0 * t
        else:
            val = torch.ones_like(t)
        return self.wide(val, x)

    def sigma(self, t, x):
        # σ_t = sigma_coef * (1 - t)
        return self.sigma_coef * self.wide(1.0 - t, x)

    def sigma_dot(self, t, x):
        # d/dt σ_t = -sigma_coef
        return self.sigma_coef * self.wide(-torch.ones_like(t), x)

    def gamma(self, t, x):
        # γ_t = sqrt(t) * σ_t
        root_t = self.wide(t.sqrt(), x)
        return root_t * self.sigma(t, x)

    def sample_location_and_conditional_flow(self, x0, x1, t=None):
        B = x0.size(0)
        device = x0.device
        dtype = x0.dtype
        if t is None:
            t_scalar = torch.rand(B, device=device, dtype=dtype)  # U(0,1)
        else:
            if t.dim() > 1:
                t_scalar = t.view(B).to(device=device, dtype=dtype)
            else:
                t_scalar = t.to(device=device, dtype=dtype)
        noise = torch.randn_like(x0)

        t_broadcast = t_scalar  # (B,)
        at = self.alpha(t_broadcast, x0)      # (B,1,1,1,1) -> broadcast
        bt = self.beta(t_broadcast, x0)
        adot = self.alpha_dot(t_broadcast, x0)
        bdot = self.beta_dot(t_broadcast, x0)
        st = self.sigma(t_broadcast, x0)
        sdot = self.sigma_dot(t_broadcast, x0)
        root_t = self.wide(t_broadcast.sqrt(), x0)
        gamma_t = self.gamma(t_broadcast, x0)
        x_t = at * x0 + bt * x1 + gamma_t * noise
        u_t = adot * x0 + bdot * x1 + (sdot * root_t) * noise

        return t_scalar, x_t, u_t