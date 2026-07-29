from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PoseTimeKernelConfig:
    """RBF bandwidths for GIM-World equation (16).

    The paper does not publish numeric bandwidths.  A non-positive position
    bandwidth enables a scene-adaptive median pairwise distance.  Rotation is
    in radians and time is in latent-frame indices.
    """

    sigma_position: float = -1.0
    sigma_rotation: float = math.pi / 6.0
    sigma_time: float = 50.0
    jitter: float = 1e-5


def _median_nonzero(values: torch.Tensor, fallback: float) -> torch.Tensor:
    values = values[torch.isfinite(values) & (values > 1e-6)]
    if values.numel() == 0:
        return torch.as_tensor(fallback, dtype=torch.float64, device=values.device)
    return values.median().clamp_min(1e-6)


def pose_time_kernel(
    c2w: torch.Tensor,
    times: torch.Tensor,
    config: PoseTimeKernelConfig,
) -> torch.Tensor:
    if c2w.ndim != 3 or c2w.shape[-2:] != (4, 4):
        raise ValueError(f"expected [N,4,4] c2w, got {tuple(c2w.shape)}")
    if times.shape != (c2w.shape[0],):
        raise ValueError("times must have one value per camera")
    pose = c2w.double()
    position = pose[:, :3, 3]
    forward = torch.nn.functional.normalize(pose[:, :3, 2], dim=-1, eps=1e-12)
    distance = torch.cdist(position, position)
    dot = (forward @ forward.T).clamp(-1.0, 1.0)
    angle = torch.acos(dot)
    time_distance = (times.double()[:, None] - times.double()[None, :]).abs()

    sigma_p = (
        torch.as_tensor(config.sigma_position, dtype=torch.float64, device=c2w.device)
        if config.sigma_position > 0
        else _median_nonzero(distance, 1.0)
    )
    sigma_r = max(float(config.sigma_rotation), 1e-6)
    sigma_t = max(float(config.sigma_time), 1e-6)
    exponent = -(
        distance.square() / (2.0 * sigma_p.square())
        + angle.square() / (2.0 * sigma_r**2)
        + time_distance.square() / (2.0 * sigma_t**2)
    )
    kernel = exponent.exp()
    kernel.diagonal().fill_(1.0 + float(config.jitter))
    return kernel


class MIGreedyPruner:
    """Exact greedy implementation of GIM-World equations (15)-(18).

    The implementation keeps both required covariance terms up to date with
    rank-one Schur-complement downdates, avoiding a matrix inverse per
    candidate.  It is still the literal log posterior-variance ratio, rather
    than camera FPS or uniform sampling.
    """

    def __init__(self, config: PoseTimeKernelConfig) -> None:
        self.config = config

    @torch.no_grad()
    def select(
        self,
        c2w: torch.Tensor,
        times: torch.Tensor,
        budget: int,
    ) -> torch.Tensor:
        count = int(c2w.shape[0])
        if budget <= 0:
            raise ValueError("budget must be positive")
        if count <= budget:
            return torch.arange(count, device=c2w.device, dtype=torch.long)

        kernel = pose_time_kernel(c2w, times, self.config)
        # C is Cov(U | S), initially Cov(H).  inv_u is K_UU^-1 and gives
        # Var(h | U\{h}) = 1 / inv_u[h,h].
        conditional = kernel.clone()
        inv_unselected = torch.linalg.inv(kernel)
        unselected = torch.arange(count, device=c2w.device, dtype=torch.long)
        selected: list[int] = []
        eps = max(float(self.config.jitter), 1e-12)

        for _ in range(min(int(budget), count)):
            numerator = conditional.diagonal().clamp_min(eps)
            denominator = inv_unselected.diagonal().clamp_min(eps).reciprocal()
            gain = 0.5 * torch.log(numerator / denominator.clamp_min(eps))
            local = int(torch.argmax(gain).item())
            selected.append(int(unselected[local].item()))
            if len(selected) == budget:
                break

            keep = torch.ones(unselected.numel(), dtype=torch.bool, device=c2w.device)
            keep[local] = False
            c_rr = conditional[keep][:, keep]
            c_rj = conditional[keep, local]
            c_jj = conditional[local, local].clamp_min(eps)
            conditional = c_rr - torch.outer(c_rj, c_rj) / c_jj
            conditional = 0.5 * (conditional + conditional.T)

            # Inverse of the principal submatrix after deleting local index.
            a = inv_unselected[local, local].clamp_min(eps)
            d = inv_unselected[keep][:, keep]
            c = inv_unselected[keep, local]
            inv_unselected = d - torch.outer(c, c) / a
            inv_unselected = 0.5 * (inv_unselected + inv_unselected.T)
            unselected = unselected[keep]

        return torch.tensor(selected, device=c2w.device, dtype=torch.long)
