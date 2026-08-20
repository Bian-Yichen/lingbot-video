from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .three_drae import WanThreeDRAEModel
from .wan_latent_training import PreparedWanLatentBatch


class ThreeDRAEObjective(nn.Module):
    """Wan-adapted 3DRAE reconstruction objective before adversarial loss."""

    def __init__(
        self,
        *,
        latent_weight: float,
        rgb_weight: float,
        lpips_weight: float,
    ) -> None:
        super().__init__()
        self.latent_weight = float(latent_weight)
        self.rgb_weight = float(rgb_weight)
        self.lpips_weight = float(lpips_weight)
        if min(self.latent_weight, self.rgb_weight, self.lpips_weight) < 0:
            raise ValueError("3DRAE reconstruction weights cannot be negative")
        if self.latent_weight == self.rgb_weight == self.lpips_weight == 0:
            raise ValueError("at least one 3DRAE reconstruction loss is required")
        self.perceptual: nn.Module | None = None
        if self.lpips_weight > 0:
            try:
                import lpips
            except ImportError as error:
                raise ImportError(
                    "LPIPS is enabled. Install with "
                    "`pip install -e '.[memory-reconstruction]'`."
                ) from error
            self.perceptual = lpips.LPIPS(net="vgg").eval().requires_grad_(False)

    def forward(
        self,
        predicted_latents: torch.Tensor,
        target_latents: torch.Tensor,
        predicted_rgb: torch.Tensor,
        target_rgb: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if predicted_latents.shape != target_latents.shape:
            raise ValueError(
                "predicted/target latent shape mismatch: "
                f"{predicted_latents.shape} vs {target_latents.shape}"
            )
        if predicted_rgb.shape != target_rgb.shape:
            raise ValueError(
                "predicted/target RGB shape mismatch: "
                f"{predicted_rgb.shape} vs {target_rgb.shape}"
            )
        latent_mse = F.mse_loss(
            predicted_latents.float(),
            target_latents.float(),
        )
        rgb_mse = F.mse_loss(predicted_rgb.float(), target_rgb.float())
        if self.perceptual is None:
            perceptual = rgb_mse.new_zeros(())
        else:
            batch, views, channels, height, width = predicted_rgb.shape
            prediction_2d = predicted_rgb.float().reshape(
                batch * views,
                channels,
                height,
                width,
            )
            target_2d = target_rgb.float().reshape_as(prediction_2d)
            perceptual = self.perceptual(
                prediction_2d.mul(2.0).sub(1.0),
                target_2d.mul(2.0).sub(1.0),
            ).mean()
        reconstruction_loss = (
            self.latent_weight * latent_mse
            + self.rgb_weight * rgb_mse
            + self.lpips_weight * perceptual
        )
        psnr = -10.0 * torch.log10(rgb_mse.detach().clamp_min(1e-12))
        return {
            "loss": reconstruction_loss,
            "reconstruction_loss": reconstruction_loss,
            "latent_mse": latent_mse.detach(),
            "rgb_mse": rgb_mse.detach(),
            "lpips": perceptual.detach(),
            "psnr": psnr,
        }


class DINOv2Discriminator(nn.Module):
    """Paper discriminator initialized from a pretrained DINOv2-Small."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        image_size: int = 224,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as error:
            raise ImportError(
                "DINOv2 adversarial training requires transformers."
            ) from error
        self.backbone = AutoModel.from_pretrained(model_name_or_path)
        if gradient_checkpointing and hasattr(
            self.backbone,
            "gradient_checkpointing_enable",
        ):
            self.backbone.gradient_checkpointing_enable()
        hidden_size = int(self.backbone.config.hidden_size)
        self.head = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.head.bias)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        self.image_size = int(image_size)
        self.register_buffer(
            "image_mean",
            torch.tensor((0.485, 0.456, 0.406)).reshape(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor((0.229, 0.224, 0.225)).reshape(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.ndim == 5:
            rgb = rgb.flatten(0, 1)
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("discriminator RGB must be [N,3,H,W]")
        resized = F.interpolate(
            rgb.float(),
            size=(self.image_size, self.image_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        normalized = (resized - self.image_mean) / self.image_std
        output = self.backbone(pixel_values=normalized)
        cls_token = output.last_hidden_state[:, 0]
        return self.head(cls_token).flatten()


def hinge_discriminator_loss(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
) -> torch.Tensor:
    return 0.5 * (
        F.relu(1.0 - real_logits).mean()
        + F.relu(1.0 + fake_logits).mean()
    )


def adaptive_adversarial_weight(
    reconstruction_loss: torch.Tensor,
    adversarial_loss: torch.Tensor,
    last_layer: torch.Tensor,
    *,
    maximum: float = 1.0e4,
) -> torch.Tensor:
    """VAE-style adaptive omega_G from last-layer gradient magnitudes."""

    reconstruction_gradient = torch.autograd.grad(
        reconstruction_loss,
        last_layer,
        retain_graph=True,
    )[0]
    adversarial_gradient = torch.autograd.grad(
        adversarial_loss,
        last_layer,
        retain_graph=True,
    )[0]
    weight = reconstruction_gradient.float().norm() / (
        adversarial_gradient.float().norm() + 1.0e-4
    )
    return weight.clamp(0.0, float(maximum)).detach()


@dataclass
class ThreeDRAEStepOutput:
    metrics: dict[str, torch.Tensor]
    predicted_rgb: torch.Tensor
    target_rgb: torch.Tensor


def three_drae_training_step(
    model: WanThreeDRAEModel,
    batch: PreparedWanLatentBatch,
    *,
    objective: ThreeDRAEObjective,
    device: torch.device,
    compute_dtype: torch.dtype,
) -> ThreeDRAEStepOutput:
    target_latents = batch.target_latents.to(
        device=device,
        dtype=compute_dtype,
    ).permute(0, 2, 1, 3, 4)
    target_rgb = batch.target_rgb.to(device=device, dtype=compute_dtype)
    predicted_latents, predicted_rgb, memory, visibility = model(
        batch.history_latents.to(device=device, dtype=compute_dtype),
        batch.history_c2w.to(device),
        batch.history_intrinsics.to(device),
        batch.target_c2w.to(device),
        batch.target_intrinsics.to(device),
    )
    metrics = objective(
        predicted_latents,
        target_latents,
        predicted_rgb,
        target_rgb,
    )
    metrics["memory_norm"] = memory.detach().float().norm(dim=-1).mean()
    metrics["latent_prediction_std"] = predicted_latents.detach().float().std()
    metrics["rgb_prediction_std"] = predicted_rgb.detach().float().std()
    metrics["visible_history_fraction"] = visibility.detach().float().mean()
    return ThreeDRAEStepOutput(
        metrics=metrics,
        predicted_rgb=predicted_rgb,
        target_rgb=target_rgb,
    )
