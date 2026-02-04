from __future__ import annotations
import torch
import torch.nn.functional as F

@torch.no_grad()
def nearest_latents(h: torch.Tensor, Href: torch.Tensor) -> torch.Tensor:
    """
    h: (B,D)
    Href: (M,D)
    return: h_near (B,D)
    """
    # dist^2 = ||h||^2 + ||Href||^2 - 2 h Href^T
    h2 = (h * h).sum(dim=1, keepdim=True)          # (B,1)
    H2 = (Href * Href).sum(dim=1).unsqueeze(0)     # (1,M)
    dist = h2 + H2 - 2.0 * (h @ Href.t())          # (B,M)
    idx = dist.argmin(dim=1)                       # (B,)
    return Href[idx]                               # (B,D)

def latent_reference_loss(h: torch.Tensor, Href: torch.Tensor) -> torch.Tensor:
    h_near = nearest_latents(h.detach(), Href)     # no grad through NN selection
    return F.mse_loss(h, h_near)
