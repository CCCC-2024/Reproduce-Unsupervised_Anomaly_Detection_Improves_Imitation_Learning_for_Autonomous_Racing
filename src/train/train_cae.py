from __future__ import annotations
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.models.cae import CAE
from src.losses.reconstruction import recon_loss
from src.losses.latent_reference import latent_reference_loss

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def build_reference_latents(model: CAE, dataset, ref_indices, batch_size, device):
    loader = DataLoader(Subset(dataset, ref_indices), batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    H = []
    for x, _, _ in loader:
        x = x.to(device)
        h = model.encode(x)
        H.append(h.detach().cpu())
    return torch.cat(H, dim=0).to(device)  # (M,256)

def train_cae(cfg: dict):
    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")
    set_seed(int(cfg["train"]["seed"]))

    os.makedirs(cfg["output"]["out_dir"], exist_ok=True)

    # dataset & loader
    from src.datasets.torch_dataset import NpzMixDataset
    ds = NpzMixDataset(
        npz_path=cfg["data"]["npz_path"],
        img_size=int(cfg["data"]["img_size"]),
        only_clean=bool(cfg["data"]["only_clean"]),
        limit=cfg["data"]["limit"],
    )

    dl = DataLoader(
        ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=True,
        drop_last=True,
    )

    # model
    model = CAE(latent_dim=256).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg["train"]["lr"]))

    lam = float(cfg["train"]["lambda_refer"])
    refer_m = int(cfg["train"]["refer_m"])
    epochs = int(cfg["train"]["epochs"])
    save_every = int(cfg["train"]["save_every"])

    # fixed reference indices (Drefer)
    rng = np.random.default_rng(int(cfg["train"]["seed"]))
    M = min(refer_m, len(ds))
    ref_indices = rng.choice(len(ds), size=M, replace=False).tolist()

    for ep in range(1, epochs + 1):
        # build Href each epoch (paper: encode Drefer each epoch)
        Href = build_reference_latents(model, ds, ref_indices, batch_size=256, device=device)

        model.train()
        pbar = tqdm(dl, desc=f"epoch {ep}/{epochs}", ncols=100)
        running = {"loss": 0.0, "rec": 0.0, "ref": 0.0}
        steps = 0

        for x, _, _ in pbar:
            x = x.to(device, non_blocking=True)
            x_hat, h = model(x)

            l_rec = recon_loss(x, x_hat)
            l_ref = latent_reference_loss(h, Href) if lam > 0 else torch.zeros_like(l_rec)
            loss = l_rec + lam * l_ref

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            steps += 1
            running["loss"] += loss.item()
            running["rec"] += l_rec.item()
            running["ref"] += l_ref.item()

            pbar.set_postfix({
                "loss": f"{running['loss']/steps:.4f}",
                "rec": f"{running['rec']/steps:.4f}",
                "ref": f"{running['ref']/steps:.4f}",
            })

        if ep % save_every == 0 or ep == epochs:
            ckpt = {
                "epoch": ep,
                "cfg": cfg,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "ref_indices": ref_indices,
            }
            out = os.path.join(cfg["output"]["out_dir"], f"ckpt_ep{ep:03d}.pt")
            torch.save(ckpt, out)

    return cfg["output"]["out_dir"]
