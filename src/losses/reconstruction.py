import torch.nn.functional as F

def recon_loss(x, x_hat):
    return F.mse_loss(x_hat, x)
