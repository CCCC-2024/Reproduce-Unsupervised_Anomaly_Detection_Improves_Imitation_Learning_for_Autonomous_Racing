from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class CAE(nn.Module):
    def __init__(self, latent_dim: int = 256):
        super().__init__()
        # Encoder: Conv1 16, Conv2 32, Conv3 64, FC 4096->256
        self.enc_conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.enc_conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # 64->32->16->8

        self.fc_enc = nn.Linear(64 * 8 * 8, latent_dim)

        # Decoder: FC 256->4096, Deconv1 64, Deconv2 32, Deconv3 16, final Sigmoid to [0,1]
        self.fc_dec = nn.Linear(latent_dim, 64 * 8 * 8)

        self.deconv1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.out_conv = nn.Conv2d(16, 3, kernel_size=3, padding=1)

    def encode(self, x):
        x = F.relu(self.enc_conv1(x))
        x = self.pool(x)
        x = F.relu(self.enc_conv2(x))
        x = self.pool(x)
        x = F.relu(self.enc_conv3(x))
        x = self.pool(x)  # -> (B,64,8,8) if input is (B,3,64,64)

        x = x.flatten(1)  # (B,4096)
        h = F.relu(self.fc_enc(x))  # (B,256)
        return h

    def decode(self, h):
        x = F.relu(self.fc_dec(h))               # (B,4096)
        x = x.view(-1, 64, 8, 8)                 # (B,64,8,8)
        x = F.relu(self.deconv1(x))              # (B,64,16,16)
        x = F.relu(self.deconv2(x))              # (B,32,32,32)
        x = torch.sigmoid(self.deconv3(x))       # (B,16,64,64)  Table I says sigmoid at last deconv
        x = torch.sigmoid(self.out_conv(x))      # final normalize to [0,1]
        return x

    def forward(self, x):
        h = self.encode(x)
        x_hat = self.decode(h)
        return x_hat, h
