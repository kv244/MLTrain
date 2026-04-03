import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # bias=False: BatchNorm absorbs the bias, saving parameters
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class AlphaNet(nn.Module):
    def __init__(self, num_res_blocks=8, channels=128):
        super().__init__()
        self.start_block = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        self.res_blocks = nn.Sequential(*[ResBlock(channels) for _ in range(num_res_blocks)])

        # Policy head — returns raw logits (no softmax here).
        # Softmax is applied in MCTS; log_softmax is used in the training loss.
        # This avoids the numerical instability of log(softmax(x)).
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * 6 * 7, 7)
        )

        # Value head — outputs scalar in (-1, 1) via Tanh
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(6 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.start_block(x)
        x = self.res_blocks(x)
        return self.policy_head(x), self.value_head(x)
