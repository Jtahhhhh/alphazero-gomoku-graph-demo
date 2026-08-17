"""CNN baseline network for Gomoku, compatible with RGCN/RGAT interface."""

import torch
from torch import nn
from .common import PolicyValueHeads


class ResidualBlock(nn.Module):
    """Residual convolutional block with batch norm and ReLU."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + residual)


class CNNBaseline(nn.Module):
    """
    CNN baseline for Gomoku.
    
    Takes input (batch, 6, board_size, board_size) where 6 planes are:
    - Current player stones
    - Opponent stones
    - Last move indicator
    - Current player sign
    - Row position (normalized)
    - Column position (normalized)
    
    Returns (policy_logits, value_logits) or (policy, value, evidence) if return_evidence=True.
    """

    def __init__(self, board_size=15, hidden_dim=256, num_residual_blocks=4, **_):
        super().__init__()
        self.board_size = board_size

        # Initial convolution: 6 input planes -> hidden_dim channels
        self.input_conv = nn.Sequential(
            nn.Conv2d(6, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
        )

        # Residual blocks
        self.residual_blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim) for _ in range(num_residual_blocks)]
        )

        # Policy head: output logits for each position
        self.policy_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(),
        )

        # Value head: output single value for position
        self.value_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )

        self.value_fc = nn.Sequential(
            nn.Linear(board_size * board_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

    def forward(self, x, return_evidence=False):
        """
        Forward pass.
        
        Args:
            x: Tensor of shape (batch, 6, board_size, board_size)
            return_evidence: If True, return (policy, value, evidence_dict)
            
        Returns:
            If return_evidence=False: (policy_logits, value_logits) both shape (batch, board_size^2)
            If return_evidence=True: (policy, value, {"attention_available": False})
        """
        # Initial convolution
        h = self.input_conv(x)

        # Residual blocks
        for block in self.residual_blocks:
            h = block(h)

        # Policy head: (batch, 2, board_size, board_size) -> average across channels -> (batch, board_size^2)
        policy_map = self.policy_head(h)  # (batch, 2, board_size, board_size)
        policy_logits = policy_map.mean(dim=1).flatten(1)  # Average over 2 channels -> (batch, board_size^2)

        # Value head: (batch, 1, board_size, board_size) -> (batch,)
        value_map = self.value_head(h).flatten(1)  # (batch, board_size^2)
        value_logits = self.value_fc(value_map).squeeze(-1)  # (batch,)

        if return_evidence:
            return policy_logits, value_logits, {"attention_available": False}
        return policy_logits, value_logits
