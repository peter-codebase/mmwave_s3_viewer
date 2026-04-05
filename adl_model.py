"""
ADL Classifier — CNN + GRU
===========================
Architecture:
  1. A small 2D CNN backbone processes each radar frame independently
     (weights are shared across all 16 frames in a window).
  2. The resulting per-frame feature vectors form a time sequence that is
     fed into a GRU, which learns the temporal rhythm of each activity.
  3. A small MLP head maps the GRU's final hidden state to class scores.

Input shape:  (batch, 16 frames, 64 range bins, 64 Doppler bins, 4 channels)
              channels: rx0_mag, rx1_mag, rx2_mag, doppler_proj
Output shape: (batch, num_classes)
"""

import torch
import torch.nn as nn


class _CNNBackbone(nn.Module):
    """2D CNN that summarises one radar frame into a feature vector.

    Applied independently to every frame in the window (shared weights).
    Three conv-BN-ReLU-MaxPool blocks reduce (in_channels,64,64) → (64,8,8),
    then AdaptiveAvgPool2d(4) keeps a 4×4 spatial grid → 1024-d vector.

    in_channels: 3 for magnitude-only, 4 for mag+doppler_proj,
                 6 for mag+doppler_proj+azimuth+elevation (default).
    """

    def __init__(self, out_dim: int = 128, in_channels: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            # Block 1 — (in_channels, 64, 64) → (8, 32, 32)
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.15),
            nn.MaxPool2d(2),

            # Block 2 — (8, 32, 32) → (16, 16, 16)
            nn.Conv2d(8, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.15),
            nn.MaxPool2d(2),

            # Block 3 — (16, 16, 16) → (32, 8, 8)
            nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.15),
            nn.MaxPool2d(2),

            # Preserve 4×4 spatial grid → (32, 4, 4) = 512-d
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),

            # Project to out_dim
            nn.Linear(32 * 4 * 4, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ADLClassifier(nn.Module):
    """
    CNN + GRU classifier for ADL radar data.

    Args:
        num_classes:  number of activity classes
        cnn_out_dim:  CNN backbone output dimension (default 64)
        gru_hidden:   GRU hidden state size (default 64)
        gru_layers:   number of stacked GRU layers (default 2)
        dropout:      dropout rate applied inside GRU and classifier (default 0.4)
        in_channels:  number of input channels per frame — 3 for magnitude only,
                      4 for mag + doppler_proj,
                      6 for mag + doppler_proj + azimuth + elevation (default 6)
    """

    def __init__(
        self,
        num_classes:   int,
        cnn_out_dim:   int   = 64,
        gru_hidden:    int   = 64,
        gru_layers:    int   = 2,
        dropout:       float = 0.4,
        bidirectional: bool  = True,
        in_channels:   int   = 4,
    ):
        super().__init__()
        self.cnn           = _CNNBackbone(out_dim=cnn_out_dim, in_channels=in_channels)
        self.bidirectional = bidirectional

        # dropout only applies between GRU layers (not on single-layer GRU)
        self.gru = nn.GRU(
            input_size=cnn_out_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        # Bidirectional GRU concatenates forward + backward last hidden states,
        # so the classifier input is gru_hidden*2 when bidirectional=True.
        clf_in = gru_hidden * 2 if bidirectional else gru_hidden
        self.classifier = nn.Sequential(
            nn.Linear(clf_in, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, T, H, W, C)  — T=frames, H=range, W=Doppler, C=channels

        Returns:
            logits: (batch, num_classes)
        """
        B, T, H, W, C = x.shape

        # Reshape so the CNN sees (B*T, C, H, W) — apply CNN to all frames at once
        x = x.permute(0, 1, 4, 2, 3)        # (B, T, C, H, W)
        x = x.reshape(B * T, C, H, W)        # (B*T, C, H, W)

        features = self.cnn(x)               # (B*T, cnn_out_dim)
        features = features.view(B, T, -1)   # (B, T, cnn_out_dim)

        _, h_n = self.gru(features)
        # h_n: (num_layers * num_directions, B, gru_hidden)
        # For the last GRU layer: forward = h_n[-2], backward = h_n[-1]
        if self.bidirectional:
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # (B, gru_hidden*2)
        else:
            h_last = h_n[-1]                                 # (B, gru_hidden)

        return self.classifier(h_last)       # (B, num_classes)
