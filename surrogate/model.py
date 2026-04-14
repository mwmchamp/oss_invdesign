"""CNN surrogate models: pixel grid → S-parameters.

Four architectures (v1, v2, v3, v4) with increasing sophistication.

v1 (SParamCNN): Simple 2-layer CNN → MLP → 1D conv decoder. Baseline.
v2 (SParamCNNv2): Multi-scale skip connections, wider, deeper decoder.
v3 (SParamResNet): ResNet encoder + SE attention + multi-scale fusion.
v4 (SParamEM300): EM300CNN-inspired 12-layer deep conv, no pooling, 5 FC.
    Based on JSSC 2023 / ISSCC 2025 reference architecture.

Output shape: (batch, n_freq, 10, 2) = n_freq × 10 upper-tri × (real, imag)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SParamCNN(nn.Module):
    """CNN encoder → flatten → MLP → frequency 1D conv decoder.

    Args:
        n_freq: number of frequency points (default 30)
        n_utri: number of upper-triangle S-param elements (default 10)
        base_channels: CNN feature channels (default 64)
        clamp_passivity: enforce |S| ≤ 1 at inference (default True)
    """

    def __init__(
        self,
        n_freq: int = 30,
        n_utri: int = 10,
        base_channels: int = 64,
        clamp_passivity: bool = True,
    ):
        super().__init__()
        self.n_freq = n_freq
        self.n_utri = n_utri
        self.clamp_passivity = clamp_passivity
        c = base_channels

        # CNN encoder: 27→9→3, preserves spatial structure
        self.conv = nn.Sequential(
            nn.Conv2d(1, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=3),  # 27→9

            nn.Conv2d(c, 2*c, 3, padding=1), nn.BatchNorm2d(2*c), nn.ReLU(inplace=True),
            nn.Conv2d(2*c, 2*c, 3, padding=1), nn.BatchNorm2d(2*c), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=3),  # 9→3
        )
        flat_dim = 2 * c * 3 * 3  # 2c × 3 × 3

        # MLP encoder: flatten spatial features → latent vector
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
        )

        # Frequency decoder: project to per-freq, smooth with 1D conv
        freq_hidden = 128
        self.freq_proj = nn.Linear(512, n_freq * freq_hidden)
        self.freq_hidden = freq_hidden

        self.freq_smooth = nn.Sequential(
            nn.Conv1d(freq_hidden, freq_hidden, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(freq_hidden, freq_hidden, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.out_proj = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(freq_hidden, n_utri * 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, 27, 27) binary pixel grid
        Returns:
            (batch, n_freq, n_utri, 2) S-parameter predictions (real, imag)
        """
        B = x.shape[0]

        # CNN + MLP encode
        h = self.encoder(self.conv(x))  # (B, 512)

        # Frequency-aware decode
        f = self.freq_proj(h).view(B, self.n_freq, self.freq_hidden)  # (B, F, H)
        f = f.permute(0, 2, 1)  # (B, H, F) for Conv1d
        f = f + self.freq_smooth(f)  # residual connection for smoothness
        f = f.permute(0, 2, 1)  # (B, F, H)

        out = self.out_proj(f)  # (B, F, 20)
        out = out.view(B, self.n_freq, self.n_utri, 2)

        # Passivity clamp at inference: scale so |S_ij| ≤ 1
        if self.clamp_passivity and not self.training:
            mag = torch.sqrt(out[..., 0] ** 2 + out[..., 1] ** 2 + 1e-12)
            scale = torch.clamp(1.0 / mag, max=1.0).unsqueeze(-1)
            out = out * scale

        return out


class SParamCNNv2(nn.Module):
    """Upgraded CNN with multi-scale skip connections and passivity clamp.

    Improvements over v1:
      - Multi-scale: 9×9 features (via adaptive pool) concatenated with 3×3 flatten
      - Wider default (base_channels=128, ~16M params vs 4M)
      - Deeper frequency decoder (3 conv layers)
      - Passivity clamp: output |S_ij| ≤ 1 (hard physics constraint)

    Args:
        n_freq: number of frequency points (default 30)
        n_utri: number of upper-triangle S-param elements (default 10)
        base_channels: CNN feature channels (default 128)
        clamp_passivity: enforce |S| ≤ 1 in output (default True)
    """

    def __init__(
        self,
        n_freq: int = 30,
        n_utri: int = 10,
        base_channels: int = 128,
        clamp_passivity: bool = True,
    ):
        super().__init__()
        self.n_freq = n_freq
        self.n_utri = n_utri
        self.clamp_passivity = clamp_passivity
        c = base_channels

        # Stage 1: 27→9
        self.stage1 = nn.Sequential(
            nn.Conv2d(1, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=3),
        )

        # Stage 2: 9→3
        self.stage2 = nn.Sequential(
            nn.Conv2d(c, 2 * c, 3, padding=1), nn.BatchNorm2d(2 * c), nn.ReLU(inplace=True),
            nn.Conv2d(2 * c, 2 * c, 3, padding=1), nn.BatchNorm2d(2 * c), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=3),
        )

        # Skip: pool 9×9 features down to 3×3 and project to same channel count
        self.skip_pool = nn.AdaptiveAvgPool2d(3)
        self.skip_proj = nn.Sequential(
            nn.Conv2d(c, 2 * c, 1), nn.BatchNorm2d(2 * c), nn.ReLU(inplace=True),
        )

        # Combined: 2*c (stage2) + 2*c (skip) = 4*c at 3×3
        combined_dim = 4 * c * 3 * 3

        # MLP encoder
        latent = 768
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(combined_dim, 1536),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(1536, latent),
            nn.ReLU(inplace=True),
        )

        # Frequency decoder
        freq_hidden = 192
        self.freq_proj = nn.Linear(latent, n_freq * freq_hidden)
        self.freq_hidden = freq_hidden

        self.freq_smooth = nn.Sequential(
            nn.Conv1d(freq_hidden, freq_hidden, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(freq_hidden, freq_hidden, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(freq_hidden, freq_hidden, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.out_proj = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(freq_hidden, n_utri * 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # Multi-scale CNN encode
        s1 = self.stage1(x)           # (B, c, 9, 9)
        s2 = self.stage2(s1)          # (B, 2c, 3, 3)
        skip = self.skip_proj(self.skip_pool(s1))  # (B, 2c, 3, 3)
        combined = torch.cat([s2, skip], dim=1)     # (B, 4c, 3, 3)

        h = self.encoder(combined)    # (B, latent)

        # Frequency decode
        f = self.freq_proj(h).view(B, self.n_freq, self.freq_hidden)
        f = f.permute(0, 2, 1)
        f = f + self.freq_smooth(f)   # residual
        f = f.permute(0, 2, 1)

        out = self.out_proj(f).view(B, self.n_freq, self.n_utri, 2)

        # Passivity clamp: scale (real, imag) so |S_ij| ≤ 1
        # Only at inference — during training, let gradients flow freely
        if self.clamp_passivity and not self.training:
            mag = torch.sqrt(out[..., 0] ** 2 + out[..., 1] ** 2 + 1e-12)
            scale = torch.clamp(1.0 / mag, max=1.0).unsqueeze(-1)
            out = out * scale

        return out


# ── v3: ResNet encoder + frequency-conditioned cross-attention ──────────


class ResBlock(nn.Module):
    """Pre-activation residual block (He et al. 2016)."""
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class _SE(nn.Module):
    """Squeeze-and-Excitation channel attention (Hu et al. 2018).

    Global-avg-pools spatial dims, learns per-channel importance weights.
    Cheap (~0.1% params) but lets the model suppress noisy channels.
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


class SParamResNet(nn.Module):
    """v3: Deep ResNet encoder + multi-scale fusion + proven freq decoder.

    Improvements over v1/v2:
    1. 8 residual blocks (vs 4 plain convs) — captures long-range pixel
       interactions without gradient degradation
    2. Multi-scale fusion: 9×9 features (via adaptive pool to 3×3) concatenated
       with 3×3 features, giving the MLP access to both fine and coarse structure
    3. Wider frequency decoder with 3 conv layers for spectral smoothness
    4. SE (squeeze-and-excitation) channel attention after each stage —
       learns which feature channels matter without the overhead of
       spatial cross-attention

    Uses the proven MLP→freq_proj→Conv1d decoder from v1 (which works better
    than cross-attention at dataset sizes under ~10K).

    Args:
        n_freq: number of frequency points (default 30)
        n_utri: number of upper-triangle S-param elements (default 10)
        base_channels: ResNet feature width (default 64)
        clamp_passivity: enforce |S| ≤ 1 at inference
    """

    def __init__(
        self,
        n_freq: int = 30,
        n_utri: int = 10,
        base_channels: int = 64,
        clamp_passivity: bool = True,
        # Accept and ignore cross-attention args for backward compat
        **kwargs,
    ):
        super().__init__()
        self.n_freq = n_freq
        self.n_utri = n_utri
        self.clamp_passivity = clamp_passivity
        c = base_channels

        # ── ResNet Encoder ──────────────────────────────────────────
        # Stage 1: 1→c, 27×27
        self.stem = nn.Sequential(
            nn.Conv2d(1, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.res1a = ResBlock(c)
        self.res1b = ResBlock(c)
        # Downsample 27→9
        self.down1 = nn.Sequential(
            nn.Conv2d(c, 2 * c, 3, stride=3, padding=0, bias=False),
            nn.BatchNorm2d(2 * c),
            nn.ReLU(inplace=True),
        )
        self.se1 = _SE(2 * c)

        # Stage 2: 2c, 9×9
        self.res2a = ResBlock(2 * c)
        self.res2b = ResBlock(2 * c)
        # Downsample 9→3
        self.down2 = nn.Sequential(
            nn.Conv2d(2 * c, 4 * c, 3, stride=3, padding=0, bias=False),
            nn.BatchNorm2d(4 * c),
            nn.ReLU(inplace=True),
        )
        self.se2 = _SE(4 * c)

        # Stage 3: 4c, 3×3
        self.res3a = ResBlock(4 * c)
        self.res3b = ResBlock(4 * c)

        # ── Multi-scale fusion ──────────────────────────────────────
        # Pool 9×9 features to 3×3 and project to same channel count
        self.skip_pool = nn.AdaptiveAvgPool2d(3)
        self.skip_proj = nn.Sequential(
            nn.Conv2d(2 * c, 4 * c, 1, bias=False),
            nn.BatchNorm2d(4 * c),
            nn.ReLU(inplace=True),
        )
        # stage3 (4c×3×3) + skip (4c×3×3) = 8c×3×3
        fused_dim = 8 * c * 3 * 3

        # ── MLP encoder ─────────────────────────────────────────────
        latent = 512
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fused_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(1024, latent),
            nn.ReLU(inplace=True),
        )

        # ── Frequency decoder ───────────────────────────────────────
        freq_hidden = 128
        self.freq_proj = nn.Linear(latent, n_freq * freq_hidden)
        self.freq_hidden = freq_hidden

        self.freq_smooth = nn.Sequential(
            nn.Conv1d(freq_hidden, freq_hidden, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(freq_hidden, freq_hidden, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(freq_hidden, freq_hidden, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.out_proj = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(freq_hidden, n_utri * 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]

        # Encode with residual blocks + SE attention
        h = self.stem(x)                    # (B, c, 27, 27)
        h = self.res1b(self.res1a(h))       # (B, c, 27, 27)
        h9 = self.se1(self.down1(h))        # (B, 2c, 9, 9)
        h = self.res2b(self.res2a(h9))      # (B, 2c, 9, 9)
        h3 = self.se2(self.down2(h))        # (B, 4c, 3, 3)
        h3 = self.res3b(self.res3a(h3))     # (B, 4c, 3, 3)

        # Multi-scale fusion
        skip = self.skip_proj(self.skip_pool(h9))  # (B, 4c, 3, 3)
        fused = torch.cat([h3, skip], dim=1)       # (B, 8c, 3, 3)

        # MLP
        latent = self.encoder(fused)               # (B, 768)

        # Frequency decode
        f = self.freq_proj(latent).view(B, self.n_freq, self.freq_hidden)
        f = f.permute(0, 2, 1)              # (B, H, F)
        f = f + self.freq_smooth(f)          # residual
        f = f.permute(0, 2, 1)              # (B, F, H)

        out = self.out_proj(f).view(B, self.n_freq, self.n_utri, 2)

        if self.clamp_passivity and not self.training:
            mag = torch.sqrt(out[..., 0] ** 2 + out[..., 1] ** 2 + 1e-12)
            scale = torch.clamp(1.0 / mag, max=1.0).unsqueeze(-1)
            out = out * scale

        return out


# ── v4: EM300-inspired deep conv + FC (no pooling in conv stack) ─────────


class SParamEM300(nn.Module):
    """v4: EM300CNN-inspired architecture from JSSC 2023 / ISSCC 2025.

    12 convolutional layers with decreasing kernel sizes (12→3), all 64 filters,
    BatchNorm + LeakyReLU(0.01), no pooling within conv stack.
    AdaptiveAvgPool2d reduces spatial dims before 5 FC layers.
    Tanh output scaled to match normalized S-param range.

    Only predicts upper-triangle (10 elements for 4-port), reciprocity
    enforced externally (S_ij = S_ji). SVD passivity enforcement at inference.

    Architecture adapted for 27×27 input (vs original 24×24) by using
    AdaptiveAvgPool2d to standardize the spatial size before FC layers.

    Args:
        n_freq: number of frequency points (default 30)
        n_utri: number of upper-triangle S-param elements (default 10)
        base_channels: conv filter count (default 64)
        clamp_passivity: enforce |S| ≤ 1 at inference (default True)
    """

    # Kernel sizes from the reference: 12 layers, decreasing
    KERNEL_SIZES = [12, 10, 8, 6, 5, 5, 4, 4, 4, 3, 3, 3]

    def __init__(
        self,
        n_freq: int = 30,
        n_utri: int = 10,
        base_channels: int = 64,
        clamp_passivity: bool = True,
        dropout: float = 0.5,
        output_activation: str = "none",
        **kwargs,
    ):
        super().__init__()
        self.n_freq = n_freq
        self.n_utri = n_utri
        self.clamp_passivity = clamp_passivity
        self.output_activation = output_activation
        c = base_channels

        # 12 conv layers: decreasing kernels, same-padding, no pooling
        layers = []
        in_ch = 1
        for k in self.KERNEL_SIZES:
            pad = k // 2  # approximate same-padding
            layers.extend([
                nn.Conv2d(in_ch, c, k, padding=pad, bias=False),
                nn.BatchNorm2d(c),
                nn.LeakyReLU(0.01, inplace=True),
            ])
            in_ch = c
        self.conv_stack = nn.Sequential(*layers)

        # Reduce spatial dims: 27×27 (after convs) → 6×6
        self.pool = nn.AdaptiveAvgPool2d(6)
        fc_in = c * 6 * 6  # 64 * 36 = 2304

        # 5 FC layers with BatchNorm + LeakyReLU + Dropout
        fc_dims = [fc_in, 512, 512, 512, 512]
        fc_layers = []
        for i in range(len(fc_dims) - 1):
            fc_layers.extend([
                nn.Linear(fc_dims[i], fc_dims[i + 1]),
                nn.BatchNorm1d(fc_dims[i + 1]),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Dropout(dropout),
            ])
        self.fc = nn.Sequential(*fc_layers)

        # Output: n_freq * n_utri * 2 (real, imag).
        # Karahan et al. JSSC 2023 trains in-air with no output activation
        # (avoids vanishing gradients) and only adds Tanh after transfer to
        # dielectric. We default to "none"; flip to "tanh" for fine-tune.
        self.out_dim = n_freq * n_utri * 2
        self._head_in = fc_dims[-1]
        head_layers: list[nn.Module] = [nn.Linear(fc_dims[-1], self.out_dim)]
        if output_activation == "tanh":
            head_layers.append(nn.Tanh())
        elif output_activation not in ("none", None):
            raise ValueError(f"Unknown output_activation: {output_activation}")
        self.head = nn.Sequential(*head_layers)

    def set_output_activation(self, activation: str) -> None:
        """Swap the output activation in-place, preserving Linear head weights.

        Used to implement the Karahan JSSC 2023 delayed-Tanh transfer protocol:
        Stage 1 (pretrain) runs with 'none' to avoid saturating gradients while
        the head is still learning gross magnitude structure; Stage 2 (fine-tune)
        switches to 'tanh' to softly bound the outputs to the normalised range.
        """
        if activation not in ("none", "tanh"):
            raise ValueError(f"Unknown output_activation: {activation}")
        linear = self.head[0]
        assert isinstance(linear, nn.Linear)
        new_layers: list[nn.Module] = [linear]
        if activation == "tanh":
            new_layers.append(nn.Tanh())
        self.head = nn.Sequential(*new_layers).to(linear.weight.device)
        self.output_activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, 27, 27) binary pixel grid
        Returns:
            (batch, n_freq, n_utri, 2) S-parameter predictions (real, imag)
        """
        B = x.shape[0]

        h = self.conv_stack(x)       # (B, 64, ~27, ~27)
        h = self.pool(h)             # (B, 64, 6, 6)
        h = h.view(B, -1)           # (B, 2304)
        h = self.fc(h)              # (B, 512)
        out = self.head(h)          # (B, n_freq*n_utri*2)

        out = out.view(B, self.n_freq, self.n_utri, 2)

        # Passivity clamp at inference
        if self.clamp_passivity and not self.training:
            mag = torch.sqrt(out[..., 0] ** 2 + out[..., 1] ** 2 + 1e-12)
            scale = torch.clamp(1.0 / mag, max=1.0).unsqueeze(-1)
            out = out * scale

        return out
