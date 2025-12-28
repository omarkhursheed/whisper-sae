"""SAE model implementations.

This module provides SAE architectures, either wrapping SAELens or custom implementations.
We prefer SAELens for battle-tested training stability, but provide fallback for flexibility.
"""

from typing import Literal, NamedTuple

import torch
from torch import Tensor, nn

from ..config import SAEConfig


class SAEOutput(NamedTuple):
    """Output from SAE forward pass."""

    reconstructed: Tensor  # Reconstructed activations
    hidden: Tensor  # SAE latent activations (sparse)
    loss: Tensor  # Total loss
    reconstruction_loss: Tensor  # MSE reconstruction loss
    sparsity_loss: Tensor  # L0 or auxiliary sparsity loss
    l0: Tensor  # Number of active features per token


class TopKSAE(nn.Module):
    """TopK Sparse Autoencoder.

    Uses TopK activation instead of L1 regularization for more stable training.
    Mozilla Builders found TopK more stable for Whisper specifically.

    Architecture:
        - Encoder: Linear(input_dim, hidden_dim)
        - TopK activation: Keep only k largest activations
        - Decoder: Linear(hidden_dim, input_dim) with unit-norm columns
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        k: int = 32,
        normalize_decoder: bool = True,
        dead_feature_threshold: int = 10_000,
    ):
        """Initialize TopK SAE.

        Args:
            input_dim: Dimension of input activations.
            hidden_dim: Number of SAE features (latent dimension).
            k: Number of active features to keep per token.
            normalize_decoder: Whether to normalize decoder columns to unit norm.
            dead_feature_threshold: Steps without activation before feature is "dead".
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k = k
        self.normalize_decoder = normalize_decoder
        self.dead_feature_threshold = dead_feature_threshold

        # Encoder and decoder
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)

        # Pre-encoder bias (for centering)
        self.b_pre = nn.Parameter(torch.zeros(input_dim))

        # Initialize decoder with unit-norm columns
        self._init_decoder()

        # Track dead features
        self.register_buffer(
            "feature_last_activated",
            torch.zeros(hidden_dim, dtype=torch.long),
        )
        self.register_buffer("step_count", torch.tensor(0, dtype=torch.long))

    def _init_decoder(self) -> None:
        """Initialize decoder with unit-norm columns."""
        with torch.no_grad():
            # Initialize with small random weights
            nn.init.xavier_uniform_(self.decoder.weight)
            # Normalize columns to unit norm
            self.decoder.weight.data = nn.functional.normalize(
                self.decoder.weight.data, dim=0
            )
            # Scale down initially for stability
            self.decoder.weight.data *= 0.1

    def normalize_decoder_weights(self) -> None:
        """Normalize decoder columns to unit norm (call after optimizer step)."""
        with torch.no_grad():
            self.decoder.weight.data = nn.functional.normalize(
                self.decoder.weight.data, dim=0
            )

    def encode(self, x: Tensor) -> Tensor:
        """Encode input to latent space with TopK activation.

        Args:
            x: Input activations [batch, input_dim].

        Returns:
            Sparse latent activations [batch, hidden_dim].
        """
        # Center the input
        x_centered = x - self.b_pre

        # Encode
        pre_activation = self.encoder(x_centered)

        # TopK activation
        topk_values, topk_indices = torch.topk(pre_activation, self.k, dim=-1)
        hidden = torch.zeros_like(pre_activation)
        hidden.scatter_(-1, topk_indices, torch.relu(topk_values))

        return hidden

    def decode(self, hidden: Tensor) -> Tensor:
        """Decode latent to reconstruction.

        Args:
            hidden: Latent activations [batch, hidden_dim].

        Returns:
            Reconstructed activations [batch, input_dim].
        """
        return self.decoder(hidden) + self.b_pre

    def forward(self, x: Tensor) -> SAEOutput:
        """Forward pass with loss computation.

        Args:
            x: Input activations [batch, input_dim].

        Returns:
            SAEOutput with reconstruction, hidden, and losses.
        """
        # Encode and decode
        hidden = self.encode(x)
        reconstructed = self.decode(hidden)

        # Compute losses
        reconstruction_loss = nn.functional.mse_loss(reconstructed, x)

        # L0 sparsity (number of active features)
        l0 = (hidden > 0).float().sum(dim=-1).mean()

        # For TopK, sparsity loss is 0 (sparsity is enforced by design)
        sparsity_loss = torch.tensor(0.0, device=x.device)

        # Total loss is just reconstruction for TopK
        loss = reconstruction_loss

        # Update dead feature tracking
        self._update_dead_features(hidden)

        return SAEOutput(
            reconstructed=reconstructed,
            hidden=hidden,
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            sparsity_loss=sparsity_loss,
            l0=l0,
        )

    def _update_dead_features(self, hidden: Tensor) -> None:
        """Update tracking of dead features.

        Args:
            hidden: Latent activations from current batch.
        """
        if self.training:
            self.step_count += 1

            # Find which features were active
            active_features = (hidden > 0).any(dim=0)

            # Reset counter for active features
            self.feature_last_activated[active_features] = self.step_count

    def get_dead_features(self) -> Tensor:
        """Get mask of dead features.

        Returns:
            Boolean tensor of shape [hidden_dim], True for dead features.
        """
        steps_since_active = self.step_count - self.feature_last_activated
        return steps_since_active > self.dead_feature_threshold

    def get_dead_feature_ratio(self) -> float:
        """Get ratio of dead features."""
        dead = self.get_dead_features()
        return dead.float().mean().item()

    def resample_dead_features(
        self,
        inputs: Tensor,
        num_resample: int | None = None,
    ) -> int:
        """Resample dead features using high-residual examples.

        Dead features are reinitialized to point toward input examples that
        have high reconstruction error, giving them a chance to learn useful
        patterns that the current features are missing.

        Args:
            inputs: Batch of input activations [batch, input_dim].
            num_resample: Maximum features to resample. If None, resample all dead.

        Returns:
            Number of features resampled.
        """
        dead_mask = self.get_dead_features()
        dead_indices = torch.where(dead_mask)[0]
        num_dead = len(dead_indices)

        if num_dead == 0:
            return 0

        # Limit number of features to resample
        if num_resample is not None:
            num_dead = min(num_dead, num_resample)
            dead_indices = dead_indices[:num_dead]

        # Compute reconstruction errors
        with torch.no_grad():
            output = self.forward(inputs)
            residuals = inputs - output.reconstructed
            errors = (residuals ** 2).sum(dim=-1)  # [batch]

            # Select high-error examples
            _, top_indices = torch.topk(errors, min(num_dead, len(errors)))

            # Get the high-error input examples
            high_error_inputs = inputs[top_indices]  # [num_dead, input_dim]

            # Normalize to unit norm for encoder weights
            high_error_normalized = nn.functional.normalize(high_error_inputs, dim=-1)

            # Reinitialize dead features
            for i, dead_idx in enumerate(dead_indices):
                if i >= len(high_error_normalized):
                    break

                # Set encoder row to point toward high-error example
                self.encoder.weight.data[dead_idx] = high_error_normalized[i]
                self.encoder.bias.data[dead_idx] = 0.0

                # Set decoder column to same direction (transposed)
                self.decoder.weight.data[:, dead_idx] = high_error_normalized[i]

                # Reset dead feature counter
                self.feature_last_activated[dead_idx] = self.step_count

        return num_dead


class ReLUSAE(nn.Module):
    """ReLU SAE with L1 sparsity regularization.

    Traditional SAE architecture for comparison and fallback.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        sparsity_weight: float = 0.01,
        normalize_decoder: bool = True,
    ):
        """Initialize ReLU SAE.

        Args:
            input_dim: Dimension of input activations.
            hidden_dim: Number of SAE features.
            sparsity_weight: Weight for L1 sparsity penalty.
            normalize_decoder: Whether to normalize decoder columns.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_weight = sparsity_weight
        self.normalize_decoder = normalize_decoder

        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)

        if normalize_decoder:
            with torch.no_grad():
                self.decoder.weight.data = nn.functional.normalize(
                    self.decoder.weight.data, dim=0
                )

    def normalize_decoder_weights(self) -> None:
        """Normalize decoder columns to unit norm."""
        if self.normalize_decoder:
            with torch.no_grad():
                self.decoder.weight.data = nn.functional.normalize(
                    self.decoder.weight.data, dim=0
                )

    def forward(self, x: Tensor) -> SAEOutput:
        """Forward pass."""
        hidden = torch.relu(self.encoder(x))
        reconstructed = self.decoder(hidden)

        reconstruction_loss = nn.functional.mse_loss(reconstructed, x)
        sparsity_loss = hidden.abs().mean()
        loss = reconstruction_loss + self.sparsity_weight * sparsity_loss

        l0 = (hidden > 0).float().sum(dim=-1).mean()

        return SAEOutput(
            reconstructed=reconstructed,
            hidden=hidden,
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            sparsity_loss=sparsity_loss,
            l0=l0,
        )


def create_sae(
    config: SAEConfig,
    input_dim: int,
) -> nn.Module:
    """Create an SAE from configuration.

    Args:
        config: SAE configuration.
        input_dim: Dimension of input activations.

    Returns:
        Initialized SAE module.
    """
    hidden_dim = config.get_hidden_dim(input_dim)

    if config.activation == "topk":
        return TopKSAE(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            k=config.k,
            normalize_decoder=config.normalize_decoder,
            dead_feature_threshold=config.dead_feature_threshold,
        )
    else:
        # Fallback to ReLU SAE
        return ReLUSAE(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            normalize_decoder=config.normalize_decoder,
        )
