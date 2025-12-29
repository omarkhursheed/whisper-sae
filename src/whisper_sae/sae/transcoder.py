"""Transcoder model implementations.

Transcoders learn to predict MLP output from MLP input with a sparse bottleneck.
Unlike SAEs which reconstruct the same activation, transcoders model the
transformation performed by MLPs.

Reference: Paulo et al. (2025) - "Transcoders Are More Interpretable Than SAEs"
https://arxiv.org/abs/2502.02070

Key differences from SAEs:
- SAE: activation -> sparse -> reconstruction of same activation
- Transcoder: MLP_input -> sparse -> MLP_output (different input/output)
"""

from typing import NamedTuple

import torch
from torch import Tensor, nn


class TranscoderOutput(NamedTuple):
    """Output from Transcoder forward pass."""

    predicted: Tensor  # Predicted MLP output
    hidden: Tensor  # Sparse latent activations
    loss: Tensor  # Total loss
    reconstruction_loss: Tensor  # MSE prediction loss
    sparsity_loss: Tensor  # L0 or auxiliary sparsity loss
    l0: Tensor  # Number of active features per token


class TopKTranscoder(nn.Module):
    """TopK Transcoder for predicting MLP output from MLP input.

    Transcoders model the MLP transformation:
        MLP_output = Transcoder(MLP_input)

    With a sparse bottleneck, this decomposes the MLP into interpretable features.

    Architecture:
        - Encoder: Linear(input_dim, hidden_dim)
        - TopK activation: Keep only k largest activations
        - Decoder: Linear(hidden_dim, output_dim) with unit-norm columns

    For Whisper MLP blocks:
        - input_dim = d_model (384 for tiny)
        - output_dim = d_model (384 for tiny)
        - The MLP expands to 4*d_model internally, but we model input->output
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        k: int = 32,
        normalize_decoder: bool = True,
        dead_feature_threshold: int = 10_000,
    ):
        """Initialize TopK Transcoder.

        Args:
            input_dim: Dimension of MLP input (d_model).
            output_dim: Dimension of MLP output (d_model, same as input for residual).
            hidden_dim: Number of sparse features (latent dimension).
            k: Number of active features to keep per token.
            normalize_decoder: Whether to normalize decoder columns to unit norm.
            dead_feature_threshold: Steps without activation before feature is "dead".
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.k = k
        self.normalize_decoder = normalize_decoder
        self.dead_feature_threshold = dead_feature_threshold

        # Encoder: MLP input -> latent
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)

        # Decoder: latent -> MLP output
        self.decoder = nn.Linear(hidden_dim, output_dim, bias=True)

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
            nn.init.xavier_uniform_(self.decoder.weight)
            self.decoder.weight.data = nn.functional.normalize(
                self.decoder.weight.data, dim=0
            )
            self.decoder.weight.data *= 0.1

    def normalize_decoder_weights(self) -> None:
        """Normalize decoder columns to unit norm (call after optimizer step)."""
        with torch.no_grad():
            self.decoder.weight.data = nn.functional.normalize(
                self.decoder.weight.data, dim=0
            )

    def encode(self, x: Tensor) -> Tensor:
        """Encode MLP input to latent space with TopK activation.

        Args:
            x: MLP input activations [batch, input_dim].

        Returns:
            Sparse latent activations [batch, hidden_dim].
        """
        pre_activation = self.encoder(x)

        # TopK activation
        topk_values, topk_indices = torch.topk(pre_activation, self.k, dim=-1)
        hidden = torch.zeros_like(pre_activation)
        hidden.scatter_(-1, topk_indices, torch.relu(topk_values))

        return hidden

    def decode(self, hidden: Tensor) -> Tensor:
        """Decode latent to predicted MLP output.

        Args:
            hidden: Latent activations [batch, hidden_dim].

        Returns:
            Predicted MLP output [batch, output_dim].
        """
        return self.decoder(hidden)

    def forward(self, mlp_input: Tensor, mlp_output: Tensor) -> TranscoderOutput:
        """Forward pass with loss computation.

        Args:
            mlp_input: MLP input activations [batch, input_dim].
            mlp_output: MLP output activations [batch, output_dim] (target).

        Returns:
            TranscoderOutput with prediction, hidden, and losses.
        """
        # Encode and decode
        hidden = self.encode(mlp_input)
        predicted = self.decode(hidden)

        # Compute losses
        reconstruction_loss = nn.functional.mse_loss(predicted, mlp_output)

        # L0 sparsity (number of active features)
        l0 = (hidden > 0).float().sum(dim=-1).mean()

        # For TopK, sparsity loss is 0
        sparsity_loss = torch.tensor(0.0, device=mlp_input.device)

        # Total loss is just reconstruction for TopK
        loss = reconstruction_loss

        # Update dead feature tracking
        self._update_dead_features(hidden)

        return TranscoderOutput(
            predicted=predicted,
            hidden=hidden,
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            sparsity_loss=sparsity_loss,
            l0=l0,
        )

    def _update_dead_features(self, hidden: Tensor) -> None:
        """Update tracking of dead features."""
        if self.training:
            self.step_count += 1
            active_features = (hidden > 0).any(dim=0)
            self.feature_last_activated[active_features] = self.step_count

    def get_dead_features(self) -> Tensor:
        """Get mask of dead features."""
        steps_since_active = self.step_count - self.feature_last_activated
        return steps_since_active > self.dead_feature_threshold

    def get_dead_feature_ratio(self) -> float:
        """Get ratio of dead features."""
        dead = self.get_dead_features()
        return dead.float().mean().item()

    def resample_dead_features(
        self,
        mlp_inputs: Tensor,
        mlp_outputs: Tensor,
        num_resample: int | None = None,
    ) -> int:
        """Resample dead features using high-residual examples.

        Args:
            mlp_inputs: Batch of MLP inputs [batch, input_dim].
            mlp_outputs: Batch of MLP outputs [batch, output_dim].
            num_resample: Maximum features to resample.

        Returns:
            Number of features resampled.
        """
        dead_mask = self.get_dead_features()
        dead_indices = torch.where(dead_mask)[0]
        num_dead = len(dead_indices)

        if num_dead == 0:
            return 0

        if num_resample is not None:
            num_dead = min(num_dead, num_resample)
            dead_indices = dead_indices[:num_dead]

        with torch.no_grad():
            output = self.forward(mlp_inputs, mlp_outputs)
            residuals = mlp_outputs - output.predicted
            errors = (residuals**2).sum(dim=-1)

            _, top_indices = torch.topk(errors, min(num_dead, len(errors)))
            high_error_inputs = mlp_inputs[top_indices]
            high_error_normalized = nn.functional.normalize(high_error_inputs, dim=-1)

            for i, dead_idx in enumerate(dead_indices):
                if i >= len(high_error_normalized):
                    break

                self.encoder.weight.data[dead_idx] = high_error_normalized[i]
                self.encoder.bias.data[dead_idx] = 0.0
                self.decoder.weight.data[:, dead_idx] = nn.functional.normalize(
                    residuals[top_indices[i]], dim=-1
                )
                self.feature_last_activated[dead_idx] = self.step_count

        return num_dead


class SkipTranscoder(nn.Module):
    """Transcoder with affine skip connection.

    Adds a linear skip connection from input to output, allowing the sparse
    component to model only the non-linear parts of the MLP transformation.

    Architecture:
        predicted = decoder(sparse_hidden) + skip(input)

    This typically achieves lower reconstruction loss than standard transcoders
    while maintaining interpretability of the sparse features.

    Reference: Paulo et al. (2025) found skip connections improve reconstruction
    without hurting interpretability.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        k: int = 32,
        normalize_decoder: bool = True,
        dead_feature_threshold: int = 10_000,
    ):
        """Initialize SkipTranscoder.

        Args:
            input_dim: Dimension of MLP input.
            output_dim: Dimension of MLP output.
            hidden_dim: Number of sparse features.
            k: Number of active features per token.
            normalize_decoder: Whether to normalize decoder columns.
            dead_feature_threshold: Steps until feature is considered dead.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.k = k
        self.normalize_decoder = normalize_decoder
        self.dead_feature_threshold = dead_feature_threshold

        # Sparse pathway
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
        self.decoder = nn.Linear(hidden_dim, output_dim, bias=True)

        # Skip connection (affine transformation)
        self.skip = nn.Linear(input_dim, output_dim, bias=True)

        # Initialize
        self._init_weights()

        # Dead feature tracking
        self.register_buffer(
            "feature_last_activated",
            torch.zeros(hidden_dim, dtype=torch.long),
        )
        self.register_buffer("step_count", torch.tensor(0, dtype=torch.long))

    def _init_weights(self) -> None:
        """Initialize weights following Paulo et al. (2025).

        Key insight: W₂ (decoder) and W_skip start at zero, b₂ (decoder bias)
        initializes to the empirical mean of MLP outputs (zero here, can be
        set via set_output_bias). This ensures the transcoder begins as a
        constant function.
        """
        with torch.no_grad():
            # Decoder weights start at zero (paper recommendation)
            nn.init.zeros_(self.decoder.weight)
            nn.init.zeros_(self.decoder.bias)

            # Skip weights start at zero (paper recommendation)
            nn.init.zeros_(self.skip.weight)
            nn.init.zeros_(self.skip.bias)

    def set_output_bias(self, mean_output: Tensor) -> None:
        """Set decoder bias to empirical mean of MLP outputs.

        This should be called before training with:
            mean_output = mlp_outputs.mean(dim=0)
            transcoder.set_output_bias(mean_output)

        Args:
            mean_output: Mean MLP output across training data [output_dim].
        """
        with torch.no_grad():
            self.decoder.bias.data = mean_output.clone()

    def normalize_decoder_weights(self) -> None:
        """Normalize decoder columns to unit norm."""
        with torch.no_grad():
            self.decoder.weight.data = nn.functional.normalize(
                self.decoder.weight.data, dim=0
            )

    def encode(self, x: Tensor) -> Tensor:
        """Encode to sparse latent space."""
        pre_activation = self.encoder(x)

        topk_values, topk_indices = torch.topk(pre_activation, self.k, dim=-1)
        hidden = torch.zeros_like(pre_activation)
        hidden.scatter_(-1, topk_indices, torch.relu(topk_values))

        return hidden

    def decode(self, hidden: Tensor) -> Tensor:
        """Decode sparse latent (without skip)."""
        return self.decoder(hidden)

    def forward(self, mlp_input: Tensor, mlp_output: Tensor) -> TranscoderOutput:
        """Forward pass with loss computation.

        Args:
            mlp_input: MLP input activations [batch, input_dim].
            mlp_output: MLP output activations [batch, output_dim] (target).

        Returns:
            TranscoderOutput with prediction, hidden, and losses.
        """
        # Sparse pathway
        hidden = self.encode(mlp_input)
        sparse_output = self.decode(hidden)

        # Skip pathway
        skip_output = self.skip(mlp_input)

        # Combined prediction
        predicted = sparse_output + skip_output

        # Losses
        reconstruction_loss = nn.functional.mse_loss(predicted, mlp_output)
        l0 = (hidden > 0).float().sum(dim=-1).mean()
        sparsity_loss = torch.tensor(0.0, device=mlp_input.device)
        loss = reconstruction_loss

        self._update_dead_features(hidden)

        return TranscoderOutput(
            predicted=predicted,
            hidden=hidden,
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            sparsity_loss=sparsity_loss,
            l0=l0,
        )

    def _update_dead_features(self, hidden: Tensor) -> None:
        """Update dead feature tracking."""
        if self.training:
            self.step_count += 1
            active_features = (hidden > 0).any(dim=0)
            self.feature_last_activated[active_features] = self.step_count

    def get_dead_features(self) -> Tensor:
        """Get mask of dead features."""
        steps_since_active = self.step_count - self.feature_last_activated
        return steps_since_active > self.dead_feature_threshold

    def get_dead_feature_ratio(self) -> float:
        """Get ratio of dead features."""
        return self.get_dead_features().float().mean().item()

    def get_skip_contribution(self, mlp_input: Tensor, mlp_output: Tensor) -> float:
        """Compute what fraction of output variance is explained by skip connection.

        Useful for diagnosing if sparse component is learning anything meaningful.

        Returns:
            Fraction of output variance explained by skip (0-1).
        """
        with torch.no_grad():
            skip_pred = self.skip(mlp_input)
            skip_var = ((skip_pred - mlp_output) ** 2).mean()
            total_var = ((mlp_output - mlp_output.mean(dim=0)) ** 2).mean()
            # Explained variance ratio
            skip_r2 = 1 - (skip_var / (total_var + 1e-8))
            return skip_r2.item()


def create_transcoder(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    k: int = 32,
    use_skip: bool = True,
    **kwargs,
) -> nn.Module:
    """Create a Transcoder from parameters.

    Args:
        input_dim: Dimension of MLP input.
        output_dim: Dimension of MLP output.
        hidden_dim: Number of sparse features.
        k: Number of active features per token.
        use_skip: Whether to use skip connection (SkipTranscoder).
        **kwargs: Additional arguments passed to constructor.

    Returns:
        Initialized Transcoder module.
    """
    if use_skip:
        return SkipTranscoder(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            k=k,
            **kwargs,
        )
    else:
        return TopKTranscoder(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            k=k,
            **kwargs,
        )
