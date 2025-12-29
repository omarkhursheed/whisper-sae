"""Crosscoder model implementations.

Crosscoders learn shared sparse representations across multiple layers,
allowing features to span the residual stream and tracking persistent
features through the model.

Reference: Anthropic (2024) - "Sparse Crosscoders for Cross-Layer Features"
https://transformer-circuits.pub/2024/crosscoders/index.html

Key differences from SAEs:
- SAE: Single layer activation -> sparse -> reconstruction
- Crosscoder: Multiple layer activations -> shared sparse -> per-layer reconstructions

For Whisper encoder analysis:
- Find features that persist across encoder layers (e.g., phoneme features)
- Understand how acoustic features transform into semantic features
- Resolve cross-layer superposition
"""

from typing import NamedTuple

import torch
from torch import Tensor, nn


class CrosscoderOutput(NamedTuple):
    """Output from Crosscoder forward pass."""

    reconstructed: dict[int, Tensor]  # Per-layer reconstructions
    hidden: Tensor  # Shared sparse latent activations
    loss: Tensor  # Total loss
    reconstruction_loss: Tensor  # Sum of per-layer MSE losses
    sparsity_loss: Tensor  # Decoder norm-weighted L1
    l0: Tensor  # Number of active features
    per_layer_loss: dict[int, Tensor]  # Individual layer losses


class CrossLayerCrosscoder(nn.Module):
    """Cross-layer sparse crosscoder for finding shared features across layers.

    Architecture follows Anthropic's crosscoder paper:
        - W_enc: [n_layers, d_model, d_sae] - per-layer encoders
        - W_dec: [d_sae, n_layers, d_model] - shared decoder
        - Uses sum of per-layer encoder outputs for latent

    For Whisper encoder:
        - Input: Activations from layers 0-3
        - Output: Shared sparse features that may span multiple layers
        - Features active in multiple layers = cross-layer features
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        d_sae: int,
        layer_indices: list[int] | None = None,
        activation: str = "relu",
        sparsity_weight: float = 0.01,
        normalize_decoder: bool = True,
        dead_feature_threshold: int = 10_000,
    ):
        """Initialize CrossLayerCrosscoder.

        Args:
            d_model: Model hidden dimension (e.g., 384 for Whisper-tiny).
            n_layers: Number of layers to encode from.
            d_sae: Number of sparse features (latent dimension).
            layer_indices: Which layer indices this crosscoder covers.
            activation: Activation function ('relu' or 'topk').
            sparsity_weight: Weight for L1 sparsity penalty (for relu).
            normalize_decoder: Whether to normalize decoder columns.
            dead_feature_threshold: Steps until feature is considered dead.
        """
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_sae = d_sae
        self.layer_indices = layer_indices or list(range(n_layers))
        self.activation = activation
        self.sparsity_weight = sparsity_weight
        self.normalize_decoder = normalize_decoder
        self.dead_feature_threshold = dead_feature_threshold

        # Per-layer encoders: [n_layers, d_model, d_sae]
        # Each layer has its own encoder to the shared latent space
        self.W_enc = nn.Parameter(torch.empty(n_layers, d_model, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))

        # Shared decoder: [d_sae, n_layers, d_model]
        # Single decoder that writes to all layers
        self.W_dec = nn.Parameter(torch.empty(d_sae, n_layers, d_model))
        self.b_dec = nn.Parameter(torch.zeros(n_layers, d_model))

        # Initialize weights
        self._init_weights()

        # Dead feature tracking
        self.register_buffer(
            "feature_last_activated",
            torch.zeros(d_sae, dtype=torch.long),
        )
        self.register_buffer("step_count", torch.tensor(0, dtype=torch.long))

    def _init_weights(self) -> None:
        """Initialize weights following Anthropic's recommendations."""
        with torch.no_grad():
            # Initialize decoder with small random weights, then normalize
            nn.init.xavier_uniform_(self.W_dec)
            if self.normalize_decoder:
                # Normalize across the (n_layers, d_model) output dimension
                W_dec_flat = self.W_dec.view(self.d_sae, -1)  # [d_sae, n_layers*d_model]
                W_dec_flat = nn.functional.normalize(W_dec_flat, dim=1)
                self.W_dec.data = W_dec_flat.view(self.d_sae, self.n_layers, self.d_model)
                self.W_dec.data *= 0.1

            # Initialize encoder as transpose of decoder (Anthropic method)
            # This ensures encoder and decoder start aligned
            for l in range(self.n_layers):
                self.W_enc.data[l] = self.W_dec.data[:, l, :].T

    def normalize_decoder_weights(self) -> None:
        """Normalize decoder columns (call after optimizer step)."""
        with torch.no_grad():
            W_dec_flat = self.W_dec.view(self.d_sae, -1)
            W_dec_flat = nn.functional.normalize(W_dec_flat, dim=1)
            self.W_dec.data = W_dec_flat.view(self.d_sae, self.n_layers, self.d_model)

    def get_decoder_norms(self) -> Tensor:
        """Get L2 norm of each feature's decoder weights across all layers.

        Returns:
            Tensor of shape [d_sae] with decoder norms.
        """
        W_dec_flat = self.W_dec.view(self.d_sae, -1)  # [d_sae, n_layers*d_model]
        return torch.norm(W_dec_flat, dim=1)

    def encode(self, layer_activations: dict[int, Tensor]) -> Tensor:
        """Encode multi-layer activations to shared latent space.

        Args:
            layer_activations: Dict mapping layer index to activations [batch, d_model].

        Returns:
            Sparse latent activations [batch, d_sae].
        """
        batch_size = next(iter(layer_activations.values())).shape[0]
        device = next(iter(layer_activations.values())).device

        # Sum contributions from each layer
        pre_activation = torch.zeros(batch_size, self.d_sae, device=device)

        for layer_idx, acts in layer_activations.items():
            # Map layer index to internal index
            internal_idx = self.layer_indices.index(layer_idx)
            # Encode: [batch, d_model] @ [d_model, d_sae] -> [batch, d_sae]
            pre_activation = pre_activation + torch.einsum(
                "bd,ds->bs", acts, self.W_enc[internal_idx]
            )

        pre_activation = pre_activation + self.b_enc

        # Apply activation
        if self.activation == "relu":
            hidden = torch.relu(pre_activation)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

        return hidden

    def decode(self, hidden: Tensor) -> dict[int, Tensor]:
        """Decode shared latent to per-layer reconstructions.

        Args:
            hidden: Sparse latent activations [batch, d_sae].

        Returns:
            Dict mapping layer index to reconstructed activations [batch, d_model].
        """
        reconstructed = {}

        for i, layer_idx in enumerate(self.layer_indices):
            # Decode: [batch, d_sae] @ [d_sae, d_model] -> [batch, d_model]
            recon = torch.einsum("bs,sd->bd", hidden, self.W_dec[:, i, :])
            recon = recon + self.b_dec[i]
            reconstructed[layer_idx] = recon

        return reconstructed

    def forward(self, layer_activations: dict[int, Tensor]) -> CrosscoderOutput:
        """Forward pass with loss computation.

        Args:
            layer_activations: Dict mapping layer index to activations [batch, d_model].

        Returns:
            CrosscoderOutput with reconstructions, hidden, and losses.
        """
        # Encode and decode
        hidden = self.encode(layer_activations)
        reconstructed = self.decode(hidden)

        # Compute per-layer reconstruction losses
        per_layer_loss = {}
        total_recon_loss = torch.tensor(0.0, device=hidden.device)

        for layer_idx, recon in reconstructed.items():
            target = layer_activations[layer_idx]
            layer_loss = torch.mean((recon - target) ** 2)
            per_layer_loss[layer_idx] = layer_loss
            total_recon_loss = total_recon_loss + layer_loss

        # Sparsity loss: decoder norm-weighted L1
        # Following Anthropic: sum decoder norms separately, multiply by activation
        decoder_norms = self.get_decoder_norms()  # [d_sae]
        sparsity_loss = torch.mean(hidden.abs() @ decoder_norms)

        # Total loss
        loss = total_recon_loss + self.sparsity_weight * sparsity_loss

        # L0 (number of active features)
        l0 = (hidden > 0).float().sum(dim=-1).mean()

        # Update dead feature tracking
        self._update_dead_features(hidden)

        return CrosscoderOutput(
            reconstructed=reconstructed,
            hidden=hidden,
            loss=loss,
            reconstruction_loss=total_recon_loss,
            sparsity_loss=sparsity_loss,
            l0=l0,
            per_layer_loss=per_layer_loss,
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
        return self.get_dead_features().float().mean().item()

    def get_feature_layer_norms(self) -> Tensor:
        """Get per-layer decoder norms for each feature.

        Useful for identifying which layers each feature primarily affects.

        Returns:
            Tensor of shape [d_sae, n_layers] with per-layer norms.
        """
        # [d_sae, n_layers, d_model] -> [d_sae, n_layers]
        return torch.norm(self.W_dec, dim=2)

    def get_cross_layer_features(self, threshold: float = 0.1) -> Tensor:
        """Identify features that are active across multiple layers.

        A feature is considered "cross-layer" if its decoder has substantial
        magnitude in multiple layers.

        Args:
            threshold: Minimum relative norm to count as active in a layer.

        Returns:
            Boolean tensor [d_sae] indicating cross-layer features.
        """
        layer_norms = self.get_feature_layer_norms()  # [d_sae, n_layers]
        max_norms = layer_norms.max(dim=1, keepdim=True).values
        relative_norms = layer_norms / (max_norms + 1e-8)

        # Count how many layers each feature is active in
        layers_active = (relative_norms > threshold).sum(dim=1)

        # Cross-layer = active in 2+ layers
        return layers_active >= 2


class TopKCrossLayerCrosscoder(CrossLayerCrosscoder):
    """Cross-layer crosscoder with TopK activation for enforced sparsity.

    Uses TopK instead of ReLU+L1 for more stable training (like our SAEs).
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        d_sae: int,
        k: int = 32,
        layer_indices: list[int] | None = None,
        normalize_decoder: bool = True,
        dead_feature_threshold: int = 10_000,
    ):
        """Initialize TopK CrossLayerCrosscoder.

        Args:
            d_model: Model hidden dimension.
            n_layers: Number of layers.
            d_sae: Number of sparse features.
            k: Number of active features to keep.
            layer_indices: Which layer indices this covers.
            normalize_decoder: Whether to normalize decoder.
            dead_feature_threshold: Steps until feature is dead.
        """
        super().__init__(
            d_model=d_model,
            n_layers=n_layers,
            d_sae=d_sae,
            layer_indices=layer_indices,
            activation="relu",  # Will be overridden in encode
            sparsity_weight=0.0,  # TopK doesn't need L1
            normalize_decoder=normalize_decoder,
            dead_feature_threshold=dead_feature_threshold,
        )
        self.k = k

    def encode(self, layer_activations: dict[int, Tensor]) -> Tensor:
        """Encode with TopK activation."""
        batch_size = next(iter(layer_activations.values())).shape[0]
        device = next(iter(layer_activations.values())).device

        # Sum contributions from each layer
        pre_activation = torch.zeros(batch_size, self.d_sae, device=device)

        for layer_idx, acts in layer_activations.items():
            internal_idx = self.layer_indices.index(layer_idx)
            pre_activation = pre_activation + torch.einsum(
                "bd,ds->bs", acts, self.W_enc[internal_idx]
            )

        pre_activation = pre_activation + self.b_enc

        # TopK activation
        topk_values, topk_indices = torch.topk(pre_activation, self.k, dim=-1)
        hidden = torch.zeros_like(pre_activation)
        hidden.scatter_(-1, topk_indices, torch.relu(topk_values))

        return hidden

    def forward(self, layer_activations: dict[int, Tensor]) -> CrosscoderOutput:
        """Forward pass with TopK activation."""
        hidden = self.encode(layer_activations)
        reconstructed = self.decode(hidden)

        # Per-layer reconstruction losses
        per_layer_loss = {}
        total_recon_loss = torch.tensor(0.0, device=hidden.device)

        for layer_idx, recon in reconstructed.items():
            target = layer_activations[layer_idx]
            layer_loss = torch.mean((recon - target) ** 2)
            per_layer_loss[layer_idx] = layer_loss
            total_recon_loss = total_recon_loss + layer_loss

        # No sparsity loss for TopK (sparsity is enforced)
        sparsity_loss = torch.tensor(0.0, device=hidden.device)
        loss = total_recon_loss

        l0 = (hidden > 0).float().sum(dim=-1).mean()
        self._update_dead_features(hidden)

        return CrosscoderOutput(
            reconstructed=reconstructed,
            hidden=hidden,
            loss=loss,
            reconstruction_loss=total_recon_loss,
            sparsity_loss=sparsity_loss,
            l0=l0,
            per_layer_loss=per_layer_loss,
        )


def create_crosscoder(
    d_model: int,
    n_layers: int,
    d_sae: int,
    k: int | None = None,
    use_topk: bool = True,
    **kwargs,
) -> nn.Module:
    """Create a Crosscoder from parameters.

    Args:
        d_model: Model hidden dimension.
        n_layers: Number of layers.
        d_sae: Number of sparse features.
        k: Number of active features (for TopK).
        use_topk: Whether to use TopK activation.
        **kwargs: Additional arguments.

    Returns:
        Initialized Crosscoder module.
    """
    if use_topk:
        return TopKCrossLayerCrosscoder(
            d_model=d_model,
            n_layers=n_layers,
            d_sae=d_sae,
            k=k or 32,
            **kwargs,
        )
    else:
        return CrossLayerCrosscoder(
            d_model=d_model,
            n_layers=n_layers,
            d_sae=d_sae,
            **kwargs,
        )
