"""Whisper activation extraction using hooks.

This module provides utilities to extract intermediate activations from
Whisper encoder and decoder layers for SAE training.
"""

from dataclasses import dataclass, field
from typing import Callable, Literal

import torch
from torch import Tensor, nn
from transformers import WhisperForConditionalGeneration


@dataclass
class ActivationCache:
    """Cache for storing activations from multiple layers."""

    encoder: dict[int, list[Tensor]] = field(default_factory=dict)
    decoder: dict[int, list[Tensor]] = field(default_factory=dict)

    def clear(self) -> None:
        """Clear all cached activations."""
        self.encoder.clear()
        self.decoder.clear()

    def get_encoder_activations(self, layer: int) -> Tensor | None:
        """Get concatenated encoder activations for a layer."""
        if layer not in self.encoder or not self.encoder[layer]:
            return None
        return torch.cat(self.encoder[layer], dim=0)

    def get_decoder_activations(self, layer: int) -> Tensor | None:
        """Get concatenated decoder activations for a layer."""
        if layer not in self.decoder or not self.decoder[layer]:
            return None
        return torch.cat(self.decoder[layer], dim=0)


class WhisperActivationExtractor:
    """Extract activations from Whisper model using hooks.

    This class uses forward hooks to capture intermediate layer activations
    from both the encoder and decoder of Whisper models.

    Key insight from aiOla paper: Apply layer norm before SAE for better features.
    """

    def __init__(
        self,
        model: WhisperForConditionalGeneration,
        encoder_layers: list[int] | None = None,
        decoder_layers: list[int] | None = None,
        apply_layer_norm: bool = True,
    ):
        """Initialize the extractor.

        Args:
            model: The Whisper model to extract activations from.
            encoder_layers: Which encoder layers to extract (0-indexed).
            decoder_layers: Which decoder layers to extract (0-indexed).
            apply_layer_norm: Whether to apply layer norm to activations.
        """
        self.model = model
        self.encoder_layers = encoder_layers or []
        self.decoder_layers = decoder_layers or []
        self.apply_layer_norm = apply_layer_norm
        self.cache = ActivationCache()
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []

        # Get layer norm modules for post-processing
        self._encoder_layer_norm = model.model.encoder.layer_norm
        self._decoder_layer_norm = model.model.decoder.layer_norm

    def _make_encoder_hook(self, layer_idx: int) -> Callable:
        """Create a hook function for an encoder layer."""

        def hook(module: nn.Module, input: tuple, output: Tensor | tuple) -> None:
            # Whisper encoder layers return (hidden_states, attention_weights)
            # We want just the hidden states
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # output shape: [batch, seq_len, hidden_dim]
            activation = hidden_states.detach()
            if self.apply_layer_norm:
                activation = self._encoder_layer_norm(activation)
            if layer_idx not in self.cache.encoder:
                self.cache.encoder[layer_idx] = []
            self.cache.encoder[layer_idx].append(activation.cpu())

        return hook

    def _make_decoder_hook(self, layer_idx: int) -> Callable:
        """Create a hook function for a decoder layer."""

        def hook(module: nn.Module, input: tuple, output: tuple) -> None:
            # Decoder layers return a tuple, first element is hidden states
            hidden_states = output[0].detach()
            if self.apply_layer_norm:
                hidden_states = self._decoder_layer_norm(hidden_states)
            if layer_idx not in self.cache.decoder:
                self.cache.decoder[layer_idx] = []
            self.cache.decoder[layer_idx].append(hidden_states.cpu())

        return hook

    def register_hooks(self) -> None:
        """Register forward hooks on specified layers."""
        self.remove_hooks()  # Clear any existing hooks

        # Register encoder hooks
        for layer_idx in self.encoder_layers:
            layer = self.model.model.encoder.layers[layer_idx]
            hook = layer.register_forward_hook(self._make_encoder_hook(layer_idx))
            self._hooks.append(hook)

        # Register decoder hooks
        for layer_idx in self.decoder_layers:
            layer = self.model.model.decoder.layers[layer_idx]
            hook = layer.register_forward_hook(self._make_decoder_hook(layer_idx))
            self._hooks.append(hook)

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def clear_cache(self) -> None:
        """Clear the activation cache."""
        self.cache.clear()

    def __enter__(self) -> "WhisperActivationExtractor":
        """Context manager entry - register hooks."""
        self.register_hooks()
        return self

    def __exit__(self, *args) -> None:
        """Context manager exit - remove hooks."""
        self.remove_hooks()


def extract_features_batch(
    model: WhisperForConditionalGeneration,
    input_features: Tensor,
    encoder_layers: list[int],
    decoder_layers: list[int],
    apply_layer_norm: bool = True,
    device: torch.device | str = "cpu",
) -> dict[str, dict[int, Tensor]]:
    """Extract features from a single batch.

    Args:
        model: Whisper model.
        input_features: Input mel spectrogram [batch, 80, 3000].
        encoder_layers: Which encoder layers to extract.
        decoder_layers: Which decoder layers to extract.
        apply_layer_norm: Whether to apply final layer norm.
        device: Device to run on.

    Returns:
        Dictionary with 'encoder' and 'decoder' keys, each containing
        a dict mapping layer index to activation tensor.
    """
    model.eval()
    input_features = input_features.to(device)

    extractor = WhisperActivationExtractor(
        model=model,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        apply_layer_norm=apply_layer_norm,
    )

    with torch.no_grad(), extractor:
        # Run encoder
        encoder_outputs = model.model.encoder(input_features)
        encoder_hidden = encoder_outputs.last_hidden_state

        # Run decoder with start token
        if decoder_layers:
            batch_size = input_features.size(0)
            decoder_input_ids = torch.full(
                (batch_size, 1),
                model.config.decoder_start_token_id,
                dtype=torch.long,
                device=device,
            )
            _ = model.model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_hidden,
            )

    # Collect results
    results: dict[str, dict[int, Tensor]] = {"encoder": {}, "decoder": {}}

    for layer_idx in encoder_layers:
        activations = extractor.cache.get_encoder_activations(layer_idx)
        if activations is not None:
            results["encoder"][layer_idx] = activations

    for layer_idx in decoder_layers:
        activations = extractor.cache.get_decoder_activations(layer_idx)
        if activations is not None:
            results["decoder"][layer_idx] = activations

    return results


def flatten_activations(
    activations: Tensor,
    component: Literal["encoder", "decoder"],
) -> Tensor:
    """Flatten activations for SAE training.

    Encoder activations: [batch, seq_len, hidden] -> [batch * seq_len, hidden]
    Decoder activations: [batch, seq_len, hidden] -> [batch * seq_len, hidden]

    Args:
        activations: Activation tensor from Whisper.
        component: Whether this is encoder or decoder.

    Returns:
        Flattened activations [num_tokens, hidden_dim].
    """
    batch_size, seq_len, hidden_dim = activations.shape
    return activations.view(-1, hidden_dim)
