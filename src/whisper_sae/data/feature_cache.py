"""Multi-layer feature caching for efficient SAE training.

This module handles caching extracted Whisper activations to disk
so they don't need to be re-extracted for each SAE training run.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Literal

import torch
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from ..config import DataConfig, WhisperConfig
from ..sae.hooks import WhisperActivationExtractor, flatten_activations


@dataclass
class CacheMetadata:
    """Metadata about a cached feature file."""

    model_name: str
    component: Literal["encoder", "decoder"]
    layer_idx: int
    hidden_dim: int
    num_samples: int
    num_tokens: int
    created_at: str
    data_config: dict

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.__dict__, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "CacheMetadata":
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        return cls(**data)


class FeatureCache:
    """Cache for extracted Whisper features.

    Stores activations from each layer separately with metadata.
    """

    def __init__(
        self,
        cache_dir: Path,
        whisper_config: WhisperConfig,
        data_config: DataConfig,
    ):
        """Initialize feature cache.

        Args:
            cache_dir: Directory to store cached features.
            whisper_config: Whisper model configuration.
            data_config: Data configuration.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.whisper_config = whisper_config
        self.data_config = data_config

        # Model name for cache file naming
        self.model_short = whisper_config.model_name.split("/")[-1]

    def _get_cache_path(
        self,
        component: Literal["encoder", "decoder"],
        layer_idx: int,
    ) -> Path:
        """Get path for a cached feature file."""
        return self.cache_dir / f"{self.model_short}_{component}_layer{layer_idx}.pt"

    def _get_metadata_path(
        self,
        component: Literal["encoder", "decoder"],
        layer_idx: int,
    ) -> Path:
        """Get path for metadata file."""
        return self.cache_dir / f"{self.model_short}_{component}_layer{layer_idx}_meta.json"

    def has_cache(
        self,
        component: Literal["encoder", "decoder"],
        layer_idx: int,
    ) -> bool:
        """Check if cache exists for a specific layer."""
        cache_path = self._get_cache_path(component, layer_idx)
        meta_path = self._get_metadata_path(component, layer_idx)
        return cache_path.exists() and meta_path.exists()

    def load(
        self,
        component: Literal["encoder", "decoder"],
        layer_idx: int,
    ) -> tuple[Tensor, CacheMetadata]:
        """Load cached features for a layer.

        Args:
            component: encoder or decoder.
            layer_idx: Layer index.

        Returns:
            Tuple of (features tensor, metadata).
        """
        cache_path = self._get_cache_path(component, layer_idx)
        meta_path = self._get_metadata_path(component, layer_idx)

        features = torch.load(cache_path, weights_only=True)
        with open(meta_path) as f:
            metadata = CacheMetadata.from_json(f.read())

        return features, metadata

    def save(
        self,
        features: Tensor,
        component: Literal["encoder", "decoder"],
        layer_idx: int,
        num_samples: int,
    ) -> None:
        """Save features to cache.

        Args:
            features: Flattened activation tensor [num_tokens, hidden_dim].
            component: encoder or decoder.
            layer_idx: Layer index.
            num_samples: Number of audio samples processed.
        """
        cache_path = self._get_cache_path(component, layer_idx)
        meta_path = self._get_metadata_path(component, layer_idx)

        torch.save(features, cache_path)

        metadata = CacheMetadata(
            model_name=self.whisper_config.model_name,
            component=component,
            layer_idx=layer_idx,
            hidden_dim=features.shape[-1],
            num_samples=num_samples,
            num_tokens=features.shape[0],
            created_at=datetime.now().isoformat(),
            data_config=self.data_config.model_dump(),
        )
        with open(meta_path, "w") as f:
            f.write(metadata.to_json())

    def get_dataloader(
        self,
        component: Literal["encoder", "decoder"],
        layer_idx: int,
        batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> DataLoader:
        """Get a DataLoader for cached features.

        Args:
            component: encoder or decoder.
            layer_idx: Layer index.
            batch_size: Batch size.
            shuffle: Whether to shuffle.
            num_workers: Number of data loading workers.

        Returns:
            DataLoader for the cached features.
        """
        features, _ = self.load(component, layer_idx)
        dataset = TensorDataset(features)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )


def extract_and_cache_features(
    whisper_model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    audio_dataloader: DataLoader,
    cache: FeatureCache,
    encoder_layers: list[int],
    decoder_layers: list[int],
    device: torch.device | str = "cpu",
    max_samples: int | None = None,
) -> None:
    """Extract features from Whisper and cache them.

    Args:
        whisper_model: The Whisper model.
        processor: Whisper processor.
        audio_dataloader: DataLoader yielding processed audio features.
        cache: Feature cache to save to.
        encoder_layers: Which encoder layers to extract.
        decoder_layers: Which decoder layers to extract.
        device: Device to run on.
        max_samples: Maximum samples to process.
    """
    whisper_model = whisper_model.to(device)
    whisper_model.eval()

    # Initialize accumulators for each layer
    encoder_features: dict[int, list[Tensor]] = {l: [] for l in encoder_layers}
    decoder_features: dict[int, list[Tensor]] = {l: [] for l in decoder_layers}

    extractor = WhisperActivationExtractor(
        model=whisper_model,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        apply_layer_norm=True,
    )

    num_samples = 0
    target_samples = max_samples or float("inf")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        task = progress.add_task(
            "[cyan]Extracting features...",
            total=max_samples if max_samples else None,
        )

        with torch.no_grad(), extractor:
            for batch in audio_dataloader:
                if num_samples >= target_samples:
                    break

                # Handle different batch formats
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]

                batch = batch.to(device)
                batch_size = batch.shape[0]

                # Run encoder
                encoder_outputs = whisper_model.model.encoder(batch)
                encoder_hidden = encoder_outputs.last_hidden_state

                # Run decoder with start token
                if decoder_layers:
                    decoder_input_ids = torch.full(
                        (batch_size, 1),
                        whisper_model.config.decoder_start_token_id,
                        dtype=torch.long,
                        device=device,
                    )
                    _ = whisper_model.model.decoder(
                        input_ids=decoder_input_ids,
                        encoder_hidden_states=encoder_hidden,
                    )

                # Collect activations from cache
                for layer_idx in encoder_layers:
                    if extractor.cache.encoder.get(layer_idx):
                        acts = extractor.cache.encoder[layer_idx][-1]
                        flattened = flatten_activations(acts, "encoder")
                        encoder_features[layer_idx].append(flattened)

                for layer_idx in decoder_layers:
                    if extractor.cache.decoder.get(layer_idx):
                        acts = extractor.cache.decoder[layer_idx][-1]
                        flattened = flatten_activations(acts, "decoder")
                        decoder_features[layer_idx].append(flattened)

                num_samples += batch_size
                progress.update(task, completed=min(num_samples, target_samples))

    # Concatenate and save
    for layer_idx in encoder_layers:
        if encoder_features[layer_idx]:
            features = torch.cat(encoder_features[layer_idx], dim=0)
            cache.save(features, "encoder", layer_idx, num_samples)
            print(f"Cached encoder layer {layer_idx}: {features.shape}")

    for layer_idx in decoder_layers:
        if decoder_features[layer_idx]:
            features = torch.cat(decoder_features[layer_idx], dim=0)
            cache.save(features, "decoder", layer_idx, num_samples)
            print(f"Cached decoder layer {layer_idx}: {features.shape}")
