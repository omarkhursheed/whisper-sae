"""LibriSpeech dataset loading and preprocessing.

This module provides utilities for loading and processing LibriSpeech data
for Whisper SAE training. Based on the v1 caching pattern with improvements.
"""

import os
from itertools import islice
from pathlib import Path
from typing import Iterator

import torch
import torchaudio
from datasets import load_dataset, IterableDataset
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperProcessor

from ..config import DataConfig


class LibriSpeechDataset(Dataset):
    """LibriSpeech dataset with caching for Whisper feature extraction.

    This dataset processes audio files into mel spectrograms suitable for
    Whisper model input. Processed samples are cached to disk for efficiency.
    """

    def __init__(
        self,
        processor: WhisperProcessor,
        config: DataConfig,
        split: str = "train",
    ):
        """Initialize the dataset.

        Args:
            processor: Whisper processor for audio preprocessing.
            config: Data configuration.
            split: Which split to use (train, validation, test).
        """
        self.processor = processor
        self.config = config
        self.samples: list[Tensor] = []
        self.metadata: list[dict] = []

        # Set up cache
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / f"librispeech_{config.dataset_subset}_{split}_{config.max_samples}.pt"
        self.meta_file = self.cache_dir / f"librispeech_{config.dataset_subset}_{split}_{config.max_samples}_meta.pt"

        # Load or process
        if self.cache_file.exists() and self.meta_file.exists():
            self._load_from_cache()
        else:
            self._process_and_cache()

    def _load_from_cache(self) -> None:
        """Load processed samples from cache."""
        print(f"Loading cached samples from {self.cache_file}")
        self.samples = torch.load(self.cache_file, weights_only=True)
        self.metadata = torch.load(self.meta_file, weights_only=False)
        print(f"Loaded {len(self.samples)} samples from cache")

    def _process_and_cache(self) -> None:
        """Process raw audio and cache to disk."""
        print(f"Processing LibriSpeech {self.config.dataset_subset} split...")

        # Load streaming dataset
        dataset = load_dataset(
            self.config.dataset_name,
            self.config.dataset_subset,
            split=self.config.dataset_split,
            streaming=self.config.streaming,
            trust_remote_code=True,
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ) as progress:
            task = progress.add_task(
                "[cyan]Processing audio samples...",
                total=self.config.max_samples,
            )

            for sample in islice(dataset, self.config.max_samples):
                processed = self._process_sample(sample)
                if processed is not None:
                    input_features, meta = processed
                    self.samples.append(input_features)
                    self.metadata.append(meta)
                    progress.update(task, advance=1)

        # Save to cache
        print(f"Saving {len(self.samples)} samples to cache...")
        torch.save(self.samples, self.cache_file)
        torch.save(self.metadata, self.meta_file)
        print(f"Cache saved to {self.cache_file}")

    def _process_sample(self, sample: dict) -> tuple[Tensor, dict] | None:
        """Process a single audio sample.

        Args:
            sample: Raw sample from HuggingFace dataset.

        Returns:
            Tuple of (input_features, metadata) or None if processing fails.
        """
        try:
            speech_array = sample["audio"]["array"]
            sampling_rate = sample["audio"]["sampling_rate"]

            # Resample to 16kHz if needed
            if sampling_rate != 16000:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sampling_rate,
                    new_freq=16000,
                )
                speech_tensor = torch.from_numpy(speech_array).float()
                speech_array = resampler(speech_tensor).numpy()
            else:
                speech_array = speech_array

            # Handle multi-channel audio by averaging
            if speech_array.ndim > 1:
                speech_array = speech_array.mean(axis=0)

            # Process through Whisper processor
            input_features = self.processor(
                speech_array,
                sampling_rate=16000,
                return_tensors="pt",
            ).input_features.squeeze(0)

            # Extract metadata
            metadata = {
                "id": sample.get("id", ""),
                "text": sample.get("text", ""),
                "speaker_id": sample.get("speaker_id", ""),
                "chapter_id": sample.get("chapter_id", ""),
            }

            return input_features, metadata

        except Exception as e:
            print(f"Error processing sample: {e}")
            return None

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, dict]:
        """Get a sample by index.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (input_features, metadata).
        """
        return self.samples[idx], self.metadata[idx]


class LibriSpeechFeaturesOnly(Dataset):
    """Wrapper that only returns input features (no metadata).

    Useful for DataLoaders where we just need the audio features.
    """

    def __init__(self, base_dataset: LibriSpeechDataset):
        """Initialize wrapper.

        Args:
            base_dataset: The underlying LibriSpeechDataset.
        """
        self.base = base_dataset

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> Tensor:
        return self.base.samples[idx]


def create_librispeech_dataloader(
    processor: WhisperProcessor,
    config: DataConfig,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader for LibriSpeech.

    Args:
        processor: Whisper processor.
        config: Data configuration.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        shuffle: Whether to shuffle.

    Returns:
        DataLoader yielding batches of input features.
    """
    dataset = LibriSpeechDataset(processor, config)
    features_only = LibriSpeechFeaturesOnly(dataset)

    return DataLoader(
        features_only,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
