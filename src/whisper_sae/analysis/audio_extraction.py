"""Audio clip extraction for feature interpretation.

Extracts audio clips centered on high-activation positions, allowing
researchers to listen to what each feature responds to.

Whisper timing:
- 16kHz audio sampling rate
- 10ms per encoder frame (160 samples)
- Typical clip: 1 second centered on activation (50 frames before/after)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import json

import torch
from torch import Tensor

from .feature_viz import FeatureActivation, TopKTracker


@dataclass
class AudioClipConfig:
    """Configuration for audio clip extraction."""

    sample_rate: int = 16000  # Whisper uses 16kHz
    samples_per_frame: int = 160  # 10ms at 16kHz
    clip_duration_ms: float = 1000.0  # 1 second clips by default
    context_before_ms: float = 500.0  # Half before, half after
    output_format: str = "wav"
    normalize_audio: bool = True


class AudioClipExtractor:
    """Extract audio clips for feature activations.

    Given a TopKTracker with activation data and access to the original
    audio, extracts clips centered on each activation.

    Usage:
        extractor = AudioClipExtractor(
            tracker=tracker,
            audio_loader=load_audio_fn,  # fn(sample_idx) -> audio tensor
            output_dir="clips/",
        )
        extractor.extract_all_clips()
    """

    def __init__(
        self,
        tracker: TopKTracker,
        audio_loader: Callable[[int], Tensor],
        output_dir: Path | str,
        config: AudioClipConfig | None = None,
    ):
        """Initialize extractor.

        Args:
            tracker: TopKTracker with activation data.
            audio_loader: Function that loads audio given sample index.
                         Should return Tensor of shape [num_samples] or [1, num_samples].
            output_dir: Directory to save audio clips.
            config: Audio extraction configuration.
        """
        self.tracker = tracker
        self.audio_loader = audio_loader
        self.output_dir = Path(output_dir)
        self.config = config or AudioClipConfig()

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _frame_to_sample(self, frame_idx: int) -> int:
        """Convert frame index to audio sample index."""
        return frame_idx * self.config.samples_per_frame

    def _ms_to_samples(self, ms: float) -> int:
        """Convert milliseconds to audio samples."""
        return int(ms * self.config.sample_rate / 1000)

    def extract_clip(
        self,
        activation: FeatureActivation,
        audio: Tensor | None = None,
    ) -> Tensor | None:
        """Extract audio clip for a single activation.

        Args:
            activation: The activation to extract a clip for.
            audio: Pre-loaded audio tensor, or None to load from audio_loader.

        Returns:
            Audio clip tensor, or None if extraction failed.
        """
        if audio is None:
            try:
                audio = self.audio_loader(activation.sample_idx)
            except Exception:
                return None

        if audio.ndim == 2:
            audio = audio.squeeze(0)

        # Calculate clip boundaries
        center_sample = self._frame_to_sample(activation.position_idx)
        context_samples = self._ms_to_samples(self.config.context_before_ms)
        clip_samples = self._ms_to_samples(self.config.clip_duration_ms)

        start_sample = max(0, center_sample - context_samples)
        end_sample = min(len(audio), start_sample + clip_samples)

        # Extract clip
        clip = audio[start_sample:end_sample]

        # Normalize if requested
        if self.config.normalize_audio and clip.abs().max() > 0:
            clip = clip / clip.abs().max() * 0.95

        return clip

    def extract_feature_clips(
        self,
        feature_idx: int,
        max_clips: int | None = None,
    ) -> list[Path]:
        """Extract all clips for a single feature.

        Args:
            feature_idx: Which feature to extract clips for.
            max_clips: Maximum number of clips to extract.

        Returns:
            List of paths to saved audio clips.
        """
        examples = self.tracker.get_top_examples(feature_idx)
        if max_clips:
            examples = examples[:max_clips]

        # Create feature directory
        feature_dir = self.output_dir / f"feature_{feature_idx:05d}"
        feature_dir.mkdir(exist_ok=True)

        saved_paths = []
        audio_cache: dict[int, Tensor] = {}

        for rank, activation in enumerate(examples):
            # Load audio (with caching for multiple clips from same sample)
            sample_idx = activation.sample_idx
            if sample_idx not in audio_cache:
                try:
                    audio_cache[sample_idx] = self.audio_loader(sample_idx)
                except Exception as e:
                    print(f"Failed to load audio for sample {sample_idx}: {e}")
                    continue

            audio = audio_cache[sample_idx]
            clip = self.extract_clip(activation, audio)

            if clip is None:
                continue

            # Save clip
            clip_path = feature_dir / f"rank{rank:02d}_act{activation.activation_value:.3f}.{self.config.output_format}"
            self._save_audio(clip, clip_path)
            saved_paths.append(clip_path)

            # Update activation with clip path
            activation.audio_path = str(clip_path)

        return saved_paths

    def extract_all_clips(
        self,
        feature_indices: list[int] | None = None,
        max_clips_per_feature: int | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[int, list[Path]]:
        """Extract clips for all features (or a subset).

        Args:
            feature_indices: Which features to extract. None = all with examples.
            max_clips_per_feature: Max clips per feature.
            progress_callback: Optional callback(current, total) for progress.

        Returns:
            Dict mapping feature_idx to list of clip paths.
        """
        if feature_indices is None:
            # Find features with examples
            feature_indices = [
                i for i in range(self.tracker.num_features)
                if self.tracker.get_top_examples(i)
            ]

        all_clips = {}
        total = len(feature_indices)

        for idx, feat_idx in enumerate(feature_indices):
            if progress_callback:
                progress_callback(idx, total)

            clips = self.extract_feature_clips(
                feat_idx,
                max_clips=max_clips_per_feature,
            )
            if clips:
                all_clips[feat_idx] = clips

        return all_clips

    def _save_audio(self, audio: Tensor, path: Path) -> None:
        """Save audio tensor to file.

        Args:
            audio: Audio tensor [num_samples].
            path: Output path.
        """
        try:
            import soundfile as sf

            audio_np = audio.numpy()
            sf.write(path, audio_np, self.config.sample_rate)
        except ImportError:
            # Fallback to torchaudio if available
            import torchaudio

            if audio.ndim == 1:
                audio = audio.unsqueeze(0)
            torchaudio.save(str(path), audio, self.config.sample_rate)

    def save_manifest(self) -> Path:
        """Save a manifest of all extracted clips.

        Returns:
            Path to manifest file.
        """
        manifest = {
            "config": {
                "sample_rate": self.config.sample_rate,
                "clip_duration_ms": self.config.clip_duration_ms,
                "output_format": self.config.output_format,
            },
            "features": {},
        }

        for feat_idx in range(self.tracker.num_features):
            examples = self.tracker.get_top_examples(feat_idx)
            if examples:
                manifest["features"][str(feat_idx)] = [
                    {
                        "rank": i,
                        "activation_value": ex.activation_value,
                        "sample_idx": ex.sample_idx,
                        "position_idx": ex.position_idx,
                        "timestamp_ms": ex.timestamp_ms,
                        "audio_path": ex.audio_path,
                        "transcription": ex.transcription,
                    }
                    for i, ex in enumerate(examples)
                    if ex.audio_path
                ]

        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        return manifest_path


def create_librispeech_audio_loader(
    dataset_path: str | None = None,
    split: str = "train.100",
) -> Callable[[int], Tensor]:
    """Create an audio loader for LibriSpeech dataset.

    Args:
        dataset_path: Path to local LibriSpeech or None to stream.
        split: Dataset split to use.

    Returns:
        Function that loads audio by sample index.
    """
    import io
    import soundfile as sf
    from datasets import Audio, load_dataset

    # Load dataset with decode=False for raw bytes
    dataset = load_dataset(
        "librispeech_asr",
        "clean",
        split=split,
        streaming=True,
    ).cast_column("audio", Audio(decode=False))

    # Convert to list for indexing (only works for non-streaming)
    # For streaming, we'll need a different approach
    _cache: dict[int, Tensor] = {}
    _dataset_iter = iter(dataset)
    _current_idx = 0

    def load_audio(sample_idx: int) -> Tensor:
        nonlocal _current_idx, _dataset_iter

        # Check cache
        if sample_idx in _cache:
            return _cache[sample_idx]

        # Seek to the right position (inefficient for random access)
        while _current_idx <= sample_idx:
            try:
                sample = next(_dataset_iter)
                audio_bytes = sample["audio"]["bytes"]
                audio_array, sr = sf.read(io.BytesIO(audio_bytes))
                _cache[_current_idx] = torch.from_numpy(audio_array).float()
                _current_idx += 1
            except StopIteration:
                raise IndexError(f"Sample index {sample_idx} out of range")

        return _cache[sample_idx]

    return load_audio


def create_indexed_audio_loader(
    audio_paths: list[Path | str],
) -> Callable[[int], Tensor]:
    """Create an audio loader from a list of audio file paths.

    Args:
        audio_paths: List of paths to audio files, indexed by sample_idx.

    Returns:
        Function that loads audio by sample index.
    """
    import soundfile as sf

    def load_audio(sample_idx: int) -> Tensor:
        if sample_idx >= len(audio_paths):
            raise IndexError(f"Sample index {sample_idx} out of range")

        path = audio_paths[sample_idx]
        audio_array, sr = sf.read(path)
        return torch.from_numpy(audio_array).float()

    return load_audio
