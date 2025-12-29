"""Feature visualization and top-k activating examples.

This module provides tools for finding and visualizing the examples that
maximally activate each SAE/Transcoder/Crosscoder feature.

Key components:
- FeatureActivation: Data class for a single activation
- TopKTracker: Efficiently tracks top-k activations per feature
- FeatureReport: Generates interpretation reports for features
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import heapq
import json

import torch
from torch import Tensor


@dataclass
class FeatureActivation:
    """A single activation of a feature.

    Stores all metadata needed to understand what caused the activation.
    """

    feature_idx: int  # Which feature fired
    activation_value: float  # How strongly it fired
    sample_idx: int  # Which sample in the dataset
    position_idx: int  # Position in sequence (for Whisper: frame index)
    timestamp_ms: float | None = None  # Time in audio (10ms per frame for Whisper)
    transcription: str | None = None  # Full transcription of the sample
    transcription_context: str | None = None  # Words around activation
    audio_path: str | None = None  # Path to extracted audio clip
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "feature_idx": self.feature_idx,
            "activation_value": self.activation_value,
            "sample_idx": self.sample_idx,
            "position_idx": self.position_idx,
            "timestamp_ms": self.timestamp_ms,
            "transcription": self.transcription,
            "transcription_context": self.transcription_context,
            "audio_path": self.audio_path,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureActivation":
        """Create from dictionary."""
        return cls(**d)


class TopKTracker:
    """Efficiently tracks top-k activating examples per feature.

    Uses min-heaps to maintain only the k highest activations for each
    feature, avoiding memory issues with large datasets.

    Usage:
        tracker = TopKTracker(num_features=3072, k=20)

        for batch_idx, (activations, metadata) in enumerate(dataloader):
            tracker.update(activations, batch_idx, metadata)

        top_examples = tracker.get_top_examples()
    """

    def __init__(self, num_features: int, k: int = 20):
        """Initialize tracker.

        Args:
            num_features: Number of features in the SAE/Transcoder.
            k: Number of top examples to track per feature.
        """
        self.num_features = num_features
        self.k = k

        # Min-heaps per feature: (activation_value, FeatureActivation)
        # We use min-heap and keep only k largest by popping smallest
        self._heaps: list[list[tuple[float, FeatureActivation]]] = [
            [] for _ in range(num_features)
        ]

        # Statistics
        self.total_activations = 0
        self.samples_processed = 0

    def update(
        self,
        activations: Tensor,
        sample_indices: list[int] | Tensor,
        transcriptions: list[str] | None = None,
        metadata_list: list[dict] | None = None,
    ) -> None:
        """Update with a batch of activations.

        Args:
            activations: Activations tensor [batch, seq_len, num_features] or [batch, num_features].
            sample_indices: Sample indices in the dataset for this batch.
            transcriptions: Optional transcriptions for each sample.
            metadata_list: Optional metadata dicts for each sample.
        """
        activations = activations.detach().cpu()

        # Handle both [batch, features] and [batch, seq, features]
        if activations.ndim == 2:
            activations = activations.unsqueeze(1)  # [batch, 1, features]

        batch_size, seq_len, num_features = activations.shape
        assert num_features == self.num_features

        if isinstance(sample_indices, Tensor):
            sample_indices = sample_indices.tolist()

        for batch_idx in range(batch_size):
            sample_idx = sample_indices[batch_idx]
            transcription = transcriptions[batch_idx] if transcriptions else None
            metadata = metadata_list[batch_idx] if metadata_list else {}

            for pos_idx in range(seq_len):
                # Find active features in this position
                pos_acts = activations[batch_idx, pos_idx]  # [num_features]

                # Get indices of non-zero activations
                active_mask = pos_acts > 0
                active_indices = torch.where(active_mask)[0]

                for feat_idx in active_indices.tolist():
                    act_value = pos_acts[feat_idx].item()
                    self.total_activations += 1

                    # Calculate timestamp (Whisper uses 10ms frames)
                    timestamp_ms = pos_idx * 10.0

                    activation = FeatureActivation(
                        feature_idx=feat_idx,
                        activation_value=act_value,
                        sample_idx=sample_idx,
                        position_idx=pos_idx,
                        timestamp_ms=timestamp_ms,
                        transcription=transcription,
                        metadata=metadata.copy() if metadata else {},
                    )

                    heap = self._heaps[feat_idx]

                    if len(heap) < self.k:
                        heapq.heappush(heap, (act_value, activation))
                    elif act_value > heap[0][0]:
                        heapq.heapreplace(heap, (act_value, activation))

        self.samples_processed += batch_size

    def get_top_examples(self, feature_idx: int) -> list[FeatureActivation]:
        """Get top-k examples for a specific feature.

        Args:
            feature_idx: Which feature to get examples for.

        Returns:
            List of FeatureActivation sorted by activation value (descending).
        """
        heap = self._heaps[feature_idx]
        examples = [item[1] for item in heap]
        examples.sort(key=lambda x: x.activation_value, reverse=True)
        return examples

    def get_all_top_examples(self) -> dict[int, list[FeatureActivation]]:
        """Get top-k examples for all features.

        Returns:
            Dict mapping feature_idx to list of FeatureActivation.
        """
        return {i: self.get_top_examples(i) for i in range(self.num_features)}

    def get_feature_stats(self) -> dict[int, dict]:
        """Get statistics for each feature.

        Returns:
            Dict with per-feature stats: num_examples, max_activation, etc.
        """
        stats = {}
        for i in range(self.num_features):
            examples = self.get_top_examples(i)
            if examples:
                activations = [e.activation_value for e in examples]
                stats[i] = {
                    "num_examples": len(examples),
                    "max_activation": max(activations),
                    "min_activation": min(activations),
                    "mean_activation": sum(activations) / len(activations),
                }
            else:
                stats[i] = {
                    "num_examples": 0,
                    "max_activation": 0.0,
                    "min_activation": 0.0,
                    "mean_activation": 0.0,
                }
        return stats

    def save(self, path: Path | str) -> None:
        """Save tracker state to JSON.

        Args:
            path: Path to save to.
        """
        path = Path(path)
        data = {
            "num_features": self.num_features,
            "k": self.k,
            "total_activations": self.total_activations,
            "samples_processed": self.samples_processed,
            "features": {},
        }

        for i in range(self.num_features):
            examples = self.get_top_examples(i)
            if examples:
                data["features"][str(i)] = [e.to_dict() for e in examples]

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> "TopKTracker":
        """Load tracker from JSON.

        Args:
            path: Path to load from.

        Returns:
            Loaded TopKTracker.
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        tracker = cls(
            num_features=data["num_features"],
            k=data["k"],
        )
        tracker.total_activations = data["total_activations"]
        tracker.samples_processed = data["samples_processed"]

        for feat_idx_str, examples in data["features"].items():
            feat_idx = int(feat_idx_str)
            for e_dict in examples:
                activation = FeatureActivation.from_dict(e_dict)
                heap = tracker._heaps[feat_idx]
                heapq.heappush(heap, (activation.activation_value, activation))

        return tracker


@dataclass
class FeatureInterpretation:
    """Interpretation of what a feature represents."""

    feature_idx: int
    category: str  # e.g., "phoneme", "prosody", "speaker", "semantic", "unknown"
    description: str  # Human-readable description
    confidence: float  # 0-1 confidence in interpretation
    evidence: list[str] = field(default_factory=list)  # Supporting evidence
    automated_labels: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "feature_idx": self.feature_idx,
            "category": self.category,
            "description": self.description,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "automated_labels": self.automated_labels,
        }


class FeatureReport:
    """Generate interpretation reports for features."""

    def __init__(
        self,
        tracker: TopKTracker,
        output_dir: Path | str,
    ):
        """Initialize report generator.

        Args:
            tracker: TopKTracker with activation data.
            output_dir: Directory to save reports.
        """
        self.tracker = tracker
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.interpretations: dict[int, FeatureInterpretation] = {}

    def generate_feature_report(
        self,
        feature_idx: int,
        include_audio_paths: bool = True,
    ) -> dict:
        """Generate a report for a single feature.

        Args:
            feature_idx: Which feature to report on.
            include_audio_paths: Whether to include audio clip paths.

        Returns:
            Report dictionary.
        """
        examples = self.tracker.get_top_examples(feature_idx)
        stats = self.tracker.get_feature_stats()[feature_idx]

        report = {
            "feature_idx": feature_idx,
            "stats": stats,
            "top_examples": [],
        }

        for ex in examples:
            ex_data = {
                "activation_value": ex.activation_value,
                "sample_idx": ex.sample_idx,
                "position_idx": ex.position_idx,
                "timestamp_ms": ex.timestamp_ms,
                "transcription": ex.transcription,
            }
            if include_audio_paths and ex.audio_path:
                ex_data["audio_path"] = ex.audio_path
            report["top_examples"].append(ex_data)

        if feature_idx in self.interpretations:
            report["interpretation"] = self.interpretations[feature_idx].to_dict()

        return report

    def generate_summary_report(self, top_n: int = 100) -> dict:
        """Generate summary report for top features.

        Args:
            top_n: Number of features to include (by max activation).

        Returns:
            Summary report dictionary.
        """
        stats = self.tracker.get_feature_stats()

        # Sort features by max activation
        sorted_features = sorted(
            stats.items(),
            key=lambda x: x[1]["max_activation"],
            reverse=True,
        )[:top_n]

        return {
            "num_features": self.tracker.num_features,
            "samples_processed": self.tracker.samples_processed,
            "total_activations": self.tracker.total_activations,
            "top_features": [
                {
                    "feature_idx": feat_idx,
                    **feat_stats,
                }
                for feat_idx, feat_stats in sorted_features
            ],
        }

    def save_reports(self, top_n: int = 100) -> None:
        """Save all reports to output directory.

        Args:
            top_n: Number of top features to save individual reports for.
        """
        # Save summary
        summary = self.generate_summary_report(top_n=top_n)
        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Save individual feature reports
        features_dir = self.output_dir / "features"
        features_dir.mkdir(exist_ok=True)

        for feat_data in summary["top_features"]:
            feat_idx = feat_data["feature_idx"]
            report = self.generate_feature_report(feat_idx)
            with open(features_dir / f"feature_{feat_idx:05d}.json", "w") as f:
                json.dump(report, f, indent=2)

        # Save tracker state
        self.tracker.save(self.output_dir / "tracker_state.json")

    def add_interpretation(
        self,
        feature_idx: int,
        category: str,
        description: str,
        confidence: float = 0.5,
        evidence: list[str] | None = None,
    ) -> None:
        """Add manual interpretation for a feature.

        Args:
            feature_idx: Which feature.
            category: Category (phoneme, prosody, speaker, semantic, unknown).
            description: Human-readable description.
            confidence: Confidence in interpretation (0-1).
            evidence: Supporting evidence.
        """
        self.interpretations[feature_idx] = FeatureInterpretation(
            feature_idx=feature_idx,
            category=category,
            description=description,
            confidence=confidence,
            evidence=evidence or [],
        )


def collect_top_activations(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_features: int,
    k: int = 20,
    device: str = "cpu",
) -> TopKTracker:
    """Collect top-k activating examples from a dataset.

    Args:
        model: SAE/Transcoder/Crosscoder model.
        dataloader: DataLoader yielding (activations, metadata) or just activations.
        num_features: Number of features in the model.
        k: Number of top examples per feature.
        device: Device to run on.

    Returns:
        TopKTracker with collected activations.
    """
    tracker = TopKTracker(num_features=num_features, k=k)
    model.eval()

    sample_idx = 0
    with torch.no_grad():
        for batch in dataloader:
            # Handle different batch formats
            if isinstance(batch, (tuple, list)):
                activations = batch[0]
                metadata = batch[1] if len(batch) > 1 else None
            else:
                activations = batch
                metadata = None

            activations = activations.to(device)

            # Get hidden activations from model
            if hasattr(model, "encode"):
                hidden = model.encode(activations)
            else:
                output = model(activations)
                hidden = output.hidden if hasattr(output, "hidden") else output[1]

            # Generate sample indices for this batch
            batch_size = hidden.shape[0]
            sample_indices = list(range(sample_idx, sample_idx + batch_size))

            # Extract transcriptions if available
            transcriptions = None
            if metadata is not None and isinstance(metadata, dict):
                transcriptions = metadata.get("transcriptions")

            tracker.update(
                activations=hidden,
                sample_indices=sample_indices,
                transcriptions=transcriptions,
            )

            sample_idx += batch_size

    return tracker
