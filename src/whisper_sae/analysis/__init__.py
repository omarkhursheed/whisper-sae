"""Analysis tools for feature interpretation.

This module provides tools for understanding what SAE/Transcoder/Crosscoder
features represent:

- TopKTracker: Track top-k activating examples per feature
- AudioClipExtractor: Extract audio clips around high activations
- FeatureReport: Generate interpretation reports
"""

from .feature_viz import (
    FeatureActivation,
    FeatureInterpretation,
    FeatureReport,
    TopKTracker,
    collect_top_activations,
)

from .audio_extraction import (
    AudioClipConfig,
    AudioClipExtractor,
    create_indexed_audio_loader,
    create_librispeech_audio_loader,
)

__all__ = [
    # Feature visualization
    "FeatureActivation",
    "FeatureInterpretation",
    "FeatureReport",
    "TopKTracker",
    "collect_top_activations",
    # Audio extraction
    "AudioClipConfig",
    "AudioClipExtractor",
    "create_indexed_audio_loader",
    "create_librispeech_audio_loader",
]
