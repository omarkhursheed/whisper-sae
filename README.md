# Whisper SAE: Sparse Autoencoder Analysis for Whisper

Mechanistic interpretability for OpenAI's Whisper speech model using Sparse Autoencoders.

## Overview

Train SAEs on Whisper's internal activations to discover interpretable features in speech processing. Analyze both encoder (acoustic/phonetic) and decoder (linguistic/semantic) representations.

## Quick Start

```bash
# Setup
uv sync

# Run tests
uv run pytest tests/ -v

# Analysis notebook
uv run jupyter notebook notebooks/logit_lens_and_attention.ipynb
```

## Architecture

- **TopK SAE**: Sparse autoencoder with TopK activation (more stable than L1 for speech)
- **8x expansion**: 384 -> 3072 features for whisper-tiny
- **Unit-norm decoder**: Normalized columns for training stability
- **Dead feature resampling**: Reinitialize unused features

## Project Structure

```
whisper_analysis/
├── src/whisper_sae/
│   ├── config.py           # Pydantic configuration
│   ├── sae/
│   │   ├── hooks.py        # Whisper activation extraction
│   │   ├── model.py        # TopKSAE, ReLUSAE
│   │   └── training.py     # Training loop with AMP
│   └── data/
│       ├── librispeech.py  # Dataset loading
│       └── feature_cache.py # Activation caching
├── tests/                   # 89 tests
├── notebooks/               # Analysis notebooks
├── configs/                 # YAML configs
└── docs/                    # Documentation
```

## Features

- **Activation Hooks**: Extract from any encoder/decoder layer with layer norm
- **CircuitsVis Integration**: Interactive attention visualization
- **Encoder Lens**: Project encoder layers through decoder to see when transcription emerges
- **Activation Patching**: Measure causal importance of layers

## Based On

- aiOla "Beyond Transcription" (2025) - Encoder lens technique
- Mozilla Builders (2024) - TopK SAEs for Whisper
- TransformerLens/SAELens conventions

## Requirements

- Python 3.11+
- PyTorch 2.1+
- MPS (Apple Silicon) or CUDA

## License

MIT
