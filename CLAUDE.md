# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Whisper SAE is a mechanistic interpretability research project that trains Sparse Autoencoders on OpenAI Whisper-tiny's internal activations to discover interpretable features in speech processing. Trains separate SAEs for encoder (acoustic/phonetic features) and decoder (linguistic/semantic) representations.

## Commands

```bash
# Setup environment
conda create -n whisper-sae python=3.10
conda activate whisper-sae
pip install -r requirements.txt

# Train SAEs (full pipeline: ~6-7 hours GPU, uses cached features if available)
python librispeech_sae_training.py

# Analyze features
jupyter notebook analyze_features.ipynb
```

## Architecture

### Core Training Script: `librispeech_sae_training.py`

**Key Classes:**
- `LibriSpeechDataset`: Streams LibriSpeech, handles 16kHz resampling, extracts Whisper features, caches to `librispeech_features_cache_tiny.pt`
- `SparseAutoencoder`: Input(384) -> Linear -> ReLU -> Hidden(1152) -> Linear -> Output(384). Loss = MSE + 0.1 * L1(hidden)

**Training Configuration:**
- Encoder SAE: 200 epochs, batch size 64
- Decoder SAE: 50 epochs, batch size 64
- Mixed-precision training (AMP) enabled for CUDA
- Feature extraction uses 128 batch size, 4 workers

**Feature Extraction:**
- Uses `whisper_model.model.encoder()` for encoder features (384-dim)
- Uses `whisper_model.model.decoder()` for decoder features with encoder context

### Data Pipeline

1. Streams from HuggingFace LibriSpeech ("clean", "train.100", 100k samples)
2. Caches extracted features to `librispeech_features_cache_tiny.pt` (~138MB)
3. Subsequent runs skip extraction and load from cache

### Trained Models (in repo)

- `encoder_sae_tiny_100k_1152_space.pth`
- `decoder_sae_tiny_100k_1152_space.pth`

## Key Files

- `librispeech_sae_training.py` - Main training implementation
- `analyze_features.ipynb` - Feature interpretation and visualization
- `analyze.ipynb` - Additional analysis workflows
