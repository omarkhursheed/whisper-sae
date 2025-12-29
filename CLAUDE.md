# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

Whisper SAE is a mechanistic interpretability research project comparing SAEs, Transcoders, and Crosscoders on OpenAI Whisper. Goal: publication-ready research on sparse coding for speech models.

## Commands

```bash
# Setup
uv sync

# Run tests (110 tests)
uv run pytest tests/ -v

# Local training (small test)
uv run python scripts/train.py --config configs/tiny_test.yaml --no-wandb

# Full training on Modal
modal run modal_app/extract_features.py  # Extract features first
modal run modal_app/train.py --all-layers  # Train SAEs

# Analysis notebook
uv run jupyter notebook notebooks/logit_lens_and_attention.ipynb
```

## Architecture

### SAE Models (`src/whisper_sae/sae/`)
- `model.py`: TopKSAE (k active features), ReLUSAE (L1 regularization)
- `training.py`: SAETrainer with checkpointing, W&B logging, dead feature resampling
- `hooks.py`: WhisperActivationExtractor for layer-wise feature extraction

### Data Pipeline (`src/whisper_sae/data/`)
- `librispeech.py`: LibriSpeech loading with soundfile (not torchcodec - macOS compatible)
- `feature_cache.py`: FeatureCache for caching extracted activations

### Modal Apps (`modal_app/`)
- `extract_features.py`: GPU feature extraction on A10G
- `train.py`: GPU SAE training with W&B integration

## Caching Strategy (IMPORTANT)

**Always cache intermediate outputs to avoid redundant computation:**

### 1. Raw Audio Features
```
cache/librispeech_{subset}_{split}_{max_samples}.pt  # Mel spectrograms
cache/librispeech_{subset}_{split}_{max_samples}_meta.pt  # Metadata
```

### 2. Extracted Activations (per layer)
```
cache/features/{model}/encoder_layer{N}.pt  # [num_tokens, hidden_dim]
cache/features/{model}/decoder_layer{N}.pt
cache/features/{model}/metadata.json
```

### 3. Trained Models
```
outputs/{experiment_name}/
  checkpoint_epoch{N}.pt  # Periodic checkpoints
  sae_final.pt            # Final model
  metrics.json            # Training metrics
```

### 4. Analysis Results
```
outputs/{experiment_name}/
  top_activations.pt      # Top-k examples per feature
  feature_reports/        # Per-feature interpretation
  ablation_results.json   # Causal validation results
```

### Best Practices
- **Check cache first**: Always check if data exists before regenerating
- **Atomic saves**: Use temp files + rename for crash safety
- **Version in filename**: Include model name, layer, config hash
- **Save frequently**: Checkpoint every 10 epochs, save metrics every epoch
- **Commit volumes**: On Modal, always `volume.commit()` after saves

## Key Files

| File | Purpose |
|------|---------|
| `scripts/train.py` | Main training CLI |
| `configs/tiny_default.yaml` | Default config for Whisper-tiny |
| `configs/tiny_test.yaml` | Quick local testing config |
| `src/whisper_sae/config.py` | Pydantic configuration |
| `src/whisper_sae/sae/model.py` | SAE architectures |
| `src/whisper_sae/sae/training.py` | Training loop |

## Experiment Phases

1. **Phase 1**: SAE baseline on Whisper-tiny encoder/decoder (~$50)
2. **Phase 2**: Transcoders for MLP interpretation (~$30)
3. **Phase 3**: Crosscoders for cross-layer features (~$30)
4. **Phase 4**: Comparative analysis and ablations (~$20)
5. **Phase 5**: Scale to Whisper-base if results promising (~$200)

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_training.py -v

# Run with coverage
uv run pytest tests/ --cov=whisper_sae
```

## Common Issues

### Audio Loading on macOS
Use soundfile instead of torchcodec:
```python
import soundfile as sf
audio, sr = sf.read(io.BytesIO(audio_bytes))
```

### MPS (Apple Silicon)
- AMP/GradScaler not supported on MPS
- Use `device="cpu"` for debugging, MPS for faster iteration

### Dead Features
High dead ratio (>90%) is normal for short training. Enable resampling:
```python
sae.resample_dead_features(inputs, num_resample=100)
```
