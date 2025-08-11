# Whisper SAE: Sparse Autoencoder Analysis for Whisper Speech Model

An implementation of Sparse Autoencoders (SAEs) for analyzing and interpreting the internal representations of OpenAI's Whisper speech recognition model. This project explores mechanistic interpretability techniques applied to speech models.

## Overview

This project trains Sparse Autoencoders on the internal activations of Whisper-tiny model to discover interpretable features in speech processing. By applying SAEs to both encoder and decoder representations, we aim to understand how Whisper processes and represents speech information internally.

## Key Features

- **Sparse Autoencoder Training**: Implementation of SAEs with sparsity constraints for feature discovery
- **Multi-layer Analysis**: Separate SAE training for encoder and decoder representations
- **LibriSpeech Integration**: Training on clean speech data from the LibriSpeech dataset
- **Efficient Processing**: Feature caching and mixed-precision training for faster iterations
- **Visualization**: Training loss curves and feature analysis notebooks

## Technical Details

### Model Architecture
- **Base Model**: OpenAI Whisper-tiny (39M parameters)
- **SAE Hidden Dimension**: 1152 features
- **Training Data**: 100k samples from LibriSpeech clean subset
- **Sparsity Constraint**: L1 regularization on activations

### Implementation Highlights
- PyTorch-based implementation with CUDA support
- Mixed-precision training with automatic mixed precision (AMP)
- Efficient data loading with feature caching
- Separate SAEs for encoder (200 epochs) and decoder (50 epochs) representations

## Results

The trained SAEs successfully learn sparse representations of Whisper's internal features:
- **Encoder SAE**: Captures acoustic and phonetic features from speech spectrograms
- **Decoder SAE**: Learns linguistic and semantic representations
- Training converges smoothly with combined reconstruction and sparsity loss

![Training Loss](sae_encoder_training_loss_100k_1152.png)
*Example: Encoder SAE training loss over 200 epochs*

## Project Structure

```
whisper-sae/
├── librispeech_sae_training.py    # Main training script
├── analyze_features.ipynb         # Feature analysis notebook
├── analyze.ipynb                   # Additional analysis
├── encoder_sae_*.pth              # Trained encoder models
├── decoder_sae_*.pth              # Trained decoder models
├── sae_*_training_loss_*.png      # Training visualizations
└── README.md                       # This file
```

## Installation

```bash
# Clone the repository
git clone https://github.com/omarkhursheed/whisper-sae.git
cd whisper-sae

# Create conda environment
conda create -n whisper-sae python=3.10
conda activate whisper-sae

# Install dependencies
pip install torch torchaudio transformers datasets tqdm matplotlib numpy
```

## Usage

### Training SAEs

```python
# Run the main training script
python librispeech_sae_training.py
```

This will:
1. Download and process LibriSpeech data
2. Extract Whisper features (with caching)
3. Train separate SAEs for encoder and decoder
4. Save trained models and loss plots

### Analyzing Features

Use the provided Jupyter notebooks to explore learned features:

```bash
jupyter notebook analyze_features.ipynb
```

### Loading Trained Models

```python
import torch
from librispeech_sae_training import SparseAutoencoder

# Load encoder SAE
encoder_sae = SparseAutoencoder(input_dim=384, hidden_dim=1152)
encoder_sae.load_state_dict(torch.load('encoder_sae_tiny_100k_1152_space.pth'))

# Load decoder SAE  
decoder_sae = SparseAutoencoder(input_dim=384, hidden_dim=1152)
decoder_sae.load_state_dict(torch.load('decoder_sae_tiny_100k_1152_space.pth'))
```

## Mechanistic Interpretability Context

This project contributes to the growing field of mechanistic interpretability by:
- **Feature Discovery**: Identifying monosemantic features in speech model representations
- **Model Understanding**: Revealing how Whisper processes acoustic information
- **Interpretability Tools**: Providing methods to analyze learned speech features
- **Research Foundation**: Establishing baselines for speech model interpretability

## Future Directions

- Scale to larger Whisper models (base, small, medium)
- Implement dictionary learning with more sophisticated sparsity constraints
- Analyze feature activation patterns across different languages
- Investigate causal interventions using discovered features
- Extend to other speech tasks (speaker recognition, emotion detection)

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)
- 16GB+ RAM
- ~10GB disk space for cached features

## Citation

If you use this code in your research, please cite:

```bibtex
@software{whisper_sae_2024,
  author = {Omar Khursheed},
  title = {Whisper SAE: Sparse Autoencoder Analysis for Speech Models},
  year = {2024},
  url = {https://github.com/omarkhursheed/whisper-sae}
}
```

## Acknowledgments

- OpenAI for the Whisper model
- Anthropic for pioneering work on SAEs in language models
- LibriSpeech dataset creators
- The mechanistic interpretability research community

## License

MIT License - see LICENSE file for details

## Contact

For questions or collaborations, please open an issue on GitHub or reach out via the repository.

---

*This project demonstrates practical applications of mechanistic interpretability techniques to speech recognition models, contributing to our understanding of how neural networks process and represent speech.*