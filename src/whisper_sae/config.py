"""Configuration system using Pydantic for type safety and validation."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class WhisperConfig(BaseModel):
    """Configuration for Whisper model."""

    model_name: str = Field(
        default="openai/whisper-tiny",
        description="HuggingFace model name for Whisper",
    )
    # Derived from model_name
    hidden_dim: int = Field(default=384, description="Hidden dimension of the model")
    num_encoder_layers: int = Field(default=4, description="Number of encoder layers")
    num_decoder_layers: int = Field(default=4, description="Number of decoder layers")

    @model_validator(mode="after")
    def set_model_dimensions(self) -> "WhisperConfig":
        """Set hidden dimensions based on model name."""
        model_dims = {
            "openai/whisper-tiny": (384, 4, 4),
            "openai/whisper-base": (512, 6, 6),
            "openai/whisper-small": (768, 12, 12),
            "openai/whisper-medium": (1024, 24, 24),
            "openai/whisper-large": (1280, 32, 32),
            "openai/whisper-large-v2": (1280, 32, 32),
            "openai/whisper-large-v3": (1280, 32, 32),
        }
        if self.model_name in model_dims:
            hidden, enc, dec = model_dims[self.model_name]
            self.hidden_dim = hidden
            self.num_encoder_layers = enc
            self.num_decoder_layers = dec
        return self


class SAEConfig(BaseModel):
    """Configuration for Sparse Autoencoder."""

    expansion_factor: int = Field(
        default=8,
        ge=4,
        le=32,
        description="Expansion factor for SAE hidden dimension",
    )
    activation: Literal["topk", "relu", "gelu"] = Field(
        default="topk",
        description="Activation function for SAE",
    )
    k: int = Field(
        default=32,
        ge=1,
        description="Number of active features for TopK activation",
    )
    normalize_decoder: bool = Field(
        default=True,
        description="Whether to normalize decoder columns to unit norm",
    )
    dead_feature_threshold: int = Field(
        default=10_000,
        description="Number of tokens without activation before resampling",
    )
    dead_feature_resample: bool = Field(
        default=True,
        description="Whether to resample dead features",
    )

    def get_hidden_dim(self, input_dim: int) -> int:
        """Calculate SAE hidden dimension from input dimension."""
        return input_dim * self.expansion_factor


class TrainingConfig(BaseModel):
    """Configuration for SAE training."""

    batch_size: int = Field(default=128, ge=1)
    learning_rate: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=0.0, ge=0)
    epochs: int = Field(default=50, ge=1)
    warmup_steps: int = Field(default=1000, ge=0)
    gradient_clip: float = Field(default=1.0, gt=0)
    use_amp: bool = Field(default=True, description="Use automatic mixed precision")
    checkpoint_every: int = Field(default=10, description="Checkpoint every N epochs")
    seed: int = Field(default=42)
    num_workers: int = Field(default=4, ge=0)


class DataConfig(BaseModel):
    """Configuration for data loading."""

    dataset_name: str = Field(default="librispeech_asr")
    dataset_subset: str = Field(default="clean")
    dataset_split: str = Field(default="train.100")
    max_samples: int = Field(default=100_000, ge=1)
    cache_dir: Path = Field(default=Path("cache"))
    streaming: bool = Field(default=True)


class WandbConfig(BaseModel):
    """Configuration for Weights & Biases logging."""

    enabled: bool = Field(default=True)
    project: str = Field(default="whisper-sae")
    entity: str | None = Field(default=None)
    name: str | None = Field(default=None)
    tags: list[str] = Field(default_factory=list)
    log_every: int = Field(default=100, description="Log metrics every N steps")


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration."""

    whisper: WhisperConfig = Field(default_factory=WhisperConfig)
    sae: SAEConfig = Field(default_factory=SAEConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)

    # Layer selection
    encoder_layers: list[int] = Field(
        default_factory=lambda: [0, 1, 2, 3],
        description="Which encoder layers to train SAEs on",
    )
    decoder_layers: list[int] = Field(
        default_factory=lambda: [0, 1, 2, 3],
        description="Which decoder layers to train SAEs on",
    )

    # Output paths
    output_dir: Path = Field(default=Path("outputs"))
    experiment_name: str = Field(default="default")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Load configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        # Convert to dict with mode='json' to serialize Path objects as strings
        data = self.model_dump(mode="json")
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    def get_run_dir(self) -> Path:
        """Get the directory for this experiment run."""
        run_dir = self.output_dir / self.experiment_name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir


# Layer-specific configuration for training separate SAEs
class LayerConfig(BaseModel):
    """Configuration for a specific layer's SAE."""

    component: Literal["encoder", "decoder"] = Field(description="encoder or decoder")
    layer_idx: int = Field(ge=0, description="Layer index")
    input_dim: int = Field(description="Input dimension for this layer")
    sae_config: SAEConfig = Field(default_factory=SAEConfig)
    training_config: TrainingConfig = Field(default_factory=TrainingConfig)

    @property
    def name(self) -> str:
        """Get a unique name for this layer configuration."""
        return f"{self.component}_layer{self.layer_idx}"

    @property
    def hidden_dim(self) -> int:
        """Get the SAE hidden dimension for this layer."""
        return self.sae_config.get_hidden_dim(self.input_dim)
