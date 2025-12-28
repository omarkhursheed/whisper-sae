"""Tests for configuration system."""

import tempfile
from pathlib import Path

import pytest
import yaml

from whisper_sae.config import (
    ExperimentConfig,
    WhisperConfig,
    SAEConfig,
    TrainingConfig,
    DataConfig,
    WandbConfig,
    LayerConfig,
)


class TestWhisperConfig:
    """Tests for WhisperConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = WhisperConfig()
        assert config.model_name == "openai/whisper-tiny"
        assert config.hidden_dim == 384
        assert config.num_encoder_layers == 4
        assert config.num_decoder_layers == 4

    def test_auto_dimension_setting_tiny(self):
        """Test that dimensions are set automatically for whisper-tiny."""
        config = WhisperConfig(model_name="openai/whisper-tiny")
        assert config.hidden_dim == 384
        assert config.num_encoder_layers == 4

    def test_auto_dimension_setting_base(self):
        """Test that dimensions are set automatically for whisper-base."""
        config = WhisperConfig(model_name="openai/whisper-base")
        assert config.hidden_dim == 512
        assert config.num_encoder_layers == 6

    def test_auto_dimension_setting_small(self):
        """Test that dimensions are set automatically for whisper-small."""
        config = WhisperConfig(model_name="openai/whisper-small")
        assert config.hidden_dim == 768
        assert config.num_encoder_layers == 12

    def test_auto_dimension_setting_medium(self):
        """Test that dimensions are set automatically for whisper-medium."""
        config = WhisperConfig(model_name="openai/whisper-medium")
        assert config.hidden_dim == 1024
        assert config.num_encoder_layers == 24

    def test_auto_dimension_setting_large(self):
        """Test that dimensions are set automatically for whisper-large."""
        config = WhisperConfig(model_name="openai/whisper-large")
        assert config.hidden_dim == 1280
        assert config.num_encoder_layers == 32

    def test_unknown_model_keeps_defaults(self):
        """Test that unknown model name keeps default dimensions."""
        config = WhisperConfig(model_name="some/unknown-model")
        # Should keep whatever defaults were set
        assert config.hidden_dim == 384  # default


class TestSAEConfig:
    """Tests for SAEConfig."""

    def test_default_values(self):
        """Test default SAE configuration."""
        config = SAEConfig()
        assert config.expansion_factor == 8
        assert config.activation == "topk"
        assert config.k == 32
        assert config.normalize_decoder is True
        assert config.dead_feature_threshold == 10_000

    def test_get_hidden_dim(self):
        """Test hidden dimension calculation."""
        config = SAEConfig(expansion_factor=8)
        assert config.get_hidden_dim(384) == 3072
        assert config.get_hidden_dim(512) == 4096
        assert config.get_hidden_dim(768) == 6144

    def test_expansion_factor_validation(self):
        """Test that expansion factor is validated."""
        with pytest.raises(ValueError):
            SAEConfig(expansion_factor=2)  # Below minimum of 4

        with pytest.raises(ValueError):
            SAEConfig(expansion_factor=64)  # Above maximum of 32

    def test_k_validation(self):
        """Test that k is validated."""
        with pytest.raises(ValueError):
            SAEConfig(k=0)  # Must be at least 1

    def test_activation_validation(self):
        """Test that activation type is validated."""
        # Valid activations
        SAEConfig(activation="topk")
        SAEConfig(activation="relu")
        SAEConfig(activation="gelu")

        # Invalid activation - should raise validation error
        with pytest.raises(ValueError):
            SAEConfig(activation="invalid")


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_values(self):
        """Test default training configuration."""
        config = TrainingConfig()
        assert config.batch_size == 128
        assert config.learning_rate == 1e-4
        assert config.epochs == 50
        assert config.use_amp is True
        assert config.seed == 42

    def test_validation(self):
        """Test that training parameters are validated."""
        with pytest.raises(ValueError):
            TrainingConfig(batch_size=0)

        with pytest.raises(ValueError):
            TrainingConfig(learning_rate=-0.001)

        with pytest.raises(ValueError):
            TrainingConfig(epochs=0)


class TestDataConfig:
    """Tests for DataConfig."""

    def test_default_values(self):
        """Test default data configuration."""
        config = DataConfig()
        assert config.dataset_name == "librispeech_asr"
        assert config.max_samples == 100_000
        assert config.streaming is True

    def test_cache_dir_as_path(self):
        """Test that cache_dir is converted to Path."""
        config = DataConfig(cache_dir="my/cache/dir")
        assert isinstance(config.cache_dir, Path)
        assert config.cache_dir == Path("my/cache/dir")


class TestWandbConfig:
    """Tests for WandbConfig."""

    def test_default_values(self):
        """Test default wandb configuration."""
        config = WandbConfig()
        assert config.enabled is True
        assert config.project == "whisper-sae"
        assert config.entity is None
        assert config.tags == []

    def test_tags_as_list(self):
        """Test that tags are stored as list."""
        config = WandbConfig(tags=["tag1", "tag2"])
        assert config.tags == ["tag1", "tag2"]


class TestExperimentConfig:
    """Tests for ExperimentConfig."""

    def test_default_nested_configs(self):
        """Test that nested configs have defaults."""
        config = ExperimentConfig()
        assert isinstance(config.whisper, WhisperConfig)
        assert isinstance(config.sae, SAEConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.wandb, WandbConfig)

    def test_encoder_decoder_layers(self):
        """Test default layer selection."""
        config = ExperimentConfig()
        assert config.encoder_layers == [0, 1, 2, 3]
        assert config.decoder_layers == [0, 1, 2, 3]

    def test_yaml_roundtrip(self):
        """Test that config can be saved and loaded from YAML."""
        original = ExperimentConfig(
            whisper=WhisperConfig(model_name="openai/whisper-base"),
            sae=SAEConfig(expansion_factor=16, k=64),
            training=TrainingConfig(batch_size=256),
            experiment_name="test_experiment",
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            original.to_yaml(f.name)
            loaded = ExperimentConfig.from_yaml(f.name)

        assert loaded.whisper.model_name == "openai/whisper-base"
        assert loaded.sae.expansion_factor == 16
        assert loaded.sae.k == 64
        assert loaded.training.batch_size == 256
        assert loaded.experiment_name == "test_experiment"

    def test_from_yaml_file(self):
        """Test loading from YAML file."""
        yaml_content = """
whisper:
  model_name: "openai/whisper-small"
sae:
  expansion_factor: 8
  k: 32
training:
  batch_size: 64
  epochs: 100
encoder_layers: [0, 2, 4]
decoder_layers: [1, 3, 5]
experiment_name: "yaml_test"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            config = ExperimentConfig.from_yaml(f.name)

        assert config.whisper.model_name == "openai/whisper-small"
        assert config.sae.expansion_factor == 8
        assert config.training.epochs == 100
        assert config.encoder_layers == [0, 2, 4]
        assert config.experiment_name == "yaml_test"

    def test_get_run_dir_creates_directory(self):
        """Test that get_run_dir creates the directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExperimentConfig(
                output_dir=Path(tmpdir),
                experiment_name="test_run",
            )
            run_dir = config.get_run_dir()

            assert run_dir.exists()
            assert run_dir == Path(tmpdir) / "test_run"


class TestLayerConfig:
    """Tests for LayerConfig."""

    def test_layer_config_creation(self):
        """Test creating a layer configuration."""
        config = LayerConfig(
            component="encoder",
            layer_idx=2,
            input_dim=384,
        )

        assert config.component == "encoder"
        assert config.layer_idx == 2
        assert config.input_dim == 384

    def test_layer_config_name(self):
        """Test layer config name generation."""
        encoder_config = LayerConfig(component="encoder", layer_idx=0, input_dim=384)
        decoder_config = LayerConfig(component="decoder", layer_idx=3, input_dim=384)

        assert encoder_config.name == "encoder_layer0"
        assert decoder_config.name == "decoder_layer3"

    def test_layer_config_hidden_dim(self):
        """Test layer config hidden dimension calculation."""
        config = LayerConfig(
            component="encoder",
            layer_idx=0,
            input_dim=384,
            sae_config=SAEConfig(expansion_factor=8),
        )

        assert config.hidden_dim == 384 * 8

    def test_layer_config_with_custom_sae(self):
        """Test layer config with custom SAE configuration."""
        sae_config = SAEConfig(expansion_factor=16, k=64)
        config = LayerConfig(
            component="decoder",
            layer_idx=1,
            input_dim=512,
            sae_config=sae_config,
        )

        assert config.hidden_dim == 512 * 16
        assert config.sae_config.k == 64


class TestConfigIntegration:
    """Integration tests for configuration system."""

    def test_full_config_creation(self):
        """Test creating a complete configuration."""
        config = ExperimentConfig(
            whisper=WhisperConfig(model_name="openai/whisper-base"),
            sae=SAEConfig(
                expansion_factor=8,
                activation="topk",
                k=32,
            ),
            training=TrainingConfig(
                batch_size=128,
                learning_rate=1e-4,
                epochs=50,
            ),
            data=DataConfig(
                dataset_name="librispeech_asr",
                max_samples=100_000,
            ),
            wandb=WandbConfig(
                enabled=True,
                project="whisper-sae",
                tags=["test", "whisper-base"],
            ),
            encoder_layers=[0, 1, 2, 3, 4, 5],
            decoder_layers=[0, 1, 2, 3, 4, 5],
            experiment_name="full_config_test",
        )

        assert config.whisper.hidden_dim == 512  # whisper-base dimension
        assert config.sae.get_hidden_dim(512) == 4096

    def test_config_modification(self):
        """Test modifying configuration after creation."""
        config = ExperimentConfig()

        # Modify nested config
        config.training.batch_size = 256
        config.sae.k = 64

        assert config.training.batch_size == 256
        assert config.sae.k == 64

    def test_config_dict_conversion(self):
        """Test converting config to/from dict."""
        config = ExperimentConfig(
            experiment_name="dict_test",
            encoder_layers=[0, 1],
        )

        config_dict = config.model_dump()

        assert isinstance(config_dict, dict)
        assert config_dict["experiment_name"] == "dict_test"
        assert config_dict["encoder_layers"] == [0, 1]

        # Recreate from dict
        recreated = ExperimentConfig(**config_dict)
        assert recreated.experiment_name == "dict_test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
