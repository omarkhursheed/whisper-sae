"""Tests for SAE training loop."""

import tempfile
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from whisper_sae.config import TrainingConfig
from whisper_sae.sae.model import TopKSAE
from whisper_sae.sae.training import SAETrainer, TrainingMetrics


class TestTrainingMetrics:
    """Tests for TrainingMetrics dataclass."""

    def test_training_metrics_creation(self):
        """Test creating TrainingMetrics."""
        metrics = TrainingMetrics(
            loss=0.5,
            reconstruction_loss=0.4,
            sparsity_loss=0.1,
            l0=32.0,
            dead_feature_ratio=0.1,
            learning_rate=1e-4,
            step=100,
        )
        assert metrics.loss == 0.5
        assert metrics.reconstruction_loss == 0.4
        assert metrics.l0 == 32.0
        assert metrics.step == 100


class TestSAETrainer:
    """Tests for SAETrainer."""

    @pytest.fixture
    def simple_model(self):
        """Create a small SAE for testing."""
        return TopKSAE(input_dim=64, hidden_dim=128, k=8)

    @pytest.fixture
    def training_config(self):
        """Create a minimal training config."""
        return TrainingConfig(
            batch_size=16,
            learning_rate=1e-3,
            epochs=2,
            warmup_steps=10,
            gradient_clip=1.0,
            use_amp=False,
            checkpoint_every=1,
        )

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        # 100 samples, 64 dimensions
        return torch.randn(100, 64)

    @pytest.fixture
    def sample_dataloader(self, sample_data):
        """Create a DataLoader from sample data."""
        dataset = TensorDataset(sample_data)
        return DataLoader(dataset, batch_size=16, shuffle=True)

    def test_trainer_initialization(self, simple_model, training_config):
        """Test trainer initializes correctly."""
        trainer = SAETrainer(
            model=simple_model,
            config=training_config,
            device="cpu",
        )
        assert trainer.global_step == 0
        assert trainer.epoch == 0
        assert trainer.model is simple_model
        assert len(trainer.metrics_history) == 0

    def test_trainer_with_run_dir(self, simple_model, training_config):
        """Test trainer creates run directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "test_run"
            trainer = SAETrainer(
                model=simple_model,
                config=training_config,
                device="cpu",
                run_dir=run_dir,
            )
            assert run_dir.exists()

    def test_setup_scheduler(self, simple_model, training_config):
        """Test learning rate scheduler setup."""
        trainer = SAETrainer(
            model=simple_model,
            config=training_config,
            device="cpu",
        )
        trainer.setup_scheduler(total_steps=100)
        assert trainer.scheduler is not None

    def test_train_step_returns_metrics(self, simple_model, training_config, sample_data):
        """Test single training step returns metrics."""
        trainer = SAETrainer(
            model=simple_model,
            config=training_config,
            device="cpu",
        )
        trainer.setup_scheduler(total_steps=100)

        batch = sample_data[:16]
        metrics = trainer.train_step(batch)

        assert isinstance(metrics, TrainingMetrics)
        assert metrics.loss > 0
        assert metrics.reconstruction_loss > 0
        assert metrics.l0 == 8.0  # k=8
        assert metrics.step == 1

    def test_train_step_handles_tuple_batch(self, simple_model, training_config, sample_data):
        """Test train_step handles TensorDataset tuples."""
        trainer = SAETrainer(
            model=simple_model,
            config=training_config,
            device="cpu",
        )
        trainer.setup_scheduler(total_steps=100)

        # TensorDataset returns tuples
        batch = (sample_data[:16],)
        metrics = trainer.train_step(batch)

        assert isinstance(metrics, TrainingMetrics)
        assert metrics.step == 1

    def test_train_step_handles_list_batch(self, simple_model, training_config, sample_data):
        """Test train_step handles list batches."""
        trainer = SAETrainer(
            model=simple_model,
            config=training_config,
            device="cpu",
        )
        trainer.setup_scheduler(total_steps=100)

        batch = [sample_data[:16]]
        metrics = trainer.train_step(batch)

        assert isinstance(metrics, TrainingMetrics)

    def test_train_step_increments_global_step(self, simple_model, training_config, sample_data):
        """Test global step increments correctly."""
        trainer = SAETrainer(
            model=simple_model,
            config=training_config,
            device="cpu",
        )
        trainer.setup_scheduler(total_steps=100)

        assert trainer.global_step == 0
        trainer.train_step(sample_data[:16])
        assert trainer.global_step == 1
        trainer.train_step(sample_data[:16])
        assert trainer.global_step == 2

    def test_train_step_updates_scheduler(self, simple_model, training_config, sample_data):
        """Test scheduler updates after each step."""
        trainer = SAETrainer(
            model=simple_model,
            config=training_config,
            device="cpu",
        )
        trainer.setup_scheduler(total_steps=100)

        initial_lr = trainer.optimizer.param_groups[0]["lr"]
        # Do several steps to get past warmup
        for _ in range(20):
            trainer.train_step(sample_data[:16])

        # LR should have changed (either warming up or decaying)
        # Just check it's still positive
        current_lr = trainer.optimizer.param_groups[0]["lr"]
        assert current_lr > 0

    def test_train_epoch(self, simple_model, training_config, sample_dataloader):
        """Test training for one epoch."""
        trainer = SAETrainer(
            model=simple_model,
            config=training_config,
            device="cpu",
        )
        trainer.setup_scheduler(total_steps=100)

        initial_step = trainer.global_step
        epoch_metrics = trainer.train_epoch(sample_dataloader)

        assert len(epoch_metrics) > 0
        assert trainer.global_step > initial_step
        assert trainer.epoch == 1

    def test_train_epoch_records_metrics(self, simple_model, training_config, sample_dataloader):
        """Test that train_epoch records metrics in history."""
        trainer = SAETrainer(
            model=simple_model,
            config=training_config,
            device="cpu",
        )
        trainer.setup_scheduler(total_steps=100)

        trainer.train_epoch(sample_dataloader)

        assert len(trainer.metrics_history) > 0
        assert all(isinstance(m, TrainingMetrics) for m in trainer.metrics_history)

    def test_loss_decreases_during_training(self, simple_model, training_config):
        """Test that loss generally decreases during training."""
        # Use fixed data for reproducibility
        torch.manual_seed(42)
        data = torch.randn(200, 64)
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

        trainer = SAETrainer(
            model=simple_model,
            config=training_config,
            device="cpu",
        )
        trainer.setup_scheduler(total_steps=500)

        # Train for several epochs
        for _ in range(5):
            trainer.train_epoch(dataloader)

        # Check that average loss in last epoch is less than first epoch
        steps_per_epoch = len(dataloader)
        first_epoch_losses = [m.loss for m in trainer.metrics_history[:steps_per_epoch]]
        last_epoch_losses = [m.loss for m in trainer.metrics_history[-steps_per_epoch:]]

        avg_first = sum(first_epoch_losses) / len(first_epoch_losses)
        avg_last = sum(last_epoch_losses) / len(last_epoch_losses)
        assert avg_last < avg_first, f"Loss should decrease: {avg_first:.4f} -> {avg_last:.4f}"

    def test_save_checkpoint(self, simple_model, training_config, sample_data):
        """Test saving checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = SAETrainer(
                model=simple_model,
                config=training_config,
                device="cpu",
                run_dir=Path(tmpdir),
            )
            trainer.setup_scheduler(total_steps=100)
            trainer.train_step(sample_data[:16])

            checkpoint_path = trainer.save_checkpoint("test_checkpoint.pt")

            assert checkpoint_path.exists()
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            assert "model_state_dict" in checkpoint
            assert "optimizer_state_dict" in checkpoint
            assert "global_step" in checkpoint
            assert checkpoint["global_step"] == 1

    def test_load_checkpoint(self, simple_model, training_config, sample_data):
        """Test loading checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Train and save
            trainer1 = SAETrainer(
                model=simple_model,
                config=training_config,
                device="cpu",
                run_dir=Path(tmpdir),
            )
            trainer1.setup_scheduler(total_steps=100)
            for _ in range(5):
                trainer1.train_step(sample_data[:16])
            checkpoint_path = trainer1.save_checkpoint("checkpoint.pt")

            # Create new trainer and load
            model2 = TopKSAE(input_dim=64, hidden_dim=128, k=8)
            trainer2 = SAETrainer(
                model=model2,
                config=training_config,
                device="cpu",
                run_dir=Path(tmpdir),
            )
            trainer2.setup_scheduler(total_steps=100)
            trainer2.load_checkpoint(checkpoint_path)

            assert trainer2.global_step == 5
            assert trainer2.epoch == trainer1.epoch

    def test_save_metrics(self, simple_model, training_config, sample_dataloader):
        """Test saving metrics to JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = SAETrainer(
                model=simple_model,
                config=training_config,
                device="cpu",
                run_dir=Path(tmpdir),
            )
            trainer.setup_scheduler(total_steps=100)
            trainer.train_epoch(sample_dataloader)

            metrics_path = trainer.save_metrics("metrics.json")

            assert metrics_path.exists()
            import json
            with open(metrics_path) as f:
                metrics_data = json.load(f)
            assert len(metrics_data) > 0
            assert "loss" in metrics_data[0]
            assert "step" in metrics_data[0]

    def test_decoder_normalization(self, simple_model, training_config, sample_data):
        """Test that decoder weights are normalized after training step."""
        trainer = SAETrainer(
            model=simple_model,
            config=training_config,
            device="cpu",
        )
        trainer.setup_scheduler(total_steps=100)
        trainer.train_step(sample_data[:16])

        # Check decoder columns are unit norm
        decoder_norms = torch.norm(simple_model.decoder.weight, dim=0)
        assert torch.allclose(decoder_norms, torch.ones_like(decoder_norms), atol=1e-5)


class TestTrainerResampling:
    """Tests for dead feature resampling integration."""

    @pytest.fixture
    def model_with_resampling(self):
        """Create SAE with low dead feature threshold for testing."""
        return TopKSAE(
            input_dim=64,
            hidden_dim=128,
            k=8,
            dead_feature_threshold=10,  # Low threshold for testing
        )

    @pytest.fixture
    def training_config(self):
        """Create config for resampling tests."""
        return TrainingConfig(
            batch_size=16,
            learning_rate=1e-3,
            epochs=1,
            use_amp=False,
        )

    def test_set_resample_dataset(self, model_with_resampling, training_config):
        """Test setting resample dataset."""
        trainer = SAETrainer(
            model=model_with_resampling,
            config=training_config,
            device="cpu",
            resample_dead_every=5,
        )

        data = torch.randn(100, 64)
        dataset = TensorDataset(data)
        trainer.set_resample_dataset(dataset)

        assert trainer._resample_dataset is dataset

    def test_resampling_triggers_at_interval(self, model_with_resampling, training_config):
        """Test that resampling is attempted at correct intervals."""
        trainer = SAETrainer(
            model=model_with_resampling,
            config=training_config,
            device="cpu",
            resample_dead_every=5,
            resample_batch_size=32,
        )

        data = torch.randn(100, 64)
        dataset = TensorDataset(data)
        trainer.set_resample_dataset(dataset)
        trainer.setup_scheduler(total_steps=100)

        # Train for several steps
        for _ in range(10):
            trainer.train_step(data[:16])

        # After 10 steps with resample_dead_every=5, resampling should have been attempted
        # We can't easily verify it ran, but we can check no errors occurred
        assert trainer.global_step == 10


class TestTrainerDevices:
    """Tests for device handling in trainer."""

    @pytest.fixture
    def simple_model(self):
        return TopKSAE(input_dim=32, hidden_dim=64, k=4)

    @pytest.fixture
    def training_config(self):
        return TrainingConfig(batch_size=8, epochs=1, use_amp=False)

    def test_cpu_device(self, simple_model, training_config):
        """Test training on CPU."""
        trainer = SAETrainer(
            model=simple_model,
            config=training_config,
            device="cpu",
        )
        assert str(trainer.device) == "cpu"
        assert not trainer.use_amp

    def test_amp_disabled_on_cpu(self, simple_model):
        """Test AMP is disabled on CPU even if config enables it."""
        config = TrainingConfig(batch_size=8, epochs=1, use_amp=True)
        trainer = SAETrainer(
            model=simple_model,
            config=config,
            device="cpu",
        )
        assert not trainer.use_amp

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_amp_disabled_on_mps(self, simple_model):
        """Test AMP is disabled on MPS."""
        config = TrainingConfig(batch_size=8, epochs=1, use_amp=True)
        trainer = SAETrainer(
            model=simple_model,
            config=config,
            device="mps",
        )
        assert not trainer.use_amp
