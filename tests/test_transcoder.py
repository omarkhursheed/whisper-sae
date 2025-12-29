"""Tests for Transcoder models."""

import pytest
import torch

from whisper_sae.sae.transcoder import (
    TopKTranscoder,
    SkipTranscoder,
    TranscoderOutput,
    create_transcoder,
)


class TestTopKTranscoder:
    """Tests for TopKTranscoder."""

    def test_initialization(self):
        """Test transcoder initializes with correct dimensions."""
        transcoder = TopKTranscoder(
            input_dim=64,
            output_dim=64,
            hidden_dim=128,
            k=8,
        )
        assert transcoder.input_dim == 64
        assert transcoder.output_dim == 64
        assert transcoder.hidden_dim == 128
        assert transcoder.k == 8

    def test_encoder_decoder_shapes(self):
        """Test encoder and decoder have correct weight shapes."""
        transcoder = TopKTranscoder(
            input_dim=64,
            output_dim=64,
            hidden_dim=128,
            k=8,
        )
        assert transcoder.encoder.weight.shape == (128, 64)
        assert transcoder.decoder.weight.shape == (64, 128)

    def test_different_input_output_dims(self):
        """Test transcoder works with different input and output dimensions."""
        transcoder = TopKTranscoder(
            input_dim=64,
            output_dim=32,
            hidden_dim=128,
            k=8,
        )
        assert transcoder.encoder.weight.shape == (128, 64)
        assert transcoder.decoder.weight.shape == (32, 128)

    def test_encode_output_shape(self):
        """Test encode produces correct output shape."""
        transcoder = TopKTranscoder(
            input_dim=64,
            output_dim=64,
            hidden_dim=128,
            k=8,
        )
        x = torch.randn(16, 64)
        hidden = transcoder.encode(x)
        assert hidden.shape == (16, 128)

    def test_topk_sparsity(self):
        """Test that exactly k features are active per sample."""
        transcoder = TopKTranscoder(
            input_dim=64,
            output_dim=64,
            hidden_dim=128,
            k=8,
        )
        x = torch.randn(16, 64)
        hidden = transcoder.encode(x)

        # Count non-zero features per sample
        active_per_sample = (hidden > 0).sum(dim=-1)
        assert torch.all(active_per_sample == 8)

    def test_decode_output_shape(self):
        """Test decode produces correct output shape."""
        transcoder = TopKTranscoder(
            input_dim=64,
            output_dim=64,
            hidden_dim=128,
            k=8,
        )
        hidden = torch.randn(16, 128)
        decoded = transcoder.decode(hidden)
        assert decoded.shape == (16, 64)

    def test_forward_returns_transcoder_output(self):
        """Test forward returns TranscoderOutput."""
        transcoder = TopKTranscoder(
            input_dim=64,
            output_dim=64,
            hidden_dim=128,
            k=8,
        )
        mlp_input = torch.randn(16, 64)
        mlp_output = torch.randn(16, 64)

        result = transcoder(mlp_input, mlp_output)
        assert isinstance(result, TranscoderOutput)

    def test_forward_output_shapes(self):
        """Test all outputs have correct shapes."""
        transcoder = TopKTranscoder(
            input_dim=64,
            output_dim=64,
            hidden_dim=128,
            k=8,
        )
        mlp_input = torch.randn(16, 64)
        mlp_output = torch.randn(16, 64)

        result = transcoder(mlp_input, mlp_output)

        assert result.predicted.shape == (16, 64)
        assert result.hidden.shape == (16, 128)
        assert result.loss.ndim == 0  # Scalar
        assert result.reconstruction_loss.ndim == 0
        assert result.l0.ndim == 0

    def test_loss_is_mse(self):
        """Test that loss is MSE between predicted and target."""
        transcoder = TopKTranscoder(
            input_dim=64,
            output_dim=64,
            hidden_dim=128,
            k=8,
        )
        mlp_input = torch.randn(16, 64)
        mlp_output = torch.randn(16, 64)

        result = transcoder(mlp_input, mlp_output)

        expected_mse = torch.nn.functional.mse_loss(result.predicted, mlp_output)
        assert torch.isclose(result.reconstruction_loss, expected_mse)

    def test_l0_equals_k(self):
        """Test L0 equals k for TopK activation."""
        transcoder = TopKTranscoder(
            input_dim=64,
            output_dim=64,
            hidden_dim=128,
            k=8,
        )
        mlp_input = torch.randn(16, 64)
        mlp_output = torch.randn(16, 64)

        result = transcoder(mlp_input, mlp_output)
        assert result.l0.item() == 8.0

    def test_dead_feature_tracking(self):
        """Test dead feature tracking updates during training."""
        transcoder = TopKTranscoder(
            input_dim=64,
            output_dim=64,
            hidden_dim=128,
            k=8,
            dead_feature_threshold=10,
        )
        transcoder.train()

        assert transcoder.step_count == 0

        mlp_input = torch.randn(16, 64)
        mlp_output = torch.randn(16, 64)
        transcoder(mlp_input, mlp_output)

        assert transcoder.step_count == 1

    def test_dead_feature_ratio_initially_one(self):
        """Test all features are dead initially (before threshold steps)."""
        transcoder = TopKTranscoder(
            input_dim=64,
            output_dim=64,
            hidden_dim=128,
            k=8,
            dead_feature_threshold=100,
        )

        # After many steps without activation, features are dead
        transcoder.step_count = torch.tensor(1000, dtype=torch.long)
        dead_ratio = transcoder.get_dead_feature_ratio()
        assert dead_ratio == 1.0

    def test_gradients_flow(self):
        """Test gradients flow through the transcoder."""
        transcoder = TopKTranscoder(
            input_dim=64,
            output_dim=64,
            hidden_dim=128,
            k=8,
        )
        mlp_input = torch.randn(16, 64, requires_grad=True)
        mlp_output = torch.randn(16, 64)

        result = transcoder(mlp_input, mlp_output)
        result.loss.backward()

        assert mlp_input.grad is not None
        assert transcoder.encoder.weight.grad is not None
        assert transcoder.decoder.weight.grad is not None


class TestSkipTranscoder:
    """Tests for SkipTranscoder."""

    def test_initialization(self):
        """Test skip transcoder initializes correctly."""
        transcoder = SkipTranscoder(
            input_dim=64,
            output_dim=64,
            hidden_dim=128,
            k=8,
        )
        assert transcoder.input_dim == 64
        assert transcoder.output_dim == 64
        assert transcoder.hidden_dim == 128
        assert hasattr(transcoder, 'skip')

    def test_skip_connection_exists(self):
        """Test skip connection layer exists with correct shape."""
        transcoder = SkipTranscoder(
            input_dim=64,
            output_dim=64,
            hidden_dim=128,
            k=8,
        )
        assert transcoder.skip.weight.shape == (64, 64)

    def test_paper_initialization_zeros(self):
        """Test initialization follows paper: decoder and skip start at zero."""
        transcoder = SkipTranscoder(
            input_dim=64,
            output_dim=64,
            hidden_dim=128,
            k=8,
        )
        # Per Paulo et al. (2025): W₂ and W_skip start at zero
        assert torch.allclose(transcoder.decoder.weight, torch.zeros_like(transcoder.decoder.weight))
        assert torch.allclose(transcoder.skip.weight, torch.zeros_like(transcoder.skip.weight))

    def test_set_output_bias(self):
        """Test setting output bias to mean MLP output."""
        transcoder = SkipTranscoder(
            input_dim=64,
            output_dim=64,
            hidden_dim=128,
            k=8,
        )
        mean_output = torch.randn(64)
        transcoder.set_output_bias(mean_output)

        assert torch.allclose(transcoder.decoder.bias, mean_output)

    def test_forward_returns_transcoder_output(self):
        """Test forward returns TranscoderOutput."""
        transcoder = SkipTranscoder(
            input_dim=64,
            output_dim=64,
            hidden_dim=128,
            k=8,
        )
        mlp_input = torch.randn(16, 64)
        mlp_output = torch.randn(16, 64)

        result = transcoder(mlp_input, mlp_output)
        assert isinstance(result, TranscoderOutput)

    def test_forward_includes_skip(self):
        """Test forward pass includes skip connection contribution."""
        transcoder = SkipTranscoder(
            input_dim=64,
            output_dim=64,
            hidden_dim=128,
            k=8,
        )
        # Set skip to identity to test contribution
        with torch.no_grad():
            transcoder.skip.weight.copy_(torch.eye(64))

        mlp_input = torch.randn(16, 64)
        mlp_output = torch.randn(16, 64)

        result = transcoder(mlp_input, mlp_output)

        # With identity skip, sparse=0, and zero decoder, output should be input
        # (since decoder weights are zero, sparse contribution is zero)
        expected = mlp_input  # skip(input) + sparse(=0)
        assert torch.allclose(result.predicted, expected, atol=1e-5)

    def test_l0_equals_k(self):
        """Test L0 equals k for TopK activation."""
        transcoder = SkipTranscoder(
            input_dim=64,
            output_dim=64,
            hidden_dim=128,
            k=8,
        )
        mlp_input = torch.randn(16, 64)
        mlp_output = torch.randn(16, 64)

        result = transcoder(mlp_input, mlp_output)
        assert result.l0.item() == 8.0

    def test_get_skip_contribution(self):
        """Test skip contribution metric."""
        transcoder = SkipTranscoder(
            input_dim=64,
            output_dim=64,
            hidden_dim=128,
            k=8,
        )
        # Set skip to identity
        with torch.no_grad():
            transcoder.skip.weight.copy_(torch.eye(64))

        mlp_input = torch.randn(100, 64)
        mlp_output = mlp_input.clone()  # Output equals input

        contribution = transcoder.get_skip_contribution(mlp_input, mlp_output)
        # With identity skip and output=input, skip should explain all variance
        assert contribution > 0.99

    def test_gradients_flow_through_both_paths(self):
        """Test gradients flow through both sparse and skip paths."""
        transcoder = SkipTranscoder(
            input_dim=64,
            output_dim=64,
            hidden_dim=128,
            k=8,
        )
        mlp_input = torch.randn(16, 64, requires_grad=True)
        mlp_output = torch.randn(16, 64)

        result = transcoder(mlp_input, mlp_output)
        result.loss.backward()

        assert transcoder.encoder.weight.grad is not None
        assert transcoder.decoder.weight.grad is not None
        assert transcoder.skip.weight.grad is not None


class TestCreateTranscoder:
    """Tests for create_transcoder factory function."""

    def test_create_with_skip(self):
        """Test creating SkipTranscoder."""
        transcoder = create_transcoder(
            input_dim=64,
            output_dim=64,
            hidden_dim=128,
            k=8,
            use_skip=True,
        )
        assert isinstance(transcoder, SkipTranscoder)

    def test_create_without_skip(self):
        """Test creating TopKTranscoder."""
        transcoder = create_transcoder(
            input_dim=64,
            output_dim=64,
            hidden_dim=128,
            k=8,
            use_skip=False,
        )
        assert isinstance(transcoder, TopKTranscoder)

    def test_passes_kwargs(self):
        """Test that kwargs are passed to constructor."""
        transcoder = create_transcoder(
            input_dim=64,
            output_dim=64,
            hidden_dim=128,
            k=16,  # Different k
            use_skip=True,
            dead_feature_threshold=5000,
        )
        assert transcoder.k == 16
        assert transcoder.dead_feature_threshold == 5000


class TestTranscoderTraining:
    """Integration tests for transcoder training behavior."""

    def test_loss_decreases_with_training(self):
        """Test that loss decreases with gradient updates."""
        torch.manual_seed(42)

        transcoder = SkipTranscoder(
            input_dim=64,
            output_dim=64,
            hidden_dim=256,
            k=32,
        )

        # Generate synthetic MLP data (simple linear transformation)
        W_true = torch.randn(64, 64) * 0.1
        mlp_input = torch.randn(100, 64)
        mlp_output = mlp_input @ W_true

        optimizer = torch.optim.Adam(transcoder.parameters(), lr=1e-3)

        initial_loss = None
        final_loss = None

        for i in range(100):
            optimizer.zero_grad()
            result = transcoder(mlp_input, mlp_output)
            result.loss.backward()
            optimizer.step()

            if i == 0:
                initial_loss = result.loss.item()
            if i == 99:
                final_loss = result.loss.item()

        assert final_loss < initial_loss, f"Loss should decrease: {initial_loss:.4f} -> {final_loss:.4f}"

    def test_skip_helps_linear_transformations(self):
        """Test that skip connection helps for linear MLP transformations."""
        torch.manual_seed(42)

        # Create a simple linear MLP transformation
        W_true = torch.randn(64, 64) * 0.1
        mlp_input = torch.randn(100, 64)
        mlp_output = mlp_input @ W_true

        # Train SkipTranscoder
        skip_tc = SkipTranscoder(
            input_dim=64,
            output_dim=64,
            hidden_dim=128,
            k=16,
        )
        optimizer = torch.optim.Adam(skip_tc.parameters(), lr=1e-2)

        for _ in range(50):
            optimizer.zero_grad()
            result = skip_tc(mlp_input, mlp_output)
            result.loss.backward()
            optimizer.step()

        skip_loss = skip_tc(mlp_input, mlp_output).loss.item()

        # Train TopKTranscoder (no skip)
        topk_tc = TopKTranscoder(
            input_dim=64,
            output_dim=64,
            hidden_dim=128,
            k=16,
        )
        optimizer = torch.optim.Adam(topk_tc.parameters(), lr=1e-2)

        for _ in range(50):
            optimizer.zero_grad()
            result = topk_tc(mlp_input, mlp_output)
            result.loss.backward()
            optimizer.step()

        topk_loss = topk_tc(mlp_input, mlp_output).loss.item()

        # Skip should achieve lower loss for linear transformations
        assert skip_loss < topk_loss, f"Skip should help: {skip_loss:.4f} < {topk_loss:.4f}"


class TestTranscoderResampling:
    """Tests for dead feature resampling in transcoders."""

    def test_resample_dead_features(self):
        """Test dead feature resampling works."""
        transcoder = TopKTranscoder(
            input_dim=64,
            output_dim=64,
            hidden_dim=128,
            k=8,
            dead_feature_threshold=10,
        )
        # Use eval mode to avoid updating dead feature tracking during resample
        transcoder.eval()

        # Simulate many steps without some features activating
        transcoder.step_count = torch.tensor(1000, dtype=torch.long)

        mlp_input = torch.randn(100, 64)
        mlp_output = torch.randn(100, 64)

        num_dead_before = transcoder.get_dead_features().sum().item()
        assert num_dead_before == 128  # All dead initially

        num_resampled = transcoder.resample_dead_features(mlp_input, mlp_output, num_resample=10)

        assert num_resampled == 10
        # Resampled features should no longer be dead
        num_dead_after = transcoder.get_dead_features().sum().item()
        assert num_dead_after == 128 - 10
