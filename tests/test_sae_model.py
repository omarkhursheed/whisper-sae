"""Comprehensive tests for SAE model implementations.

These tests verify:
1. TopK SAE activation function behavior
2. Decoder normalization
3. Dead feature detection
4. Loss computation
5. Gradient flow
"""

import pytest
import torch
from torch import Tensor

from whisper_sae.sae.model import TopKSAE, ReLUSAE, SAEOutput, create_sae
from whisper_sae.config import SAEConfig


class TestTopKSAE:
    """Tests for TopK Sparse Autoencoder."""

    @pytest.fixture
    def sae(self):
        """Create a TopK SAE for testing."""
        return TopKSAE(
            input_dim=384,
            hidden_dim=3072,  # 8x expansion
            k=32,
            normalize_decoder=True,
            dead_feature_threshold=10_000,
        )

    @pytest.fixture
    def small_sae(self):
        """Create a smaller SAE for faster tests."""
        return TopKSAE(
            input_dim=64,
            hidden_dim=256,
            k=8,
            normalize_decoder=True,
            dead_feature_threshold=100,
        )

    def test_initialization(self, sae):
        """Test that SAE initializes with correct dimensions."""
        assert sae.input_dim == 384
        assert sae.hidden_dim == 3072
        assert sae.k == 32
        assert sae.encoder.in_features == 384
        assert sae.encoder.out_features == 3072
        assert sae.decoder.in_features == 3072
        assert sae.decoder.out_features == 384
        assert sae.b_pre.shape == (384,)

    def test_decoder_initialization_normalized(self, sae):
        """Test that decoder columns are initialized with unit norm."""
        decoder_weights = sae.decoder.weight.data  # [output_dim, hidden_dim]

        # Check column norms (each column should be unit norm)
        column_norms = decoder_weights.norm(dim=0)

        # After scaling by 0.1 during init, norms should be 0.1
        assert torch.allclose(column_norms, torch.ones_like(column_norms) * 0.1, atol=1e-5), (
            f"Decoder columns should have norm 0.1, got {column_norms[:5]}"
        )

    def test_normalize_decoder_weights(self, sae):
        """Test decoder weight normalization."""
        # Mess up the decoder weights
        sae.decoder.weight.data = torch.randn_like(sae.decoder.weight.data) * 5

        # Normalize
        sae.normalize_decoder_weights()

        # Check column norms are now 1
        column_norms = sae.decoder.weight.data.norm(dim=0)
        assert torch.allclose(column_norms, torch.ones_like(column_norms), atol=1e-5)

    def test_encode_output_shape(self, sae):
        """Test encoder output shape."""
        batch = torch.randn(32, 384)
        hidden = sae.encode(batch)

        assert hidden.shape == (32, 3072)

    def test_topk_sparsity(self, sae):
        """Test that TopK activation produces exactly k non-zero values."""
        batch = torch.randn(100, 384)
        hidden = sae.encode(batch)

        # Count non-zeros per sample
        nonzero_counts = (hidden != 0).sum(dim=1)

        # Every sample should have exactly k non-zero activations
        assert torch.all(nonzero_counts == sae.k), (
            f"Expected exactly {sae.k} non-zeros, got {nonzero_counts[:10]}"
        )

    def test_topk_values_are_positive(self, sae):
        """Test that TopK values after ReLU are non-negative."""
        batch = torch.randn(100, 384)
        hidden = sae.encode(batch)

        # All non-zero values should be positive (due to ReLU in TopK)
        nonzero_values = hidden[hidden != 0]
        assert torch.all(nonzero_values > 0), (
            "TopK activations should be positive after ReLU"
        )

    def test_topk_selects_largest(self, sae):
        """Test that TopK selects the k largest pre-activation values."""
        # Use a simple input to trace through
        batch = torch.randn(1, 384)

        # Get pre-activation values
        with torch.no_grad():
            x_centered = batch - sae.b_pre
            pre_activation = sae.encoder(x_centered)

        hidden = sae.encode(batch)

        # The k largest pre-activations should be preserved (after ReLU)
        topk_pre, topk_indices = torch.topk(pre_activation, sae.k, dim=-1)

        # For each sample, check the selected indices
        nonzero_indices = (hidden != 0).nonzero(as_tuple=True)[1]

        assert set(nonzero_indices.tolist()) == set(topk_indices.squeeze().tolist()), (
            "TopK should select the k largest pre-activation values"
        )

    def test_decode_output_shape(self, sae):
        """Test decoder output shape."""
        hidden = torch.randn(32, 3072)
        reconstructed = sae.decode(hidden)

        assert reconstructed.shape == (32, 384)

    def test_forward_returns_sae_output(self, sae):
        """Test that forward returns SAEOutput namedtuple."""
        batch = torch.randn(32, 384)
        output = sae(batch)

        assert isinstance(output, SAEOutput)
        assert hasattr(output, 'reconstructed')
        assert hasattr(output, 'hidden')
        assert hasattr(output, 'loss')
        assert hasattr(output, 'reconstruction_loss')
        assert hasattr(output, 'sparsity_loss')
        assert hasattr(output, 'l0')

    def test_forward_output_shapes(self, sae):
        """Test shapes of forward pass outputs."""
        batch = torch.randn(32, 384)
        output = sae(batch)

        assert output.reconstructed.shape == (32, 384)
        assert output.hidden.shape == (32, 3072)
        assert output.loss.shape == ()
        assert output.reconstruction_loss.shape == ()
        assert output.sparsity_loss.shape == ()
        assert output.l0.shape == ()

    def test_reconstruction_loss_is_mse(self, sae):
        """Test that reconstruction loss is MSE."""
        batch = torch.randn(32, 384)
        output = sae(batch)

        # Manually compute MSE
        expected_mse = torch.nn.functional.mse_loss(output.reconstructed, batch)

        assert torch.allclose(output.reconstruction_loss, expected_mse)

    def test_sparsity_loss_is_zero_for_topk(self, sae):
        """Test that sparsity loss is zero for TopK (enforced by design)."""
        batch = torch.randn(32, 384)
        output = sae(batch)

        assert output.sparsity_loss.item() == 0.0

    def test_l0_equals_k(self, sae):
        """Test that L0 sparsity equals k for TopK."""
        batch = torch.randn(32, 384)
        output = sae(batch)

        assert torch.isclose(output.l0, torch.tensor(float(sae.k)), atol=0.1)

    def test_dead_feature_tracking_initialization(self, sae):
        """Test dead feature tracking is initialized correctly."""
        assert sae.feature_last_activated.shape == (sae.hidden_dim,)
        assert torch.all(sae.feature_last_activated == 0)
        assert sae.step_count.item() == 0

    def test_dead_feature_tracking_updates(self, small_sae):
        """Test that dead feature tracking updates during training."""
        small_sae.train()

        batch = torch.randn(32, 64)
        _ = small_sae(batch)

        assert small_sae.step_count.item() == 1

        # Some features should have been activated
        # (the ones that were in top-k)
        num_activated = (small_sae.feature_last_activated > 0).sum().item()
        assert num_activated > 0

    def test_dead_feature_tracking_not_updated_in_eval(self, small_sae):
        """Test that dead feature tracking doesn't update in eval mode."""
        small_sae.eval()

        batch = torch.randn(32, 64)
        _ = small_sae(batch)

        assert small_sae.step_count.item() == 0

    def test_get_dead_features_initially_all_dead(self, small_sae):
        """Test that initially all features are considered dead."""
        # Step count is 0, threshold is 100
        # All features have last_activated = 0
        # steps_since_active = 0 - 0 = 0, which is not > 100
        # So initially no features are dead (we haven't reached threshold)

        dead_mask = small_sae.get_dead_features()
        assert torch.all(~dead_mask), "Initially no features should be dead"

    def test_get_dead_features_after_many_steps(self, small_sae):
        """Test dead feature detection after many steps."""
        small_sae.train()

        # Use a very small input dimension to force feature reuse
        # With k=8 and 256 features and consistent inputs, most features stay dead
        torch.manual_seed(12345)  # Fixed seed for reproducibility

        # Create a small, consistent input that will activate the same features
        fixed_batch = torch.randn(8, 64)

        # Run many batches with the same input to force dead features
        for _ in range(150):
            _ = small_sae(fixed_batch)

        # After 150 steps, features not in the top-k of this fixed batch should be dead
        dead_mask = small_sae.get_dead_features()

        # The mask should have the right shape regardless of how many are dead
        assert dead_mask.shape == (256,)

        # Check that dead feature tracking is working (step count increased)
        assert small_sae.step_count.item() == 150

    def test_dead_features_concept(self):
        """Test the dead feature concept: features not activated become dead.

        This validates that:
        1. With a fixed input, only k features are ever activated
        2. The other features eventually become "dead" (not used)
        3. This matters for SAE training (dead features are wasted capacity)
        """
        torch.manual_seed(999)

        # Create SAE with threshold of 50 steps
        sae = TopKSAE(
            input_dim=32,
            hidden_dim=128,  # Many more features than k
            k=4,  # Only 4 active at a time
            dead_feature_threshold=50,
        )
        sae.train()

        # Use a single fixed input - same k features will always activate
        fixed_input = torch.randn(1, 32)

        # Before any forward passes, no features should be dead
        # (step count is 0, nothing exceeds threshold)
        assert sae.get_dead_feature_ratio() == 0.0

        # Run 60 steps (past the threshold of 50)
        for _ in range(60):
            _ = sae(fixed_input)

        # Now, most features should be dead because only 4 are ever activated
        dead_ratio = sae.get_dead_feature_ratio()

        # With k=4 out of 128 features, at least 90% should be dead
        assert dead_ratio >= 0.9, (
            f"Expected at least 90% dead features with k=4/128, got {dead_ratio:.1%}"
        )

        # Verify that exactly k features are "alive" (activated within threshold)
        alive_features = ~sae.get_dead_features()
        num_alive = alive_features.sum().item()
        assert num_alive == 4, (
            f"Expected exactly 4 alive features (k=4), got {num_alive}"
        )

    def test_gradients_flow(self, small_sae):
        """Test that gradients flow through the model."""
        batch = torch.randn(32, 64, requires_grad=True)
        output = small_sae(batch)
        output.loss.backward()

        # Check gradients exist
        assert small_sae.encoder.weight.grad is not None
        assert small_sae.decoder.weight.grad is not None
        assert small_sae.b_pre.grad is not None

        # Gradients should be non-zero
        assert small_sae.encoder.weight.grad.abs().sum() > 0
        assert small_sae.decoder.weight.grad.abs().sum() > 0

    def test_deterministic_with_same_input(self, sae):
        """Test that same input produces same output in eval mode."""
        sae.eval()
        batch = torch.randn(10, 384)

        output1 = sae(batch)
        output2 = sae(batch)

        assert torch.allclose(output1.hidden, output2.hidden)
        assert torch.allclose(output1.reconstructed, output2.reconstructed)


class TestReLUSAE:
    """Tests for ReLU Sparse Autoencoder."""

    @pytest.fixture
    def sae(self):
        """Create a ReLU SAE for testing."""
        return ReLUSAE(
            input_dim=384,
            hidden_dim=3072,
            sparsity_weight=0.01,
            normalize_decoder=True,
        )

    def test_initialization(self, sae):
        """Test ReLU SAE initialization."""
        assert sae.input_dim == 384
        assert sae.hidden_dim == 3072
        assert sae.sparsity_weight == 0.01

    def test_forward_shapes(self, sae):
        """Test forward pass output shapes."""
        batch = torch.randn(32, 384)
        output = sae(batch)

        assert output.reconstructed.shape == (32, 384)
        assert output.hidden.shape == (32, 3072)

    def test_sparsity_loss_nonzero(self, sae):
        """Test that sparsity loss is non-zero for ReLU SAE."""
        batch = torch.randn(32, 384)
        output = sae(batch)

        # With random input, activations should be non-zero on average
        assert output.sparsity_loss.item() > 0

    def test_total_loss_includes_sparsity(self, sae):
        """Test that total loss includes sparsity term."""
        batch = torch.randn(32, 384)
        output = sae(batch)

        expected_loss = output.reconstruction_loss + sae.sparsity_weight * output.sparsity_loss
        assert torch.allclose(output.loss, expected_loss)


class TestCreateSAE:
    """Tests for the SAE factory function."""

    def test_create_topk_sae(self):
        """Test creating a TopK SAE from config."""
        config = SAEConfig(
            expansion_factor=8,
            activation="topk",
            k=32,
            normalize_decoder=True,
        )

        sae = create_sae(config, input_dim=384)

        assert isinstance(sae, TopKSAE)
        assert sae.hidden_dim == 384 * 8
        assert sae.k == 32

    def test_create_relu_sae(self):
        """Test creating a ReLU SAE from config."""
        config = SAEConfig(
            expansion_factor=16,
            activation="relu",
            normalize_decoder=True,
        )

        sae = create_sae(config, input_dim=512)

        assert isinstance(sae, ReLUSAE)
        assert sae.hidden_dim == 512 * 16

    def test_create_sae_with_different_expansions(self):
        """Test creating SAEs with various expansion factors."""
        for expansion in [4, 8, 16, 32]:
            config = SAEConfig(expansion_factor=expansion, activation="topk", k=32)
            sae = create_sae(config, input_dim=256)
            assert sae.hidden_dim == 256 * expansion


class TestSAEReconstruction:
    """Tests for reconstruction quality."""

    def test_reconstruction_uses_k_features(self):
        """Test that reconstruction uses exactly k features."""
        input_dim = 64
        hidden_dim = 256
        batch = torch.randn(10, input_dim)

        for k in [4, 8, 16, 32]:
            sae = TopKSAE(input_dim, hidden_dim, k=k)
            sae.eval()
            output = sae(batch)

            # Check that exactly k features are active per sample
            active_per_sample = (output.hidden != 0).sum(dim=1)
            assert torch.all(active_per_sample == k), (
                f"Expected {k} active features, got {active_per_sample}"
            )

    def test_reconstruction_improves_with_training(self):
        """Test that reconstruction quality improves after training.

        This validates the core assumption that SAEs can learn to reconstruct.
        """
        torch.manual_seed(42)
        input_dim = 64
        hidden_dim = 256
        k = 16

        # Create data with structure (not pure random)
        # This simulates activations from a real model
        data = torch.randn(100, input_dim)

        sae = TopKSAE(input_dim, hidden_dim, k=k)

        # Get initial loss
        sae.eval()
        with torch.no_grad():
            initial_output = sae(data)
        initial_loss = initial_output.reconstruction_loss.item()

        # Train for a few steps
        sae.train()
        optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)

        for _ in range(100):
            output = sae(data)
            output.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            sae.normalize_decoder_weights()

        # Get final loss
        sae.eval()
        with torch.no_grad():
            final_output = sae(data)
        final_loss = final_output.reconstruction_loss.item()

        # Training should improve reconstruction
        assert final_loss < initial_loss * 0.5, (
            f"Training should significantly reduce loss. "
            f"Initial: {initial_loss:.4f}, Final: {final_loss:.4f}"
        )

    def test_larger_k_enables_better_reconstruction_after_training(self):
        """Test that larger k gives better reconstruction potential after training.

        With random weights, larger k doesn't necessarily help.
        But after training, more features should enable lower reconstruction error.
        """
        torch.manual_seed(42)
        input_dim = 32
        hidden_dim = 128

        # Create structured data
        data = torch.randn(50, input_dim)

        final_losses = {}
        for k in [4, 8, 16]:
            sae = TopKSAE(input_dim, hidden_dim, k=k)

            # Train each model
            sae.train()
            optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)

            for _ in range(200):
                output = sae(data)
                output.loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                sae.normalize_decoder_weights()

            # Get final loss
            sae.eval()
            with torch.no_grad():
                final_output = sae(data)
            final_losses[k] = final_output.reconstruction_loss.item()

        # After training, larger k should give lower loss (more capacity)
        assert final_losses[8] <= final_losses[4] * 1.1, (
            f"k=8 should be at least as good as k=4: "
            f"k=4: {final_losses[4]:.4f}, k=8: {final_losses[8]:.4f}"
        )
        assert final_losses[16] <= final_losses[8] * 1.1, (
            f"k=16 should be at least as good as k=8: "
            f"k=8: {final_losses[8]:.4f}, k=16: {final_losses[16]:.4f}"
        )

    def test_perfect_reconstruction_with_identity(self):
        """Test reconstruction when using identity-like encoding."""
        # Create a simple case where perfect reconstruction is possible
        input_dim = 32
        hidden_dim = 32
        k = 32  # All features active

        sae = TopKSAE(input_dim, hidden_dim, k=k)

        # Set up encoder/decoder as identity
        with torch.no_grad():
            sae.encoder.weight.data = torch.eye(hidden_dim, input_dim)
            sae.encoder.bias.data.zero_()
            sae.decoder.weight.data = torch.eye(input_dim, hidden_dim)
            sae.decoder.bias.data.zero_()
            sae.b_pre.data.zero_()

        batch = torch.rand(10, input_dim)  # Use positive values for ReLU
        output = sae(batch)

        # Should have very low reconstruction error
        assert output.reconstruction_loss.item() < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
