"""Tests for Crosscoder models."""

import pytest
import torch

from whisper_sae.sae.crosscoder import (
    CrossLayerCrosscoder,
    TopKCrossLayerCrosscoder,
    CrosscoderOutput,
    create_crosscoder,
)


class TestCrossLayerCrosscoder:
    """Tests for CrossLayerCrosscoder."""

    def test_initialization(self):
        """Test crosscoder initializes with correct dimensions."""
        cc = CrossLayerCrosscoder(
            d_model=64,
            n_layers=4,
            d_sae=128,
        )
        assert cc.d_model == 64
        assert cc.n_layers == 4
        assert cc.d_sae == 128

    def test_weight_shapes(self):
        """Test encoder and decoder weight shapes."""
        cc = CrossLayerCrosscoder(
            d_model=64,
            n_layers=4,
            d_sae=128,
        )
        # W_enc: [n_layers, d_model, d_sae]
        assert cc.W_enc.shape == (4, 64, 128)
        # W_dec: [d_sae, n_layers, d_model]
        assert cc.W_dec.shape == (128, 4, 64)
        # Biases
        assert cc.b_enc.shape == (128,)
        assert cc.b_dec.shape == (4, 64)

    def test_layer_indices(self):
        """Test custom layer indices."""
        cc = CrossLayerCrosscoder(
            d_model=64,
            n_layers=2,
            d_sae=128,
            layer_indices=[1, 3],  # Non-contiguous
        )
        assert cc.layer_indices == [1, 3]
        assert cc.n_layers == 2

    def test_encode_output_shape(self):
        """Test encode produces correct output shape."""
        cc = CrossLayerCrosscoder(
            d_model=64,
            n_layers=4,
            d_sae=128,
        )
        layer_acts = {
            0: torch.randn(16, 64),
            1: torch.randn(16, 64),
            2: torch.randn(16, 64),
            3: torch.randn(16, 64),
        }
        hidden = cc.encode(layer_acts)
        assert hidden.shape == (16, 128)

    def test_encode_combines_layers(self):
        """Test encode sums contributions from all layers."""
        cc = CrossLayerCrosscoder(
            d_model=64,
            n_layers=4,
            d_sae=128,
        )
        # Test with only layer 0
        layer_acts_single = {0: torch.randn(16, 64)}
        hidden_single = cc.encode({
            0: layer_acts_single[0],
            1: torch.zeros(16, 64),
            2: torch.zeros(16, 64),
            3: torch.zeros(16, 64),
        })

        # Should be different from using all layers
        layer_acts_all = {i: torch.randn(16, 64) for i in range(4)}
        hidden_all = cc.encode(layer_acts_all)

        assert not torch.allclose(hidden_single, hidden_all)

    def test_decode_output_shape(self):
        """Test decode produces per-layer outputs."""
        cc = CrossLayerCrosscoder(
            d_model=64,
            n_layers=4,
            d_sae=128,
        )
        hidden = torch.randn(16, 128)
        reconstructed = cc.decode(hidden)

        assert len(reconstructed) == 4
        for layer_idx in range(4):
            assert layer_idx in reconstructed
            assert reconstructed[layer_idx].shape == (16, 64)

    def test_forward_returns_crosscoder_output(self):
        """Test forward returns CrosscoderOutput."""
        cc = CrossLayerCrosscoder(
            d_model=64,
            n_layers=4,
            d_sae=128,
        )
        layer_acts = {i: torch.randn(16, 64) for i in range(4)}
        result = cc(layer_acts)
        assert isinstance(result, CrosscoderOutput)

    def test_forward_output_contents(self):
        """Test CrosscoderOutput has correct contents."""
        cc = CrossLayerCrosscoder(
            d_model=64,
            n_layers=4,
            d_sae=128,
        )
        layer_acts = {i: torch.randn(16, 64) for i in range(4)}
        result = cc(layer_acts)

        # Check reconstructed dict
        assert len(result.reconstructed) == 4
        for i in range(4):
            assert result.reconstructed[i].shape == (16, 64)

        # Check hidden
        assert result.hidden.shape == (16, 128)

        # Check losses are scalars
        assert result.loss.ndim == 0
        assert result.reconstruction_loss.ndim == 0
        assert result.sparsity_loss.ndim == 0
        assert result.l0.ndim == 0

        # Check per-layer losses
        assert len(result.per_layer_loss) == 4

    def test_reconstruction_loss_is_sum_of_layers(self):
        """Test reconstruction loss sums per-layer losses."""
        cc = CrossLayerCrosscoder(
            d_model=64,
            n_layers=4,
            d_sae=128,
        )
        layer_acts = {i: torch.randn(16, 64) for i in range(4)}
        result = cc(layer_acts)

        expected_sum = sum(result.per_layer_loss.values())
        assert torch.isclose(result.reconstruction_loss, expected_sum)

    def test_sparsity_loss_uses_decoder_norms(self):
        """Test sparsity loss is decoder norm-weighted."""
        cc = CrossLayerCrosscoder(
            d_model=64,
            n_layers=4,
            d_sae=128,
            sparsity_weight=1.0,
        )
        layer_acts = {i: torch.randn(16, 64) for i in range(4)}
        result = cc(layer_acts)

        # Sparsity loss should be non-zero when hidden is non-zero
        if result.l0 > 0:
            assert result.sparsity_loss > 0

    def test_dead_feature_tracking(self):
        """Test dead feature tracking updates."""
        cc = CrossLayerCrosscoder(
            d_model=64,
            n_layers=4,
            d_sae=128,
            dead_feature_threshold=10,
        )
        cc.train()

        assert cc.step_count == 0

        layer_acts = {i: torch.randn(16, 64) for i in range(4)}
        cc(layer_acts)

        assert cc.step_count == 1

    def test_get_decoder_norms(self):
        """Test decoder norm computation."""
        cc = CrossLayerCrosscoder(
            d_model=64,
            n_layers=4,
            d_sae=128,
            normalize_decoder=True,
        )
        norms = cc.get_decoder_norms()
        assert norms.shape == (128,)
        # With normalization, norms should be approximately 0.1
        assert torch.allclose(norms, torch.full_like(norms, 0.1), atol=0.01)

    def test_get_feature_layer_norms(self):
        """Test per-layer decoder norm computation."""
        cc = CrossLayerCrosscoder(
            d_model=64,
            n_layers=4,
            d_sae=128,
        )
        layer_norms = cc.get_feature_layer_norms()
        assert layer_norms.shape == (128, 4)

    def test_get_cross_layer_features(self):
        """Test cross-layer feature detection."""
        cc = CrossLayerCrosscoder(
            d_model=64,
            n_layers=4,
            d_sae=128,
        )
        cross_layer = cc.get_cross_layer_features(threshold=0.1)
        assert cross_layer.shape == (128,)
        assert cross_layer.dtype == torch.bool

    def test_gradients_flow(self):
        """Test gradients flow through crosscoder."""
        cc = CrossLayerCrosscoder(
            d_model=64,
            n_layers=4,
            d_sae=128,
        )
        layer_acts = {i: torch.randn(16, 64, requires_grad=True) for i in range(4)}
        result = cc(layer_acts)
        result.loss.backward()

        for i in range(4):
            assert layer_acts[i].grad is not None
        assert cc.W_enc.grad is not None
        assert cc.W_dec.grad is not None


class TestTopKCrossLayerCrosscoder:
    """Tests for TopKCrossLayerCrosscoder."""

    def test_initialization(self):
        """Test TopK crosscoder initializes correctly."""
        cc = TopKCrossLayerCrosscoder(
            d_model=64,
            n_layers=4,
            d_sae=128,
            k=16,
        )
        assert cc.k == 16
        assert cc.d_sae == 128

    def test_topk_sparsity(self):
        """Test exactly k features are active."""
        cc = TopKCrossLayerCrosscoder(
            d_model=64,
            n_layers=4,
            d_sae=128,
            k=16,
        )
        layer_acts = {i: torch.randn(32, 64) for i in range(4)}
        hidden = cc.encode(layer_acts)

        active_per_sample = (hidden > 0).sum(dim=-1)
        assert torch.all(active_per_sample == 16)

    def test_l0_equals_k(self):
        """Test L0 equals k for TopK."""
        cc = TopKCrossLayerCrosscoder(
            d_model=64,
            n_layers=4,
            d_sae=128,
            k=16,
        )
        layer_acts = {i: torch.randn(32, 64) for i in range(4)}
        result = cc(layer_acts)
        assert result.l0.item() == 16.0

    def test_no_sparsity_loss(self):
        """Test sparsity loss is zero for TopK."""
        cc = TopKCrossLayerCrosscoder(
            d_model=64,
            n_layers=4,
            d_sae=128,
            k=16,
        )
        layer_acts = {i: torch.randn(32, 64) for i in range(4)}
        result = cc(layer_acts)
        assert result.sparsity_loss.item() == 0.0

    def test_forward_returns_crosscoder_output(self):
        """Test forward returns proper output type."""
        cc = TopKCrossLayerCrosscoder(
            d_model=64,
            n_layers=4,
            d_sae=128,
            k=16,
        )
        layer_acts = {i: torch.randn(32, 64) for i in range(4)}
        result = cc(layer_acts)
        assert isinstance(result, CrosscoderOutput)


class TestCreateCrosscoder:
    """Tests for create_crosscoder factory."""

    def test_create_topk(self):
        """Test creating TopK crosscoder."""
        cc = create_crosscoder(
            d_model=64,
            n_layers=4,
            d_sae=128,
            k=16,
            use_topk=True,
        )
        assert isinstance(cc, TopKCrossLayerCrosscoder)
        assert cc.k == 16

    def test_create_relu(self):
        """Test creating ReLU crosscoder."""
        cc = create_crosscoder(
            d_model=64,
            n_layers=4,
            d_sae=128,
            use_topk=False,
        )
        assert isinstance(cc, CrossLayerCrosscoder)
        assert not isinstance(cc, TopKCrossLayerCrosscoder)

    def test_passes_kwargs(self):
        """Test kwargs are passed through."""
        cc = create_crosscoder(
            d_model=64,
            n_layers=4,
            d_sae=256,
            k=32,
            use_topk=True,
            layer_indices=[0, 2],
            dead_feature_threshold=5000,
        )
        assert cc.d_sae == 256
        assert cc.k == 32
        assert cc.layer_indices == [0, 2]
        assert cc.dead_feature_threshold == 5000


class TestCrosscoderTraining:
    """Integration tests for crosscoder training."""

    def test_loss_decreases(self):
        """Test loss decreases with training."""
        torch.manual_seed(42)

        cc = TopKCrossLayerCrosscoder(
            d_model=64,
            n_layers=4,
            d_sae=256,
            k=32,
        )

        # Generate correlated layer activations
        base = torch.randn(100, 64)
        layer_acts = {i: base + 0.1 * torch.randn(100, 64) for i in range(4)}

        optimizer = torch.optim.Adam(cc.parameters(), lr=1e-3)

        initial_loss = None
        final_loss = None

        for i in range(50):
            optimizer.zero_grad()
            result = cc(layer_acts)
            result.loss.backward()
            optimizer.step()
            cc.normalize_decoder_weights()

            if i == 0:
                initial_loss = result.loss.item()
            if i == 49:
                final_loss = result.loss.item()

        assert final_loss < initial_loss, f"Loss should decrease: {initial_loss:.4f} -> {final_loss:.4f}"

    def test_finds_shared_features(self):
        """Test crosscoder can find features shared across layers."""
        torch.manual_seed(42)

        cc = TopKCrossLayerCrosscoder(
            d_model=32,
            n_layers=4,
            d_sae=64,
            k=8,
        )

        # Create activations with a shared component
        shared = torch.randn(100, 32)
        layer_acts = {i: shared + 0.5 * torch.randn(100, 32) for i in range(4)}

        optimizer = torch.optim.Adam(cc.parameters(), lr=1e-2)

        for _ in range(100):
            optimizer.zero_grad()
            result = cc(layer_acts)
            result.loss.backward()
            optimizer.step()
            cc.normalize_decoder_weights()

        # Check that some features are cross-layer
        cross_layer = cc.get_cross_layer_features(threshold=0.3)
        num_cross_layer = cross_layer.sum().item()

        # With highly correlated layers, we should find cross-layer features
        assert num_cross_layer > 0, "Should find some cross-layer features"


class TestCrosscoderWhisperCompatibility:
    """Test crosscoder with Whisper-like dimensions."""

    def test_whisper_tiny_dimensions(self):
        """Test with Whisper-tiny dimensions (d_model=384, 4 layers)."""
        cc = TopKCrossLayerCrosscoder(
            d_model=384,
            n_layers=4,
            d_sae=384 * 8,  # 8x expansion
            k=32,
            layer_indices=[0, 1, 2, 3],
        )

        # Simulate encoder activations
        layer_acts = {i: torch.randn(8, 384) for i in range(4)}
        result = cc(layer_acts)

        assert result.hidden.shape == (8, 384 * 8)
        assert result.l0.item() == 32.0

    def test_subset_of_layers(self):
        """Test crosscoder on subset of layers."""
        # Only use layers 1 and 2
        cc = TopKCrossLayerCrosscoder(
            d_model=384,
            n_layers=2,
            d_sae=384 * 4,
            k=32,
            layer_indices=[1, 2],
        )

        layer_acts = {1: torch.randn(8, 384), 2: torch.randn(8, 384)}
        result = cc(layer_acts)

        assert len(result.reconstructed) == 2
        assert 1 in result.reconstructed
        assert 2 in result.reconstructed
