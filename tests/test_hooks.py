"""Comprehensive tests for Whisper activation extraction hooks.

These tests verify that:
1. Hook registration and removal works correctly
2. Activation shapes match expected dimensions at each layer
3. Activations from hooks match manual extraction
4. Layer norm application is correct
5. Behavior is consistent with SAELens/TransformerLens conventions

The tests cover all edge cases and verify correctness thoroughly.
"""

import pytest
import torch
from torch import Tensor
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from whisper_sae.sae.hooks import (
    ActivationCache,
    WhisperActivationExtractor,
    extract_features_batch,
    flatten_activations,
)


# Fixtures
@pytest.fixture(scope="module")
def whisper_model():
    """Load Whisper-tiny model for testing."""
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    model.eval()
    return model


@pytest.fixture(scope="module")
def whisper_processor():
    """Load Whisper processor."""
    return WhisperProcessor.from_pretrained("openai/whisper-tiny")


@pytest.fixture
def dummy_audio_features():
    """Create dummy mel spectrogram input."""
    # Whisper expects [batch, 80, 3000] mel spectrogram
    # 80 mel bins, 3000 frames (30 seconds at 100 frames/sec)
    batch_size = 2
    return torch.randn(batch_size, 80, 3000)


@pytest.fixture
def small_audio_features():
    """Create smaller dummy input for faster tests."""
    batch_size = 1
    return torch.randn(batch_size, 80, 3000)


class TestActivationCache:
    """Tests for the ActivationCache dataclass."""

    def test_cache_initialization(self):
        """Test that cache initializes empty."""
        cache = ActivationCache()
        assert cache.encoder == {}
        assert cache.decoder == {}

    def test_cache_clear(self):
        """Test that clear empties the cache."""
        cache = ActivationCache()
        cache.encoder[0] = [torch.randn(10, 384)]
        cache.decoder[0] = [torch.randn(10, 384)]

        cache.clear()

        assert cache.encoder == {}
        assert cache.decoder == {}

    def test_get_encoder_activations_empty(self):
        """Test getting activations from empty cache."""
        cache = ActivationCache()
        result = cache.get_encoder_activations(0)
        assert result is None

    def test_get_encoder_activations_single(self):
        """Test getting activations with single batch."""
        cache = ActivationCache()
        expected = torch.randn(10, 384)
        cache.encoder[0] = [expected]

        result = cache.get_encoder_activations(0)

        assert result is not None
        assert torch.allclose(result, expected)

    def test_get_encoder_activations_multiple(self):
        """Test getting activations from multiple batches."""
        cache = ActivationCache()
        batch1 = torch.randn(10, 384)
        batch2 = torch.randn(20, 384)
        cache.encoder[0] = [batch1, batch2]

        result = cache.get_encoder_activations(0)

        assert result is not None
        assert result.shape == (30, 384)
        assert torch.allclose(result[:10], batch1)
        assert torch.allclose(result[10:], batch2)

    def test_get_decoder_activations(self):
        """Test getting decoder activations."""
        cache = ActivationCache()
        expected = torch.randn(5, 384)
        cache.decoder[2] = [expected]

        result = cache.get_decoder_activations(2)

        assert result is not None
        assert torch.allclose(result, expected)


class TestWhisperActivationExtractor:
    """Tests for the WhisperActivationExtractor class."""

    def test_extractor_initialization(self, whisper_model):
        """Test extractor initializes correctly."""
        extractor = WhisperActivationExtractor(
            model=whisper_model,
            encoder_layers=[0, 1, 2, 3],
            decoder_layers=[0, 1],
        )

        assert extractor.model is whisper_model
        assert extractor.encoder_layers == [0, 1, 2, 3]
        assert extractor.decoder_layers == [0, 1]
        assert extractor.apply_layer_norm is True
        assert len(extractor._hooks) == 0

    def test_hook_registration(self, whisper_model):
        """Test that hooks are properly registered."""
        extractor = WhisperActivationExtractor(
            model=whisper_model,
            encoder_layers=[0, 2],
            decoder_layers=[1],
        )

        extractor.register_hooks()

        # Should have 3 hooks: 2 encoder + 1 decoder
        assert len(extractor._hooks) == 3

        extractor.remove_hooks()
        assert len(extractor._hooks) == 0

    def test_hook_removal_clears_list(self, whisper_model):
        """Test that removing hooks clears the hook list."""
        extractor = WhisperActivationExtractor(
            model=whisper_model,
            encoder_layers=[0, 1, 2, 3],
            decoder_layers=[0, 1, 2, 3],
        )

        extractor.register_hooks()
        assert len(extractor._hooks) == 8

        extractor.remove_hooks()
        assert len(extractor._hooks) == 0

    def test_context_manager_registers_hooks(self, whisper_model):
        """Test context manager registers and removes hooks."""
        extractor = WhisperActivationExtractor(
            model=whisper_model,
            encoder_layers=[0],
            decoder_layers=[0],
        )

        assert len(extractor._hooks) == 0

        with extractor:
            assert len(extractor._hooks) == 2

        assert len(extractor._hooks) == 0

    def test_encoder_activation_shapes(self, whisper_model, dummy_audio_features):
        """Test that encoder activations have correct shapes."""
        batch_size = dummy_audio_features.shape[0]
        encoder_layers = [0, 1, 2, 3]

        extractor = WhisperActivationExtractor(
            model=whisper_model,
            encoder_layers=encoder_layers,
            decoder_layers=[],
        )

        with torch.no_grad(), extractor:
            _ = whisper_model.model.encoder(dummy_audio_features)

        # Whisper-tiny encoder output: [batch, 1500, 384]
        # 1500 = 3000 / 2 (conv downsampling)
        for layer_idx in encoder_layers:
            activations = extractor.cache.get_encoder_activations(layer_idx)
            assert activations is not None, f"No activations for encoder layer {layer_idx}"
            assert activations.shape == (batch_size, 1500, 384), (
                f"Wrong shape for encoder layer {layer_idx}: {activations.shape}"
            )

    def test_decoder_activation_shapes(self, whisper_model, dummy_audio_features):
        """Test that decoder activations have correct shapes."""
        batch_size = dummy_audio_features.shape[0]
        decoder_layers = [0, 1, 2, 3]

        extractor = WhisperActivationExtractor(
            model=whisper_model,
            encoder_layers=[],
            decoder_layers=decoder_layers,
        )

        with torch.no_grad(), extractor:
            # Run encoder first
            encoder_output = whisper_model.model.encoder(dummy_audio_features)

            # Run decoder with start token
            decoder_input_ids = torch.full(
                (batch_size, 1),
                whisper_model.config.decoder_start_token_id,
                dtype=torch.long,
            )
            _ = whisper_model.model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_output.last_hidden_state,
            )

        # Decoder output with 1 token: [batch, 1, 384]
        for layer_idx in decoder_layers:
            activations = extractor.cache.get_decoder_activations(layer_idx)
            assert activations is not None, f"No activations for decoder layer {layer_idx}"
            assert activations.shape == (batch_size, 1, 384), (
                f"Wrong shape for decoder layer {layer_idx}: {activations.shape}"
            )

    def test_all_layers_captured(self, whisper_model, small_audio_features):
        """Test that all specified layers are captured."""
        all_encoder_layers = list(range(4))  # Whisper-tiny has 4 encoder layers
        all_decoder_layers = list(range(4))  # Whisper-tiny has 4 decoder layers

        extractor = WhisperActivationExtractor(
            model=whisper_model,
            encoder_layers=all_encoder_layers,
            decoder_layers=all_decoder_layers,
        )

        with torch.no_grad(), extractor:
            encoder_output = whisper_model.model.encoder(small_audio_features)
            decoder_input_ids = torch.full(
                (1, 1),
                whisper_model.config.decoder_start_token_id,
                dtype=torch.long,
            )
            _ = whisper_model.model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_output.last_hidden_state,
            )

        # Verify all encoder layers captured
        for layer_idx in all_encoder_layers:
            acts = extractor.cache.get_encoder_activations(layer_idx)
            assert acts is not None, f"Missing encoder layer {layer_idx}"

        # Verify all decoder layers captured
        for layer_idx in all_decoder_layers:
            acts = extractor.cache.get_decoder_activations(layer_idx)
            assert acts is not None, f"Missing decoder layer {layer_idx}"

    def test_layer_norm_application(self, whisper_model, small_audio_features):
        """Test that layer norm is correctly applied when requested."""
        # Get activations WITH layer norm
        extractor_with_ln = WhisperActivationExtractor(
            model=whisper_model,
            encoder_layers=[3],  # Last layer
            decoder_layers=[],
            apply_layer_norm=True,
        )

        with torch.no_grad(), extractor_with_ln:
            encoder_output = whisper_model.model.encoder(small_audio_features)

        with_ln = extractor_with_ln.cache.get_encoder_activations(3)

        # Get activations WITHOUT layer norm
        extractor_without_ln = WhisperActivationExtractor(
            model=whisper_model,
            encoder_layers=[3],
            decoder_layers=[],
            apply_layer_norm=False,
        )

        with torch.no_grad(), extractor_without_ln:
            _ = whisper_model.model.encoder(small_audio_features)

        without_ln = extractor_without_ln.cache.get_encoder_activations(3)

        # They should be different (layer norm changes the values)
        assert not torch.allclose(with_ln, without_ln), (
            "Layer norm should change the activations"
        )

        # Verify the layer norm was applied correctly by doing it manually
        layer_norm = whisper_model.model.encoder.layer_norm
        manually_normed = layer_norm(without_ln)
        assert torch.allclose(with_ln, manually_normed, atol=1e-5), (
            "Layer norm application doesn't match manual application"
        )

    def test_activations_match_manual_extraction(self, whisper_model, small_audio_features):
        """Test that hook activations match manual layer access.

        This is critical for ensuring consistency with SAELens/TransformerLens conventions.
        """
        # Get activations via hooks
        extractor = WhisperActivationExtractor(
            model=whisper_model,
            encoder_layers=[2],
            decoder_layers=[],
            apply_layer_norm=False,  # Don't apply LN for direct comparison
        )

        with torch.no_grad(), extractor:
            _ = whisper_model.model.encoder(small_audio_features)

        hook_activations = extractor.cache.get_encoder_activations(2)

        # Get activations manually by running up to that layer
        with torch.no_grad():
            # Run through embedding
            hidden = whisper_model.model.encoder.conv1(small_audio_features)
            hidden = torch.nn.functional.gelu(hidden)
            hidden = whisper_model.model.encoder.conv2(hidden)
            hidden = torch.nn.functional.gelu(hidden)
            hidden = hidden.permute(0, 2, 1)

            # Add positional embedding
            positions = whisper_model.model.encoder.embed_positions.weight[:hidden.shape[1]]
            hidden = hidden + positions

            # Run through layers 0, 1, 2
            for i in range(3):
                layer = whisper_model.model.encoder.layers[i]
                # Whisper encoder layers need attention_mask and layer_head_mask
                layer_output = layer(hidden, attention_mask=None, layer_head_mask=None)
                hidden = layer_output[0] if isinstance(layer_output, tuple) else layer_output

        # For layer 2, the output should match what we got from hooks
        manual_activations = hidden

        assert hook_activations.shape == manual_activations.shape, (
            f"Shape mismatch: hook={hook_activations.shape}, manual={manual_activations.shape}"
        )
        assert torch.allclose(hook_activations, manual_activations, atol=1e-4), (
            "Hook activations don't match manual extraction"
        )

    def test_cache_clear_between_batches(self, whisper_model, small_audio_features):
        """Test that cache can be cleared between batches."""
        extractor = WhisperActivationExtractor(
            model=whisper_model,
            encoder_layers=[0],
            decoder_layers=[],
        )

        # First batch
        with torch.no_grad(), extractor:
            _ = whisper_model.model.encoder(small_audio_features)

        first_batch = extractor.cache.get_encoder_activations(0)
        assert first_batch is not None

        # Clear cache
        extractor.clear_cache()
        assert extractor.cache.get_encoder_activations(0) is None

        # Second batch (slightly different input)
        different_input = small_audio_features + 0.1

        extractor.register_hooks()
        with torch.no_grad():
            _ = whisper_model.model.encoder(different_input)
        extractor.remove_hooks()

        second_batch = extractor.cache.get_encoder_activations(0)
        assert second_batch is not None
        assert not torch.allclose(first_batch, second_batch), (
            "Different inputs should give different activations"
        )

    def test_multiple_batches_accumulate(self, whisper_model, small_audio_features):
        """Test that multiple forward passes accumulate in cache."""
        extractor = WhisperActivationExtractor(
            model=whisper_model,
            encoder_layers=[0],
            decoder_layers=[],
        )

        with torch.no_grad(), extractor:
            # Run two batches
            _ = whisper_model.model.encoder(small_audio_features)
            _ = whisper_model.model.encoder(small_audio_features + 0.1)

        activations = extractor.cache.get_encoder_activations(0)

        # Should have 2 batches worth of activations
        # Each batch: [1, 1500, 384], so concatenated: [2, 1500, 384]
        assert activations.shape[0] == 2, (
            f"Expected 2 batches, got shape {activations.shape}"
        )


class TestExtractFeaturesBatch:
    """Tests for the extract_features_batch helper function."""

    def test_basic_extraction(self, whisper_model, small_audio_features):
        """Test basic feature extraction."""
        results = extract_features_batch(
            model=whisper_model,
            input_features=small_audio_features,
            encoder_layers=[0, 1],
            decoder_layers=[0],
            apply_layer_norm=True,
        )

        assert "encoder" in results
        assert "decoder" in results
        assert 0 in results["encoder"]
        assert 1 in results["encoder"]
        assert 0 in results["decoder"]

    def test_extraction_shapes(self, whisper_model, dummy_audio_features):
        """Test that extracted features have correct shapes."""
        batch_size = dummy_audio_features.shape[0]

        results = extract_features_batch(
            model=whisper_model,
            input_features=dummy_audio_features,
            encoder_layers=[0, 3],
            decoder_layers=[0, 3],
        )

        # Encoder: [batch, 1500, 384]
        for layer_idx in [0, 3]:
            assert results["encoder"][layer_idx].shape == (batch_size, 1500, 384)

        # Decoder with 1 token: [batch, 1, 384]
        for layer_idx in [0, 3]:
            assert results["decoder"][layer_idx].shape == (batch_size, 1, 384)

    def test_empty_layer_lists(self, whisper_model, small_audio_features):
        """Test extraction with empty layer lists."""
        results = extract_features_batch(
            model=whisper_model,
            input_features=small_audio_features,
            encoder_layers=[],
            decoder_layers=[],
        )

        assert results["encoder"] == {}
        assert results["decoder"] == {}

    def test_encoder_only(self, whisper_model, small_audio_features):
        """Test extraction of only encoder layers."""
        results = extract_features_batch(
            model=whisper_model,
            input_features=small_audio_features,
            encoder_layers=[0, 1, 2, 3],
            decoder_layers=[],
        )

        assert len(results["encoder"]) == 4
        assert results["decoder"] == {}

    def test_decoder_only(self, whisper_model, small_audio_features):
        """Test extraction of only decoder layers."""
        results = extract_features_batch(
            model=whisper_model,
            input_features=small_audio_features,
            encoder_layers=[],
            decoder_layers=[0, 1, 2, 3],
        )

        assert results["encoder"] == {}
        assert len(results["decoder"]) == 4


class TestFlattenActivations:
    """Tests for the flatten_activations function."""

    def test_flatten_encoder(self):
        """Test flattening encoder activations."""
        # Encoder: [batch, seq_len, hidden]
        batch_size, seq_len, hidden = 4, 1500, 384
        activations = torch.randn(batch_size, seq_len, hidden)

        flattened = flatten_activations(activations, "encoder")

        assert flattened.shape == (batch_size * seq_len, hidden)

        # Verify the flattening is correct (row-major order)
        for b in range(batch_size):
            for s in range(seq_len):
                expected = activations[b, s]
                actual = flattened[b * seq_len + s]
                assert torch.allclose(expected, actual), (
                    f"Mismatch at batch={b}, seq={s}"
                )

    def test_flatten_decoder(self):
        """Test flattening decoder activations."""
        # Decoder: [batch, seq_len, hidden]
        batch_size, seq_len, hidden = 4, 1, 384
        activations = torch.randn(batch_size, seq_len, hidden)

        flattened = flatten_activations(activations, "decoder")

        assert flattened.shape == (batch_size * seq_len, hidden)

    def test_flatten_preserves_values(self):
        """Test that flattening preserves all values."""
        activations = torch.randn(2, 10, 384)

        flattened = flatten_activations(activations, "encoder")

        # Check that we can recover the original
        reshaped = flattened.view(2, 10, 384)
        assert torch.allclose(activations, reshaped)


class TestHooksMatchTransformerLensConvention:
    """Tests to verify hooks follow TransformerLens/SAELens conventions.

    SAELens and TransformerLens capture residual stream activations AFTER
    each layer's full computation. Our hooks should match this behavior.
    """

    def test_captures_full_layer_output(self, whisper_model, small_audio_features):
        """Test that hooks capture the full layer output, not intermediate states."""
        extractor = WhisperActivationExtractor(
            model=whisper_model,
            encoder_layers=[0],
            decoder_layers=[],
            apply_layer_norm=False,
        )

        with torch.no_grad(), extractor:
            _ = whisper_model.model.encoder(small_audio_features)

        hook_output = extractor.cache.get_encoder_activations(0)

        # The hook should capture the output AFTER the full layer computation
        # (attention + FFN + residual connections)
        # This is what SAELens expects

        # Manually run through layer 0 to verify
        with torch.no_grad():
            hidden = whisper_model.model.encoder.conv1(small_audio_features)
            hidden = torch.nn.functional.gelu(hidden)
            hidden = whisper_model.model.encoder.conv2(hidden)
            hidden = torch.nn.functional.gelu(hidden)
            hidden = hidden.permute(0, 2, 1)
            positions = whisper_model.model.encoder.embed_positions.weight[:hidden.shape[1]]
            hidden = hidden + positions

            # Run through layer 0 with required arguments
            layer = whisper_model.model.encoder.layers[0]
            manual_output = layer(hidden, attention_mask=None, layer_head_mask=None)
            if isinstance(manual_output, tuple):
                manual_output = manual_output[0]

        assert torch.allclose(hook_output, manual_output, atol=1e-4), (
            "Hook output should match full layer output"
        )

    def test_layer_order_is_correct(self, whisper_model, small_audio_features):
        """Test that layer indexing matches actual layer order."""
        extractor = WhisperActivationExtractor(
            model=whisper_model,
            encoder_layers=[0, 1, 2, 3],
            decoder_layers=[],
            apply_layer_norm=False,
        )

        with torch.no_grad(), extractor:
            _ = whisper_model.model.encoder(small_audio_features)

        # Each layer should have different activations
        # (unless by extreme coincidence)
        layer_outputs = [
            extractor.cache.get_encoder_activations(i) for i in range(4)
        ]

        for i in range(3):
            assert not torch.allclose(layer_outputs[i], layer_outputs[i + 1], atol=1e-3), (
                f"Layer {i} and {i+1} have suspiciously similar outputs"
            )

        # Later layers should have processed more information
        # Their outputs should be progressively different from layer 0
        diff_from_layer0 = [
            (layer_outputs[i] - layer_outputs[0]).abs().mean().item()
            for i in range(4)
        ]

        # Differences should generally increase with layer depth
        # (not a strict requirement but a sanity check)
        assert diff_from_layer0[0] < diff_from_layer0[3], (
            "Later layers should differ more from layer 0"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
