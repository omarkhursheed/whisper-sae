"""Tests for feature interpretation and analysis tools."""

import tempfile
from pathlib import Path

import pytest
import torch

from whisper_sae.analysis import (
    FeatureActivation,
    FeatureInterpretation,
    FeatureReport,
    TopKTracker,
    AudioClipConfig,
    AudioClipExtractor,
    create_indexed_audio_loader,
)


class TestFeatureActivation:
    """Tests for FeatureActivation dataclass."""

    def test_creation(self):
        """Test basic creation."""
        act = FeatureActivation(
            feature_idx=42,
            activation_value=0.75,
            sample_idx=100,
            position_idx=50,
        )
        assert act.feature_idx == 42
        assert act.activation_value == 0.75
        assert act.sample_idx == 100
        assert act.position_idx == 50

    def test_optional_fields(self):
        """Test optional fields have defaults."""
        act = FeatureActivation(
            feature_idx=0,
            activation_value=1.0,
            sample_idx=0,
            position_idx=0,
        )
        assert act.timestamp_ms is None
        assert act.transcription is None
        assert act.audio_path is None
        assert act.metadata == {}

    def test_to_dict(self):
        """Test serialization to dict."""
        act = FeatureActivation(
            feature_idx=42,
            activation_value=0.75,
            sample_idx=100,
            position_idx=50,
            timestamp_ms=500.0,
            transcription="hello world",
        )
        d = act.to_dict()
        assert d["feature_idx"] == 42
        assert d["activation_value"] == 0.75
        assert d["timestamp_ms"] == 500.0
        assert d["transcription"] == "hello world"

    def test_from_dict(self):
        """Test deserialization from dict."""
        d = {
            "feature_idx": 42,
            "activation_value": 0.75,
            "sample_idx": 100,
            "position_idx": 50,
            "timestamp_ms": 500.0,
            "transcription": "hello world",
            "transcription_context": None,
            "audio_path": None,
            "metadata": {"speaker_id": "123"},
        }
        act = FeatureActivation.from_dict(d)
        assert act.feature_idx == 42
        assert act.metadata["speaker_id"] == "123"


class TestTopKTracker:
    """Tests for TopKTracker."""

    def test_initialization(self):
        """Test tracker initializes correctly."""
        tracker = TopKTracker(num_features=128, k=10)
        assert tracker.num_features == 128
        assert tracker.k == 10
        assert tracker.total_activations == 0
        assert tracker.samples_processed == 0

    def test_update_single_sample(self):
        """Test updating with a single sample."""
        tracker = TopKTracker(num_features=64, k=5)

        # Create activations [batch=1, seq=1, features=64]
        activations = torch.zeros(1, 64)
        activations[0, 10] = 0.5  # Feature 10 activates
        activations[0, 20] = 0.8  # Feature 20 activates

        tracker.update(activations, sample_indices=[0])

        assert tracker.samples_processed == 1
        assert tracker.total_activations == 2

        # Check feature 10
        examples = tracker.get_top_examples(10)
        assert len(examples) == 1
        assert examples[0].activation_value == pytest.approx(0.5)

        # Check feature 20
        examples = tracker.get_top_examples(20)
        assert len(examples) == 1
        assert examples[0].activation_value == pytest.approx(0.8)

    def test_update_batch(self):
        """Test updating with a batch of samples."""
        tracker = TopKTracker(num_features=64, k=10)

        # Batch of 4 samples
        activations = torch.zeros(4, 64)
        activations[0, 5] = 0.9
        activations[1, 5] = 0.7
        activations[2, 5] = 0.5
        activations[3, 5] = 0.3

        tracker.update(activations, sample_indices=[0, 1, 2, 3])

        assert tracker.samples_processed == 4

        # Feature 5 should have 4 examples
        examples = tracker.get_top_examples(5)
        assert len(examples) == 4
        # Should be sorted by activation descending
        assert examples[0].activation_value == pytest.approx(0.9)
        assert examples[1].activation_value == pytest.approx(0.7)

    def test_top_k_limit(self):
        """Test that tracker keeps only top k examples."""
        tracker = TopKTracker(num_features=64, k=3)

        # Add 5 activations for feature 0
        for i in range(5):
            activations = torch.zeros(1, 64)
            activations[0, 0] = float(i) / 10  # 0.0, 0.1, 0.2, 0.3, 0.4
            tracker.update(activations, sample_indices=[i])

        examples = tracker.get_top_examples(0)
        assert len(examples) == 3
        # Should have top 3: 0.4, 0.3, 0.2
        assert examples[0].activation_value == pytest.approx(0.4)
        assert examples[1].activation_value == pytest.approx(0.3)
        assert examples[2].activation_value == pytest.approx(0.2)

    def test_update_with_transcriptions(self):
        """Test updating with transcription metadata."""
        tracker = TopKTracker(num_features=64, k=5)

        activations = torch.zeros(2, 64)
        activations[0, 0] = 0.5
        activations[1, 0] = 0.7

        tracker.update(
            activations,
            sample_indices=[0, 1],
            transcriptions=["hello world", "foo bar"],
        )

        examples = tracker.get_top_examples(0)
        assert examples[0].transcription == "foo bar"  # Higher activation
        assert examples[1].transcription == "hello world"

    def test_update_with_sequence(self):
        """Test updating with sequential activations [batch, seq, features]."""
        tracker = TopKTracker(num_features=64, k=10)

        # Batch=1, Seq=5, Features=64
        activations = torch.zeros(1, 5, 64)
        activations[0, 0, 10] = 0.5  # Position 0
        activations[0, 2, 10] = 0.8  # Position 2
        activations[0, 4, 10] = 0.3  # Position 4

        tracker.update(activations, sample_indices=[0])

        examples = tracker.get_top_examples(10)
        assert len(examples) == 3
        # Sorted by activation
        assert examples[0].position_idx == 2
        assert examples[0].activation_value == pytest.approx(0.8)

    def test_timestamp_calculation(self):
        """Test timestamp is calculated correctly (10ms per frame)."""
        tracker = TopKTracker(num_features=64, k=5)

        activations = torch.zeros(1, 100, 64)  # 100 frames
        activations[0, 50, 0] = 1.0  # Activate at frame 50

        tracker.update(activations, sample_indices=[0])

        examples = tracker.get_top_examples(0)
        assert examples[0].position_idx == 50
        assert examples[0].timestamp_ms == 500.0  # 50 * 10ms

    def test_get_feature_stats(self):
        """Test feature statistics computation."""
        tracker = TopKTracker(num_features=64, k=5)

        # Add varying activations for feature 0
        activations = torch.zeros(5, 64)
        activations[:, 0] = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])

        tracker.update(activations, sample_indices=list(range(5)))

        stats = tracker.get_feature_stats()
        assert stats[0]["num_examples"] == 5
        assert stats[0]["max_activation"] == pytest.approx(0.5)
        assert stats[0]["min_activation"] == pytest.approx(0.1)
        assert stats[0]["mean_activation"] == pytest.approx(0.3)

    def test_save_and_load(self):
        """Test saving and loading tracker state."""
        tracker = TopKTracker(num_features=64, k=5)

        activations = torch.zeros(3, 64)
        activations[0, 10] = 0.5
        activations[1, 10] = 0.7
        activations[2, 20] = 0.9

        tracker.update(
            activations,
            sample_indices=[0, 1, 2],
            transcriptions=["a", "b", "c"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tracker.json"
            tracker.save(path)

            loaded = TopKTracker.load(path)

            assert loaded.num_features == 64
            assert loaded.k == 5
            assert loaded.samples_processed == 3

            examples_10 = loaded.get_top_examples(10)
            assert len(examples_10) == 2
            assert examples_10[0].activation_value == pytest.approx(0.7)


class TestFeatureReport:
    """Tests for FeatureReport."""

    def test_generate_feature_report(self):
        """Test generating a single feature report."""
        tracker = TopKTracker(num_features=64, k=5)

        activations = torch.zeros(3, 64)
        activations[:, 0] = torch.tensor([0.5, 0.7, 0.9])

        tracker.update(
            activations,
            sample_indices=[0, 1, 2],
            transcriptions=["one", "two", "three"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            report = FeatureReport(tracker, output_dir=tmpdir)
            feat_report = report.generate_feature_report(0)

            assert feat_report["feature_idx"] == 0
            assert feat_report["stats"]["num_examples"] == 3
            assert len(feat_report["top_examples"]) == 3
            assert feat_report["top_examples"][0]["activation_value"] == pytest.approx(0.9)

    def test_generate_summary_report(self):
        """Test generating summary report."""
        tracker = TopKTracker(num_features=64, k=5)

        activations = torch.zeros(5, 64)
        activations[0, 0] = 0.9  # Feature 0 has high activation
        activations[1, 10] = 0.5  # Feature 10 has medium
        activations[2, 20] = 0.3  # Feature 20 has low

        tracker.update(activations, sample_indices=list(range(5)))

        with tempfile.TemporaryDirectory() as tmpdir:
            report = FeatureReport(tracker, output_dir=tmpdir)
            summary = report.generate_summary_report(top_n=10)

            assert summary["num_features"] == 64
            assert summary["samples_processed"] == 5
            # Top feature should be 0 with highest activation
            assert summary["top_features"][0]["feature_idx"] == 0

    def test_add_interpretation(self):
        """Test adding manual interpretation."""
        tracker = TopKTracker(num_features=64, k=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            report = FeatureReport(tracker, output_dir=tmpdir)
            report.add_interpretation(
                feature_idx=42,
                category="phoneme",
                description="Responds to /s/ sounds",
                confidence=0.8,
                evidence=["High activation on 'snake', 'sister'"],
            )

            assert 42 in report.interpretations
            assert report.interpretations[42].category == "phoneme"

    def test_save_reports(self):
        """Test saving all reports."""
        tracker = TopKTracker(num_features=64, k=5)

        activations = torch.randn(10, 64).abs()
        tracker.update(activations, sample_indices=list(range(10)))

        with tempfile.TemporaryDirectory() as tmpdir:
            report = FeatureReport(tracker, output_dir=tmpdir)
            report.save_reports(top_n=5)

            # Check files were created
            assert (Path(tmpdir) / "summary.json").exists()
            assert (Path(tmpdir) / "tracker_state.json").exists()
            assert (Path(tmpdir) / "features").is_dir()


class TestAudioClipExtractor:
    """Tests for AudioClipExtractor."""

    @pytest.fixture
    def mock_audio_loader(self):
        """Create a mock audio loader that returns sine waves."""
        def load_audio(sample_idx: int) -> torch.Tensor:
            # Generate 3 seconds of audio at 16kHz
            duration = 3.0
            sample_rate = 16000
            t = torch.linspace(0, duration, int(duration * sample_rate))
            # Different frequency for each sample
            freq = 440 + sample_idx * 100
            audio = torch.sin(2 * 3.14159 * freq * t)
            return audio

        return load_audio

    def test_config_defaults(self):
        """Test AudioClipConfig defaults."""
        config = AudioClipConfig()
        assert config.sample_rate == 16000
        assert config.samples_per_frame == 160
        assert config.clip_duration_ms == 1000.0

    def test_extract_clip(self, mock_audio_loader):
        """Test extracting a single clip."""
        tracker = TopKTracker(num_features=64, k=5)

        # Add an activation
        activations = torch.zeros(1, 100, 64)  # 100 frames = 1 second
        activations[0, 50, 0] = 1.0  # Activation at 500ms

        tracker.update(activations, sample_indices=[0])

        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = AudioClipExtractor(
                tracker=tracker,
                audio_loader=mock_audio_loader,
                output_dir=tmpdir,
            )

            activation = tracker.get_top_examples(0)[0]
            clip = extractor.extract_clip(activation)

            assert clip is not None
            # Should be ~1 second of audio at 16kHz
            assert 15000 <= len(clip) <= 17000

    def test_extract_feature_clips(self, mock_audio_loader):
        """Test extracting all clips for a feature."""
        tracker = TopKTracker(num_features=64, k=5)

        # Add multiple activations
        activations = torch.zeros(3, 100, 64)
        activations[0, 50, 0] = 0.9
        activations[1, 30, 0] = 0.7
        activations[2, 80, 0] = 0.5

        tracker.update(activations, sample_indices=[0, 1, 2])

        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = AudioClipExtractor(
                tracker=tracker,
                audio_loader=mock_audio_loader,
                output_dir=tmpdir,
            )

            clips = extractor.extract_feature_clips(0)

            assert len(clips) == 3
            # Check files were created
            for clip_path in clips:
                assert clip_path.exists()

    def test_extract_with_max_clips(self, mock_audio_loader):
        """Test limiting number of clips extracted."""
        tracker = TopKTracker(num_features=64, k=10)

        activations = torch.zeros(5, 100, 64)
        activations[:, 50, 0] = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5])

        tracker.update(activations, sample_indices=list(range(5)))

        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = AudioClipExtractor(
                tracker=tracker,
                audio_loader=mock_audio_loader,
                output_dir=tmpdir,
            )

            clips = extractor.extract_feature_clips(0, max_clips=2)
            assert len(clips) == 2

    def test_save_manifest(self, mock_audio_loader):
        """Test saving manifest file."""
        tracker = TopKTracker(num_features=64, k=5)

        activations = torch.zeros(2, 100, 64)
        activations[0, 50, 0] = 0.9
        activations[1, 30, 0] = 0.7

        tracker.update(
            activations,
            sample_indices=[0, 1],
            transcriptions=["hello", "world"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            extractor = AudioClipExtractor(
                tracker=tracker,
                audio_loader=mock_audio_loader,
                output_dir=tmpdir,
            )

            extractor.extract_feature_clips(0)
            manifest_path = extractor.save_manifest()

            assert manifest_path.exists()

            import json
            with open(manifest_path) as f:
                manifest = json.load(f)

            assert "config" in manifest
            assert "features" in manifest
            assert "0" in manifest["features"]


class TestCreateIndexedAudioLoader:
    """Tests for create_indexed_audio_loader."""

    def test_load_from_paths(self):
        """Test loading audio from file paths."""
        import soundfile as sf

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test audio files
            paths = []
            for i in range(3):
                path = Path(tmpdir) / f"audio_{i}.wav"
                audio = torch.randn(16000).numpy()  # 1 second
                sf.write(path, audio, 16000)
                paths.append(path)

            loader = create_indexed_audio_loader(paths)

            # Test loading
            audio = loader(0)
            assert audio.shape == (16000,)

            audio = loader(2)
            assert audio.shape == (16000,)

    def test_out_of_range(self):
        """Test error on out of range index."""
        loader = create_indexed_audio_loader([])

        with pytest.raises(IndexError):
            loader(0)
