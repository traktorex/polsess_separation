"""Tests for ASR evaluation module: metrics, speaker assignment, datasets."""

import csv
import pytest
import torch
import torchaudio

from asr.metrics import (
    word_error_rate, character_error_rate, compute_metrics, aggregate_metrics,
)
from asr.evaluate_asr import assign_speakers
from asr.dataset import RealMDataset, LibriSpeechMixDataset


# --- Metrics tests ---


class TestWER:
    """Word Error Rate computation."""

    def test_identical_strings(self):
        wer, dist, ref_len = word_error_rate("hello world", "hello world")
        assert wer == 0.0
        assert dist == 0

    def test_completely_different(self):
        wer, dist, ref_len = word_error_rate("foo bar", "hello world")
        assert wer == 100.0
        assert dist == 2
        assert ref_len == 2

    def test_partial_overlap(self):
        wer, dist, ref_len = word_error_rate("the cat sat", "the dog sat")
        assert dist == 1  # one substitution
        assert ref_len == 3
        assert wer == pytest.approx(100.0 / 3, abs=0.1)

    def test_insertion(self):
        wer, dist, ref_len = word_error_rate("the big cat sat", "the cat sat")
        assert dist == 1  # one insertion
        assert ref_len == 3

    def test_deletion(self):
        wer, dist, ref_len = word_error_rate("the sat", "the cat sat")
        assert dist == 1  # one deletion
        assert ref_len == 3

    def test_case_insensitive(self):
        """WER uppercases both strings before comparing."""
        wer, _, _ = word_error_rate("Hello World", "hello world")
        assert wer == 0.0

    def test_empty_reference(self):
        wer, _, ref_len = word_error_rate("some text", "")
        assert ref_len == 0
        assert wer == 0.0  # edge case: 0/0 returns 0


class TestCER:
    """Character Error Rate computation."""

    def test_identical_strings(self):
        cer, dist, ref_len = character_error_rate("hello", "hello")
        assert cer == 0.0

    def test_one_char_difference(self):
        cer, dist, ref_len = character_error_rate("cat", "bat")
        assert dist == 1
        assert ref_len == 3

    def test_ignores_spaces(self):
        """CER strips spaces before comparing."""
        cer, _, _ = character_error_rate("h e l l o", "hello")
        assert cer == 0.0


class TestComputeMetrics:
    """Combined WER + CER computation."""

    def test_returns_all_keys(self):
        result = compute_metrics("hello world", "hello world")
        expected_keys = {"wer", "wer_distance", "wer_ref_len",
                         "cer", "cer_distance", "cer_ref_len",
                         "hypothesis", "reference"}
        assert set(result.keys()) == expected_keys

    def test_perfect_match(self):
        result = compute_metrics("hello world", "hello world")
        assert result["wer"] == 0.0
        assert result["cer"] == 0.0


class TestAggregateMetrics:
    """Micro-averaged metric aggregation."""

    def test_single_sample(self):
        results = [compute_metrics("the cat", "the dog")]
        agg = aggregate_metrics(results)
        assert agg["total_samples"] == 1
        assert agg["wer"] == pytest.approx(50.0)  # 1 error / 2 words

    def test_micro_averaging(self):
        """Micro-averaging weights by reference length, not by sample."""
        r1 = compute_metrics("a", "a b c d e")  # 4 deletions / 5 ref words
        r2 = compute_metrics("x", "x")           # 0 errors / 1 ref word
        agg = aggregate_metrics([r1, r2])

        # Global: 4 errors / 6 total ref words = 66.67%
        assert agg["wer"] == pytest.approx(4.0 / 6.0 * 100, abs=0.1)


# --- Speaker assignment tests ---


class TestAssignSpeakers:
    """Permutation-invariant speaker assignment."""

    def test_correct_assignment(self):
        """When hyp1 matches ref1 and hyp2 matches ref2."""
        m_s1, m_s2 = assign_speakers(
            "the cat sat", "a dog ran",
            "the cat sat", "a dog ran",
        )
        assert m_s1["wer"] == 0.0
        assert m_s2["wer"] == 0.0

    def test_swapped_assignment(self):
        """When hyp1 matches ref2 and hyp2 matches ref1 (should swap)."""
        m_s1, m_s2 = assign_speakers(
            "a dog ran", "the cat sat",
            "the cat sat", "a dog ran",
        )
        assert m_s1["wer"] == 0.0
        assert m_s2["wer"] == 0.0

    def test_partial_match_picks_better(self):
        """Pick the assignment with lower total WER."""
        # hyp1 is closer to ref1, hyp2 is closer to ref2
        m_s1, m_s2 = assign_speakers(
            "the cat sat on mat", "a big dog ran fast",
            "the cat sat on a mat", "a big dog ran very fast",
        )
        # This should choose the direct assignment (not swapped)
        total_wer = m_s1["wer"] + m_s2["wer"]
        # Swapped would give much higher WER
        assert total_wer < 100.0


# --- RealMDataset tests ---


class TestRealMDataset:
    """RealMDataset with mock data."""

    @pytest.fixture
    def mock_realm_dir(self, tmp_path):
        """Create a minimal REAL-M directory structure with mock data."""
        audio_dir = tmp_path / "audio_files_converted_8000Hz"
        trans_dir = tmp_path / "transcriptions"
        audio_dir.mkdir()
        trans_dir.mkdir()

        # Newer session: BcArV
        session = "BcArV"
        (audio_dir / session).mkdir()

        # Create dummy WAV files
        for i in range(3):
            filename = f"mixture_{i}_{session}.wav"
            audio = torch.randn(1, 8000)  # 1 second at 8kHz
            torchaudio.save(audio_dir / session / filename, audio, 8000)

        # Write CSV (newer format: index, sentence1, sentence2, filename)
        csv_path = trans_dir / f"{session}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "sentence1", "sentence2", "filename"])
            writer.writerow([0, "hello world", "goodbye moon", f"mixture_0_{session}.wav"])
            writer.writerow([1, "foo bar", "baz qux", f"mixture_1_{session}.wav"])
            writer.writerow([2, "alpha beta", "gamma delta", f"mixture_2_{session}.wav"])

        return tmp_path

    @pytest.fixture
    def mock_early_realm_dir(self, tmp_path):
        """Create mock REAL-M dir with early collection CSV format."""
        audio_dir = tmp_path / "audio_files_converted_8000Hz"
        trans_dir = tmp_path / "transcriptions"
        audio_dir.mkdir()
        trans_dir.mkdir()

        session = "early_collection1"
        (audio_dir / session).mkdir()

        # Early format: filenames are .mp3 in CSV, .wav on disk
        for i in range(2):
            filename = f"early_collection1_mixture_{i}.wav"
            audio = torch.randn(1, 8000)
            torchaudio.save(audio_dir / session / filename, audio, 8000)

        # Write CSV (early format: sentence1, sentence2, WorkerId, filename)
        csv_path = trans_dir / f"{session}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sentence1", "sentence2", "WorkerId", "filename"])
            writer.writerow(["hello world", "goodbye", "WORKER1",
                             "early_collection1_mixture_0.mp3"])
            writer.writerow(["foo bar", "baz qux", "WORKER2",
                             "early_collection1_mixture_1.mp3"])

        return tmp_path

    def test_loads_newer_csv_format(self, mock_realm_dir):
        ds = RealMDataset(dataset_root=str(mock_realm_dir))
        assert len(ds) == 3

    def test_sample_keys(self, mock_realm_dir):
        ds = RealMDataset(dataset_root=str(mock_realm_dir))
        sample = ds[0]
        assert "mix" in sample
        assert "transcription1" in sample
        assert "transcription2" in sample
        assert "sample_id" in sample
        assert "session_id" in sample
        assert "sample_rate" in sample

    def test_audio_shape(self, mock_realm_dir):
        ds = RealMDataset(dataset_root=str(mock_realm_dir))
        sample = ds[0]
        assert sample["mix"].shape == torch.Size([1, 8000])
        assert sample["sample_rate"] == 8000

    def test_transcriptions_correct(self, mock_realm_dir):
        ds = RealMDataset(dataset_root=str(mock_realm_dir))
        sample = ds[0]
        assert sample["transcription1"] == "hello world"
        assert sample["transcription2"] == "goodbye moon"

    def test_loads_early_csv_format(self, mock_early_realm_dir):
        """Early collections: .mp3 filenames in CSV map to .wav on disk."""
        ds = RealMDataset(dataset_root=str(mock_early_realm_dir))
        assert len(ds) == 2
        sample = ds[0]
        assert sample["transcription1"] == "hello world"

    def test_max_samples(self, mock_realm_dir):
        ds = RealMDataset(dataset_root=str(mock_realm_dir), max_samples=2)
        assert len(ds) == 2

    def test_skips_missing_audio(self, tmp_path):
        """Entries with missing audio files should be skipped."""
        audio_dir = tmp_path / "audio_files_converted_8000Hz"
        trans_dir = tmp_path / "transcriptions"
        audio_dir.mkdir()
        trans_dir.mkdir()

        session = "TestSession"
        (audio_dir / session).mkdir()

        # Only create 1 of 2 audio files
        audio = torch.randn(1, 8000)
        torchaudio.save(audio_dir / session / "mixture_0_TestSession.wav", audio, 8000)

        csv_path = trans_dir / f"{session}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "sentence1", "sentence2", "filename"])
            writer.writerow([0, "exists", "audio", "mixture_0_TestSession.wav"])
            writer.writerow([1, "missing", "audio", "mixture_1_TestSession.wav"])

        ds = RealMDataset(dataset_root=str(tmp_path))
        assert len(ds) == 1

    def test_raises_on_missing_audio_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            RealMDataset(dataset_root=str(tmp_path / "nonexistent"))


# --- LibriSpeechMixDataset tests ---


class TestLibriSpeechMixDataset:
    """Basic validation for LibriSpeechMixDataset."""

    def test_raises_on_invalid_split(self):
        with pytest.raises(ValueError, match="Invalid split"):
            LibriSpeechMixDataset(split="invalid")

    def test_raises_on_missing_metadata(self, tmp_path):
        (tmp_path / "dev").mkdir()
        with pytest.raises(FileNotFoundError, match="Metadata file not found"):
            LibriSpeechMixDataset(dataset_root=str(tmp_path), split="dev")
