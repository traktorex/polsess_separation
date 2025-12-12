"""Comprehensive tests for MM-IPC variant loading in PolSESSDataset."""

import pytest
import torch
import torchaudio
import pandas as pd
from pathlib import Path
from datasets.polsess_dataset import PolSESSDataset


@pytest.fixture
def mock_polsess_data(tmp_path):
    """Create mock PolSESS dataset structure with audio files."""
    data_root = tmp_path / "PolSESS_C_in"

    # Create directory structure
    for subset in ["train", "val"]:
        subset_path = data_root / subset
        for folder in ["mix", "clean", "scene", "event", "sp1_reverb", "sp2_reverb", "ev_reverb"]:
            (subset_path / folder).mkdir(parents=True, exist_ok=True)

        # Create mock audio files with distinguishable values
        sr = 8000
        duration = 4.0
        n_samples = int(sr * duration)

        # Create different audio signals for each component
        def create_audio(value, filename, folder):
            audio = torch.ones(1, n_samples) * value
            filepath = subset_path / folder / filename
            torchaudio.save(str(filepath), audio, sr)

        # Sample with reverb (indoor)
        create_audio(1.0, "mix_reverb.wav", "mix")
        create_audio(0.1, "sp1_reverb.wav", "clean")
        create_audio(0.2, "sp2_reverb.wav", "clean")
        create_audio(0.3, "scene_reverb.wav", "scene")
        create_audio(0.4, "event_reverb.wav", "event")
        create_audio(0.01, "sp1_reverb_reverb.wav", "sp1_reverb")
        create_audio(0.02, "sp2_reverb_reverb.wav", "sp2_reverb")
        create_audio(0.04, "ev_reverb_reverb.wav", "ev_reverb")

        # Sample without reverb (outdoor)
        create_audio(2.0, "mix_no_reverb.wav", "mix")
        create_audio(0.5, "sp1_no_reverb.wav", "clean")
        create_audio(0.6, "sp2_no_reverb.wav", "clean")
        create_audio(0.7, "scene_no_reverb.wav", "scene")
        create_audio(0.8, "event_no_reverb.wav", "event")

        # Create CSV metadata
        metadata = pd.DataFrame([
            {
                "mixFile": "mix_reverb.wav",
                "speaker1File": "sp1_reverb.wav",
                "speaker2File": "sp2_reverb.wav",
                "sceneFile": "scene_reverb.wav",
                "eventFile": "event_reverb.wav",
                "reverbForSpeaker1": "sp1_reverb_reverb.wav",
                "reverbForSpeaker2": "sp2_reverb_reverb.wav",
                "reverbForEvent": "ev_reverb_reverb.wav",
            },
            {
                "mixFile": "mix_no_reverb.wav",
                "speaker1File": "sp1_no_reverb.wav",
                "speaker2File": "sp2_no_reverb.wav",
                "sceneFile": "scene_no_reverb.wav",
                "eventFile": "event_no_reverb.wav",
                "reverbForSpeaker1": None,
                "reverbForSpeaker2": None,
                "reverbForEvent": None,
            },
        ])

        csv_path = subset_path / f"corpus_PolSESS_C_in_{subset}_final.csv"
        metadata.to_csv(csv_path, index=False)

    return data_root


class TestLazyLoadFieldsES:
    """Test _lazy_load returns correct fields for ES task."""

    @pytest.mark.parametrize("variant,expected_fields", [
        # Indoor variants (has_reverb=True)
        ("SER", {"mix", "speaker1", "speaker2", "sp2_reverb"}),
        ("SR", {"mix", "speaker1", "speaker2", "sp2_reverb", "event", "ev_reverb"}),
        ("ER", {"mix", "speaker1", "speaker2", "sp2_reverb", "scene"}),
        ("R", {"mix", "speaker1", "speaker2", "sp2_reverb", "scene", "event", "ev_reverb"}),
        ("C", {"mix", "speaker1", "speaker2", "sp2_reverb", "sp1_reverb", "scene", "event", "ev_reverb"}),
    ])
    def test_es_indoor_variants(self, mock_polsess_data, variant, expected_fields):
        """Test ES task with indoor (reverb) variants loads correct fields."""
        dataset = PolSESSDataset(
            data_root=mock_polsess_data,
            subset="train",
            task="ES",
            allowed_variants=[variant],
        )

        # Get sample with reverb (idx=0)
        sample = dataset[0]

        # Check internal _lazy_load output
        row = dataset.metadata.iloc[0]
        has_reverb = pd.notna(row["reverbForSpeaker1"])
        paths = dataset._build_paths(row, has_reverb)
        audio = dataset._lazy_load(paths, variant, has_reverb)

        assert set(audio.keys()) == expected_fields, \
            f"Variant {variant} with ES task should have fields {expected_fields}, got {set(audio.keys())}"

    @pytest.mark.parametrize("variant,expected_fields", [
        # Outdoor variants (has_reverb=False)
        ("SE", {"mix", "speaker1", "speaker2"}),
        ("S", {"mix", "speaker1", "speaker2", "event"}),
        ("E", {"mix", "speaker1", "speaker2", "scene"}),
    ])
    def test_es_outdoor_variants(self, mock_polsess_data, variant, expected_fields):
        """Test ES task with outdoor (no reverb) variants loads correct fields."""
        dataset = PolSESSDataset(
            data_root=mock_polsess_data,
            subset="train",
            task="ES",
            allowed_variants=[variant],
        )

        # Get sample without reverb
        sample = dataset[0]

        # Check internal _lazy_load output
        row = dataset.metadata.iloc[0]
        has_reverb = pd.notna(row["reverbForSpeaker1"])
        paths = dataset._build_paths(row, has_reverb)
        audio = dataset._lazy_load(paths, variant, has_reverb)

        assert set(audio.keys()) == expected_fields, \
            f"Variant {variant} with ES task should have fields {expected_fields}, got {set(audio.keys())}"

    def test_es_outdoor_variant_c(self, mock_polsess_data):
        """Test ES task with C variant (outdoor sample) loads correct fields."""
        dataset = PolSESSDataset(
            data_root=mock_polsess_data,
            subset="train",
            task="ES",
            allowed_variants=["C"],
        )

        # For C variant, both indoor and outdoor samples are included
        # Find the outdoor sample (idx=1 in original metadata)
        for idx in range(len(dataset.metadata)):
            row = dataset.metadata.iloc[idx]
            has_reverb = pd.notna(row["reverbForSpeaker1"])
            if not has_reverb:
                paths = dataset._build_paths(row, has_reverb)
                audio = dataset._lazy_load(paths, "C", has_reverb)
                expected_fields = {"mix", "speaker1", "speaker2", "scene", "event"}
                assert set(audio.keys()) == expected_fields, \
                    f"Variant C with ES task (outdoor) should have fields {expected_fields}, got {set(audio.keys())}"
                break
        else:
            pytest.fail("No outdoor sample found for C variant test")


class TestLazyLoadFieldsEB:
    """Test _lazy_load returns correct fields for EB task."""

    @pytest.mark.parametrize("variant,expected_fields", [
        # Indoor variants (has_reverb=True)
        ("SER", {"mix", "speaker1", "speaker2"}),
        ("SR", {"mix", "speaker1", "speaker2", "event", "ev_reverb"}),
        ("ER", {"mix", "speaker1", "speaker2", "scene"}),
        ("R", {"mix", "speaker1", "speaker2", "scene", "event", "ev_reverb"}),
        ("C", {"mix", "speaker1", "speaker2", "sp1_reverb", "sp2_reverb", "scene", "event", "ev_reverb"}),
    ])
    def test_eb_indoor_variants(self, mock_polsess_data, variant, expected_fields):
        """Test EB task with indoor (reverb) variants loads correct fields."""
        dataset = PolSESSDataset(
            data_root=mock_polsess_data,
            subset="train",
            task="EB",
            allowed_variants=[variant],
        )

        # Get sample with reverb (idx=0)
        sample = dataset[0]

        # Check internal _lazy_load output
        row = dataset.metadata.iloc[0]
        has_reverb = pd.notna(row["reverbForSpeaker1"])
        paths = dataset._build_paths(row, has_reverb)
        audio = dataset._lazy_load(paths, variant, has_reverb)

        assert set(audio.keys()) == expected_fields, \
            f"Variant {variant} with EB task should have fields {expected_fields}, got {set(audio.keys())}"
        assert "speaker2" in audio, "EB task should keep speaker2"

    @pytest.mark.parametrize("variant,expected_fields", [
        # Outdoor variants (has_reverb=False)
        ("SE", {"mix", "speaker1", "speaker2"}),
        ("S", {"mix", "speaker1", "speaker2", "event"}),
        ("E", {"mix", "speaker1", "speaker2", "scene"}),
    ])
    def test_eb_outdoor_variants(self, mock_polsess_data, variant, expected_fields):
        """Test EB task with outdoor (no reverb) variants loads correct fields."""
        dataset = PolSESSDataset(
            data_root=mock_polsess_data,
            subset="train",
            task="EB",
            allowed_variants=[variant],
        )

        # Get sample without reverb
        sample = dataset[0]

        # Check internal _lazy_load output
        row = dataset.metadata.iloc[0]
        has_reverb = pd.notna(row["reverbForSpeaker1"])
        paths = dataset._build_paths(row, has_reverb)
        audio = dataset._lazy_load(paths, variant, has_reverb)

        assert set(audio.keys()) == expected_fields, \
            f"Variant {variant} with EB task should have fields {expected_fields}, got {set(audio.keys())}"

    def test_eb_outdoor_variant_c(self, mock_polsess_data):
        """Test EB task with C variant (outdoor sample) loads correct fields."""
        dataset = PolSESSDataset(
            data_root=mock_polsess_data,
            subset="train",
            task="EB",
            allowed_variants=["C"],
        )

        # For C variant, both indoor and outdoor samples are included
        # Find the outdoor sample
        for idx in range(len(dataset.metadata)):
            row = dataset.metadata.iloc[idx]
            has_reverb = pd.notna(row["reverbForSpeaker1"])
            if not has_reverb:
                paths = dataset._build_paths(row, has_reverb)
                audio = dataset._lazy_load(paths, "C", has_reverb)
                expected_fields = {"mix", "speaker1", "speaker2", "scene", "event"}
                assert set(audio.keys()) == expected_fields, \
                    f"Variant C with EB task (outdoor) should have fields {expected_fields}, got {set(audio.keys())}"
                break
        else:
            pytest.fail("No outdoor sample found for C variant test")


class TestLazyLoadFieldsSB:
    """Test _lazy_load returns correct fields for SB task."""

    @pytest.mark.parametrize("variant,expected_fields", [
        # Indoor variants (has_reverb=True)
        ("SER", {"mix", "speaker1", "speaker2"}),
        ("SR", {"mix", "speaker1", "speaker2", "event", "ev_reverb"}),
        ("ER", {"mix", "speaker1", "speaker2", "scene"}),
        ("R", {"mix", "speaker1", "speaker2", "scene", "event", "ev_reverb"}),
        ("C", {"mix", "speaker1", "speaker2", "sp1_reverb", "sp2_reverb", "scene", "event", "ev_reverb"}),
    ])
    def test_sb_indoor_variants(self, mock_polsess_data, variant, expected_fields):
        """Test SB task with indoor (reverb) variants loads correct fields."""
        dataset = PolSESSDataset(
            data_root=mock_polsess_data,
            subset="train",
            task="SB",
            allowed_variants=[variant],
        )

        # Get sample with reverb (idx=0)
        sample = dataset[0]

        # Check internal _lazy_load output
        row = dataset.metadata.iloc[0]
        has_reverb = pd.notna(row["reverbForSpeaker1"])
        paths = dataset._build_paths(row, has_reverb)
        audio = dataset._lazy_load(paths, variant, has_reverb)

        assert set(audio.keys()) == expected_fields, \
            f"Variant {variant} with SB task should have fields {expected_fields}, got {set(audio.keys())}"
        assert "speaker2" in audio, "SB task should keep speaker2"

    @pytest.mark.parametrize("variant,expected_fields", [
        # Outdoor variants (has_reverb=False)
        ("SE", {"mix", "speaker1", "speaker2"}),
        ("S", {"mix", "speaker1", "speaker2", "event"}),
        ("E", {"mix", "speaker1", "speaker2", "scene"}),
    ])
    def test_sb_outdoor_variants(self, mock_polsess_data, variant, expected_fields):
        """Test SB task with outdoor (no reverb) variants loads correct fields."""
        dataset = PolSESSDataset(
            data_root=mock_polsess_data,
            subset="train",
            task="SB",
            allowed_variants=[variant],
        )

        # Get sample without reverb
        sample = dataset[0]

        # Check internal _lazy_load output
        row = dataset.metadata.iloc[0]
        has_reverb = pd.notna(row["reverbForSpeaker1"])
        paths = dataset._build_paths(row, has_reverb)
        audio = dataset._lazy_load(paths, variant, has_reverb)

        assert set(audio.keys()) == expected_fields, \
            f"Variant {variant} with SB task should have fields {expected_fields}, got {set(audio.keys())}"

    def test_sb_outdoor_variant_c(self, mock_polsess_data):
        """Test SB task with C variant (outdoor sample) loads correct fields."""
        dataset = PolSESSDataset(
            data_root=mock_polsess_data,
            subset="train",
            task="SB",
            allowed_variants=["C"],
        )

        # For C variant, both indoor and outdoor samples are included
        # Find the outdoor sample
        for idx in range(len(dataset.metadata)):
            row = dataset.metadata.iloc[idx]
            has_reverb = pd.notna(row["reverbForSpeaker1"])
            if not has_reverb:
                paths = dataset._build_paths(row, has_reverb)
                audio = dataset._lazy_load(paths, "C", has_reverb)
                expected_fields = {"mix", "speaker1", "speaker2", "scene", "event"}
                assert set(audio.keys()) == expected_fields, \
                    f"Variant C with SB task (outdoor) should have fields {expected_fields}, got {set(audio.keys())}"
                break
        else:
            pytest.fail("No outdoor sample found for C variant test")


class TestApplyMMIPC:
    """Test _apply_mmipc correctly removes components from mix."""

    def test_apply_mmipc_es_ser(self, mock_polsess_data):
        """Test ES+SER: should remove speaker2 + sp2_reverb only."""
        dataset = PolSESSDataset(
            data_root=mock_polsess_data,
            subset="train",
            task="ES",
            allowed_variants=["SER"],
        )

        row = dataset.metadata.iloc[0]
        has_reverb = True
        paths = dataset._build_paths(row, has_reverb)
        audio = dataset._lazy_load(paths, "SER", has_reverb)
        mix = dataset._apply_mmipc(audio, has_reverb)

        # Expected: mix - speaker2 - sp2_reverb
        # With mock values: 1.0 - 0.2 - 0.02 = 0.78
        expected = torch.ones(int(8000 * 4.0)) * 0.78
        assert torch.allclose(mix, expected, atol=1e-5), \
            f"ES+SER should subtract speaker2 and sp2_reverb, expected {expected[0]}, got {mix[0]}"

    def test_apply_mmipc_es_c_indoor(self, mock_polsess_data):
        """Test ES+C (indoor): should remove everything."""
        dataset = PolSESSDataset(
            data_root=mock_polsess_data,
            subset="train",
            task="ES",
            allowed_variants=["C"],
        )

        row = dataset.metadata.iloc[0]
        has_reverb = True
        paths = dataset._build_paths(row, has_reverb)
        audio = dataset._lazy_load(paths, "C", has_reverb)
        mix = dataset._apply_mmipc(audio, has_reverb)

        # Expected: mix - sp1_reverb - sp2_reverb - speaker2 - scene - event - ev_reverb
        # 1.0 - 0.01 - 0.02 - 0.2 - 0.3 - 0.4 - 0.04 = 0.03
        expected = torch.ones(int(8000 * 4.0)) * 0.03
        assert torch.allclose(mix, expected, atol=1e-5), \
            f"ES+C (indoor) should remove all components, expected {expected[0]}, got {mix[0]}"

    def test_apply_mmipc_eb_c_indoor(self, mock_polsess_data):
        """Test EB+C (indoor): should remove everything except speakers."""
        dataset = PolSESSDataset(
            data_root=mock_polsess_data,
            subset="train",
            task="EB",
            allowed_variants=["C"],
        )

        row = dataset.metadata.iloc[0]
        has_reverb = True
        paths = dataset._build_paths(row, has_reverb)
        audio = dataset._lazy_load(paths, "C", has_reverb)
        mix = dataset._apply_mmipc(audio, has_reverb)

        # Expected: mix - sp1_reverb - sp2_reverb - scene - event - ev_reverb
        # 1.0 - 0.01 - 0.02 - 0.3 - 0.4 - 0.04 = 0.23
        expected = torch.ones(int(8000 * 4.0)) * 0.23
        assert torch.allclose(mix, expected, atol=1e-5), \
            f"EB+C (indoor) should remove background only, expected {expected[0]}, got {mix[0]}"

    def test_apply_mmipc_es_se_outdoor(self, mock_polsess_data):
        """Test ES+SE (outdoor): should remove speaker2 only."""
        dataset = PolSESSDataset(
            data_root=mock_polsess_data,
            subset="train",
            task="ES",
            allowed_variants=["SE"],
        )

        row = dataset.metadata.iloc[0]  # outdoor sample (filtered dataset)
        has_reverb = False
        paths = dataset._build_paths(row, has_reverb)
        audio = dataset._lazy_load(paths, "SE", has_reverb)
        mix = dataset._apply_mmipc(audio, has_reverb)

        # Expected: mix - speaker2
        # 2.0 - 0.6 = 1.4
        expected = torch.ones(int(8000 * 4.0)) * 1.4
        assert torch.allclose(mix, expected, atol=1e-5), \
            f"ES+SE (outdoor) should remove speaker2 only, expected {expected[0]}, got {mix[0]}"


class TestComputeClean:
    """Test _compute_clean returns correct target based on task."""

    def test_compute_clean_es(self, mock_polsess_data):
        """Test ES task returns speaker1 only."""
        dataset = PolSESSDataset(
            data_root=mock_polsess_data,
            subset="train",
            task="ES",
            allowed_variants=["SER"],
        )

        row = dataset.metadata.iloc[0]
        has_reverb = True
        paths = dataset._build_paths(row, has_reverb)
        audio = dataset._lazy_load(paths, "SER", has_reverb)
        clean = dataset._compute_clean(audio)

        expected = torch.ones(int(8000 * 4.0)) * 0.1
        assert torch.allclose(clean, expected, atol=1e-5), \
            "ES task should return speaker1 only"

    def test_compute_clean_eb(self, mock_polsess_data):
        """Test EB task returns sum of both speakers."""
        dataset = PolSESSDataset(
            data_root=mock_polsess_data,
            subset="train",
            task="EB",
            allowed_variants=["SER"],
        )

        row = dataset.metadata.iloc[0]
        has_reverb = True
        paths = dataset._build_paths(row, has_reverb)
        audio = dataset._lazy_load(paths, "SER", has_reverb)
        clean = dataset._compute_clean(audio)

        # speaker1 + speaker2 = 0.1 + 0.2 = 0.3
        expected = torch.ones(int(8000 * 4.0)) * 0.3
        assert torch.allclose(clean, expected, atol=1e-5), \
            "EB task should return sum of speaker1 and speaker2"

    def test_compute_clean_sb(self, mock_polsess_data):
        """Test SB task returns stacked speakers [2, T]."""
        dataset = PolSESSDataset(
            data_root=mock_polsess_data,
            subset="train",
            task="SB",
            allowed_variants=["SER"],
        )

        row = dataset.metadata.iloc[0]
        has_reverb = True
        paths = dataset._build_paths(row, has_reverb)
        audio = dataset._lazy_load(paths, "SER", has_reverb)
        clean = dataset._compute_clean(audio)

        assert clean.shape[0] == 2, "SB task should return 2 channels"
        assert torch.allclose(clean[0], torch.ones(int(8000 * 4.0)) * 0.1, atol=1e-5), \
            "SB task channel 0 should be speaker1"
        assert torch.allclose(clean[1], torch.ones(int(8000 * 4.0)) * 0.2, atol=1e-5), \
            "SB task channel 1 should be speaker2"


class TestVariantSelection:
    """Test variant selection logic."""

    def test_variant_selection_respects_reverb(self, mock_polsess_data):
        """Test that indoor variants are only selected for reverb samples."""
        dataset = PolSESSDataset(
            data_root=mock_polsess_data,
            subset="train",
            task="ES",
            allowed_variants=["SER"],  # Indoor variant
        )

        # Should only have reverb samples
        assert len(dataset) == 1, "Dataset with indoor variant should filter to reverb samples only"

    def test_variant_selection_outdoor_only(self, mock_polsess_data):
        """Test that outdoor variants are only selected for non-reverb samples."""
        dataset = PolSESSDataset(
            data_root=mock_polsess_data,
            subset="train",
            task="ES",
            allowed_variants=["SE"],  # Outdoor variant
        )

        # Should only have non-reverb samples
        assert len(dataset) == 1, "Dataset with outdoor variant should filter to non-reverb samples only"

    def test_variant_selection_mixed(self, mock_polsess_data):
        """Test that mixed variants include both sample types."""
        dataset = PolSESSDataset(
            data_root=mock_polsess_data,
            subset="train",
            task="ES",
            allowed_variants=["C"],  # Both indoor and outdoor
        )

        # Should have both samples
        assert len(dataset) == 2, "Dataset with C variant should include both reverb and non-reverb samples"


class TestEndToEnd:
    """Test complete dataset __getitem__ flow."""

    @pytest.mark.parametrize("task,variant", [
        ("ES", "SER"), ("ES", "SR"), ("ES", "ER"), ("ES", "R"), ("ES", "C"),
        ("ES", "SE"), ("ES", "S"), ("ES", "E"),
        ("EB", "SER"), ("EB", "SR"), ("EB", "ER"), ("EB", "R"), ("EB", "C"),
        ("EB", "SE"), ("EB", "S"), ("EB", "E"),
        ("SB", "SER"), ("SB", "SR"), ("SB", "ER"), ("SB", "R"), ("SB", "C"),
        ("SB", "SE"), ("SB", "S"), ("SB", "E"),
    ])
    def test_all_task_variant_combinations(self, mock_polsess_data, task, variant):
        """Test all valid task-variant combinations work end-to-end."""
        dataset = PolSESSDataset(
            data_root=mock_polsess_data,
            subset="train",
            task=task,
            allowed_variants=[variant],
        )

        if len(dataset) == 0:
            pytest.skip(f"No samples for task={task}, variant={variant}")

        # Should be able to get a sample without errors
        sample = dataset[0]

        assert "mix" in sample, "Sample should contain 'mix'"
        assert "clean" in sample, "Sample should contain 'clean'"
        assert "background_complexity" in sample, "Sample should contain 'background_complexity'"
        assert sample["background_complexity"] == variant, f"Should return correct variant {variant}"

        # Check output shapes
        assert sample["mix"].dim() == 1, "Mix should be 1D tensor"

        if task == "SB":
            assert sample["clean"].dim() == 2, "SB clean should be 2D tensor [2, T]"
            assert sample["clean"].shape[0] == 2, "SB clean should have 2 channels"
        else:
            assert sample["clean"].dim() == 1, f"{task} clean should be 1D tensor"
