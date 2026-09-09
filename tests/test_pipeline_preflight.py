"""Tests for the config preflight checks (asr_pipeline/preflight.py, A2).

Preflight runs before any model load (SCOPE §4: fail loud, early). Each check
is exercised by pointing the config at the backend that needs a given env var /
checkpoint and monkeypatching the environment. num2words is warn-only — its
absence must NOT block (the hard-dep ruling is SCOPE §10 q2, the author's).

All CPU-only, no models loaded.
"""

from pathlib import Path

import pytest

from asr_pipeline.config import PipelineConfig
from asr_pipeline.preflight import check_preflight, preflight


# ---------------------------------------------------------------------------
# Helpers — build configs for a backend without tripping __post_init__.
#
# `PipelineConfig()` reads $HF_TOKEN via a default_factory and __post_init__
# rejects a pyannote config with no token. To exercise the *preflight* pyannote
# branch we build with a token present, then clear it — preflight itself never
# calls __post_init__, so this is a legitimate test state.
# ---------------------------------------------------------------------------


def _pyannote_cfg(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "seed-token")   # let construction pass
    cfg = PipelineConfig()                          # diarization.backend defaults to pyannote
    return cfg


def _sortformer_cfg():
    cfg = PipelineConfig()
    cfg.diarization.backend = "sortformer"
    cfg.__post_init__()                             # sortformer needs no hf_token
    return cfg


# ---------------------------------------------------------------------------
# Diarizer — pyannote HF token
# ---------------------------------------------------------------------------


def test_pyannote_missing_hf_token_is_a_failure(monkeypatch):
    cfg = _pyannote_cfg(monkeypatch)
    cfg.diarization.hf_token = None
    monkeypatch.delenv("HF_TOKEN", raising=False)
    problems = preflight(cfg)
    assert any("HF_TOKEN" in p and "pyannote" in p for p in problems)


def test_pyannote_with_hf_token_passes(monkeypatch):
    cfg = _pyannote_cfg(monkeypatch)         # hf_token resolved from env
    # Give the separator + BWE clean so only the diarizer branch is under test.
    cfg.separation.enabled = False
    problems = preflight(cfg)
    assert not any("HF_TOKEN" in p for p in problems)


def test_pyannote_env_token_satisfies_even_if_config_token_unset(monkeypatch):
    cfg = _pyannote_cfg(monkeypatch)
    cfg.diarization.hf_token = None
    monkeypatch.setenv("HF_TOKEN", "from-env")
    cfg.separation.enabled = False
    assert not any("HF_TOKEN" in p for p in preflight(cfg))


# ---------------------------------------------------------------------------
# Diarizer — sortformer venv
# ---------------------------------------------------------------------------


def test_sortformer_missing_venv_is_a_failure(monkeypatch):
    cfg = _sortformer_cfg()
    cfg.separation.enabled = False
    monkeypatch.delenv("SORTFORMER_VENV_PY", raising=False)
    problems = preflight(cfg)
    assert any("SORTFORMER_VENV_PY" in p for p in problems)


def test_sortformer_venv_points_at_missing_file(monkeypatch):
    cfg = _sortformer_cfg()
    cfg.separation.enabled = False
    monkeypatch.setenv("SORTFORMER_VENV_PY", "/no/such/python")
    problems = preflight(cfg)
    assert any("points at a missing file" in p for p in problems)


def test_sortformer_venv_present_passes(monkeypatch, tmp_path):
    fake_py = tmp_path / "python"
    fake_py.write_text("#!/bin/sh\n")
    cfg = _sortformer_cfg()
    cfg.separation.enabled = False
    monkeypatch.setenv("SORTFORMER_VENV_PY", str(fake_py))
    problems = preflight(cfg)
    assert not any("SORTFORMER_VENV_PY" in p for p in problems)


def test_sortformer_hf_token_missing_is_warn_only(monkeypatch, tmp_path, capsys):
    # $HF_TOKEN on the sortformer path is warn-only (needed only for the FIRST
    # public model download) — a warning prints, but it never blocks.
    fake_py = tmp_path / "python"
    fake_py.write_text("#!/bin/sh\n")
    cfg = _sortformer_cfg()
    cfg.separation.enabled = False
    monkeypatch.setenv("SORTFORMER_VENV_PY", str(fake_py))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    problems = preflight(cfg)
    assert not any("HF_TOKEN" in p for p in problems)      # not a blocker
    out = capsys.readouterr().out
    assert "HF_TOKEN" in out and "warn" in out.lower()


# ---------------------------------------------------------------------------
# Post-separation BWE — ap_bwe checkpoint
# ---------------------------------------------------------------------------


def test_ap_bwe_missing_checkpoint_is_a_failure(monkeypatch):
    cfg = _pyannote_cfg(monkeypatch)
    cfg.separation.enabled = False
    cfg.post_separation_processing.backend = "ap_bwe"
    cfg.post_separation_processing.checkpoint_path = "/no/such/g_8kto16k"
    problems = preflight(cfg)
    assert any("ap_bwe" in p and "checkpoint missing" in p for p in problems)


def test_ap_bwe_present_checkpoint_passes(monkeypatch, tmp_path):
    ckpt = tmp_path / "g_8kto16k"
    ckpt.write_text("x")
    cfg = _pyannote_cfg(monkeypatch)
    cfg.separation.enabled = False
    cfg.post_separation_processing.backend = "ap_bwe"
    cfg.post_separation_processing.checkpoint_path = str(ckpt)
    assert not any("ap_bwe" in p for p in preflight(cfg))


def test_naive_bwe_needs_no_checkpoint(monkeypatch):
    cfg = _pyannote_cfg(monkeypatch)
    cfg.separation.enabled = False
    cfg.post_separation_processing.backend = "naive"   # dataclass default
    assert not any("ap_bwe" in p for p in preflight(cfg))


# ---------------------------------------------------------------------------
# Transcription — coherex venv
# ---------------------------------------------------------------------------


def test_coherex_missing_venv_is_a_failure(monkeypatch):
    cfg = _pyannote_cfg(monkeypatch)
    cfg.separation.enabled = False
    cfg.transcription.backend = "coherex"
    monkeypatch.delenv("COHEREX_VENV_PY", raising=False)
    problems = preflight(cfg)
    assert any("COHEREX_VENV_PY" in p for p in problems)


def test_coherex_present_venv_passes(monkeypatch, tmp_path):
    fake_py = tmp_path / "python"
    fake_py.write_text("#!/bin/sh\n")
    cfg = _pyannote_cfg(monkeypatch)
    cfg.separation.enabled = False
    cfg.transcription.backend = "coherex"
    monkeypatch.setenv("COHEREX_VENV_PY", str(fake_py))
    assert not any("COHEREX_VENV_PY" in p for p in preflight(cfg))


def test_whisperx_backend_needs_no_coherex_venv(monkeypatch):
    cfg = _pyannote_cfg(monkeypatch)
    cfg.separation.enabled = False           # default transcription backend = whisperx
    monkeypatch.delenv("COHEREX_VENV_PY", raising=False)
    assert not any("COHEREX_VENV_PY" in p for p in preflight(cfg))


# ---------------------------------------------------------------------------
# Separator checkpoint
# ---------------------------------------------------------------------------


def test_separator_missing_checkpoint_is_a_failure(monkeypatch):
    cfg = _pyannote_cfg(monkeypatch)
    cfg.separation.enabled = True
    cfg.separation.checkpoint_path = "checkpoints/does/not/exist.pt"
    problems = preflight(cfg)
    assert any("separator checkpoint missing" in p for p in problems)


def test_separator_present_checkpoint_passes(monkeypatch, tmp_path):
    ckpt = tmp_path / "sep.pt"
    ckpt.write_text("x")
    cfg = _pyannote_cfg(monkeypatch)
    cfg.separation.enabled = True
    cfg.separation.checkpoint_path = str(ckpt)
    assert not any("separator checkpoint" in p for p in preflight(cfg))


def test_disabled_separator_skips_checkpoint_check(monkeypatch):
    cfg = _pyannote_cfg(monkeypatch)
    cfg.separation.enabled = False
    cfg.separation.checkpoint_path = "checkpoints/does/not/exist.pt"
    assert not any("separator checkpoint" in p for p in preflight(cfg))


# ---------------------------------------------------------------------------
# num2words — warn-only (SCOPE §10 q2 is the author's; preflight never blocks)
# ---------------------------------------------------------------------------


def test_num2words_absent_warns_but_does_not_block(monkeypatch, tmp_path, capsys):
    # Stub the importability probe to False → a warning prints, but num2words
    # never enters the returned failures list.
    monkeypatch.setattr(
        "asr_pipeline.preflight._num2words_importable", lambda: False
    )
    cfg = _pyannote_cfg(monkeypatch)
    cfg.separation.enabled = False
    problems = preflight(cfg)
    assert not any("num2words" in p for p in problems)
    out = capsys.readouterr().out
    assert "num2words" in out and "warn" in out.lower()


def test_num2words_present_no_warning(monkeypatch, capsys):
    monkeypatch.setattr(
        "asr_pipeline.preflight._num2words_importable", lambda: True
    )
    cfg = _pyannote_cfg(monkeypatch)
    cfg.separation.enabled = False
    preflight(cfg)
    assert "num2words" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# check_preflight — the raising wrapper (SCOPE §4)
# ---------------------------------------------------------------------------


def test_check_preflight_raises_on_failure(monkeypatch):
    cfg = _pyannote_cfg(monkeypatch)
    cfg.diarization.hf_token = None
    monkeypatch.delenv("HF_TOKEN", raising=False)
    cfg.separation.enabled = False
    with pytest.raises(RuntimeError, match="Preflight failed"):
        check_preflight(cfg)


def test_check_preflight_passes_clean_config(monkeypatch, tmp_path):
    # A fully-satisfiable config: pyannote token present, naive BWE (no ckpt),
    # whisperx (no coherex venv), separator checkpoint present.
    ckpt = tmp_path / "sep.pt"
    ckpt.write_text("x")
    cfg = _pyannote_cfg(monkeypatch)                    # HF_TOKEN in env
    cfg.post_separation_processing.backend = "naive"
    cfg.separation.enabled = True
    cfg.separation.checkpoint_path = str(ckpt)
    monkeypatch.setattr(
        "asr_pipeline.preflight._num2words_importable", lambda: True
    )
    assert check_preflight(cfg) is None                 # no raise
