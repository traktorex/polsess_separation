"""The polsess-models distribution: packaging rules and the inference bundle.

models/ is published as `polsess_models` (root pyproject.toml). That only works
while every packaged file uses relative imports, nothing from the rest of the
repository and nothing from the modules the wheel leaves out. TestPackagingRules
checks that statically (no build, no network); TestBuiltWheel builds the wheel
and imports it from a clean directory, and needs `hatchling`. TestBundleRoundTrip
covers scripts/export_separator.py -> models.inference.load_separator.
"""

import ast
import json
import re
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"

sys.path.insert(0, str(REPO_ROOT))

from models import MAMBA_MODELS, ConvTasNet, __version__  # noqa: E402
from models.inference import (  # noqa: E402
    BUNDLE_META,
    BUNDLE_WEIGHTS,
    load_separator,
)

# Everything importable from the repository root: importing one of these from a
# packaged file would work here and fail in the installed wheel.
REPO_TOP_LEVEL = (
    {p.name for p in REPO_ROOT.iterdir() if p.is_dir() and any(p.glob("*.py"))}
    | {p.stem for p in REPO_ROOT.glob("*.py")}
)

# Libraries only the excluded Mamba family may depend on.
MAMBA_ONLY_LIBRARIES = {"mamba_ssm", "causal_conv1d"}

# The one place a shipped file may name excluded modules: models/__init__.py,
# behind its ModuleNotFoundError / MAMBA_AVAILABLE guards.
GUARDED_IMPORTS = {
    ("models/__init__.py", "models/mamba"),
    ("models/__init__.py", "models/spmamba.py"),
    ("models/__init__.py", "models/mamba_tasnet.py"),
    ("models/__init__.py", "models/dpmamba.py"),
}


def _excludes():
    with open(REPO_ROOT / "pyproject.toml", "rb") as fh:
        return tomllib.load(fh)["tool"]["hatch"]["build"]["exclude"]


def _is_excluded(rel: str) -> bool:
    return any(rel == ex or rel.startswith(ex + "/") for ex in _excludes())


def _packaged_files():
    """models/**/*.py minus the pyproject.toml excludes."""
    return [
        path for path in sorted(MODELS_DIR.rglob("*.py"))
        if not _is_excluded(path.relative_to(REPO_ROOT).as_posix())
    ]


def _imports(path):
    """(node, level, dotted module name) for every import statement in a file."""
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node, 0, alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                yield node, node.level, node.module
            else:                                   # `from . import a, b`
                for alias in node.names:
                    yield node, node.level, alias.name


class TestPackagingRules:
    def test_excludes_point_at_real_paths(self):
        """A renamed file would silently slip back into the wheel."""
        for ex in _excludes():
            assert (REPO_ROOT / ex).exists(), f"pyproject.toml excludes missing path {ex}"

    def test_repo_top_level_is_detected(self):
        assert {"models", "config", "utils", "training", "datasets", "scripts"} <= REPO_TOP_LEVEL

    def test_packaged_files_import_nothing_from_the_repository(self):
        offenders = [
            f"{path.relative_to(REPO_ROOT)}:{node.lineno} imports {name}"
            for path in _packaged_files()
            for node, level, name in _imports(path)
            if level == 0 and name.split(".")[0] in REPO_TOP_LEVEL
        ]
        assert not offenders, "absolute repository imports in packaged files:\n" + "\n".join(offenders)

    def test_packaged_files_do_not_import_excluded_modules(self):
        """`from .factory import x` in a shipped file: fine here, ImportError in the wheel."""
        offenders = []
        for path in _packaged_files():
            rel = path.relative_to(REPO_ROOT).as_posix()
            for node, level, name in _imports(path):
                if level == 0:
                    continue
                base = path.parent
                for _ in range(level - 1):
                    base = base.parent
                target = base.joinpath(*name.split("."))
                for candidate in (target, target.with_suffix(".py")):
                    target_rel = candidate.relative_to(REPO_ROOT).as_posix()
                    if candidate.exists() and _is_excluded(target_rel) \
                            and (rel, target_rel) not in GUARDED_IMPORTS:
                        offenders.append(f"{rel}:{node.lineno} imports excluded {target_rel}")
        assert not offenders, "\n".join(offenders)

    def test_mamba_dependent_files_are_excluded(self):
        """The other direction: a new file that needs mamba-ssm must join the excludes."""
        offenders = [
            f"{path.relative_to(REPO_ROOT)}:{node.lineno} imports {name}"
            for path in _packaged_files()
            for node, level, name in _imports(path)
            if level == 0 and name.split(".")[0] in MAMBA_ONLY_LIBRARIES
        ]
        assert not offenders, "packaged files that need the Mamba stack:\n" + "\n".join(offenders)

    def test_version_is_a_plain_string_literal(self):
        """hatch reads __version__ from the source text of models/__init__.py, with
        a regex; a computed value would import fine here and break the build."""
        source = (MODELS_DIR / "__init__.py").read_text(encoding="utf-8")
        match = re.search(r'^__version__ = "(\d+\.\d+\.\d+)"$', source, flags=re.MULTILINE)
        assert match, "__version__ must be a literal 'X.Y.Z' assignment on its own line"
        assert match.group(1) == __version__


class TestBuiltWheel:
    """The real thing: build the wheel, import it with the repository off sys.path."""

    def test_wheel_imports_from_a_clean_directory(self, tmp_path):
        pytest.importorskip("hatchling")
        subprocess.run(
            [sys.executable, "-m", "hatchling", "build", "-t", "wheel", "-d", str(tmp_path / "dist")],
            cwd=REPO_ROOT, check=True, capture_output=True,
        )
        (wheel,) = (tmp_path / "dist").glob("polsess_models-*.whl")
        site = tmp_path / "site"
        with zipfile.ZipFile(wheel) as zf:
            names = zf.namelist()
            zf.extractall(site)

        assert f"polsess_models-{__version__}" in wheel.name
        shipped = [n for n in names if n.startswith("polsess_models/")]
        assert shipped and not [n for n in names if n.startswith("models/")]
        for ex in _excludes():
            gone = ex.replace("models/", "polsess_models/", 1)
            assert not [n for n in shipped if n == gone or n.startswith(gone + "/")], ex

        code = (
            "import polsess_models as m, sys; "
            "assert not m.MAMBA_AVAILABLE; "
            "assert sorted(m.MODELS) == ['convtasnet', 'dprnn', 'mossformer2', 'sepformer', 'tf_mossformer']; "
            "assert 'models' not in sys.modules; "
            "m.load_separator"
        )
        clean = tmp_path / "clean"
        clean.mkdir()
        subprocess.run(
            [sys.executable, "-c", code], cwd=clean, check=True,
            env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(site)},
        )


safetensors = pytest.importorskip("safetensors")
from scripts.export_separator import export  # noqa: E402


class TestBundleRoundTrip:
    PARAMS = {"N": 64, "B": 64, "H": 128, "P": 3, "X": 4, "R": 2, "C": 2}

    def _make_checkpoint(self, tmp_path, **extra):
        torch.manual_seed(0)
        model = ConvTasNet(**self.PARAMS)
        checkpoint_path = tmp_path / "run_name" / "convtasnet_SB_best.pt"
        checkpoint_path.parent.mkdir()
        config = {
            "model": {"model_type": "convtasnet", "convtasnet": dict(self.PARAMS)},
            "data": {"task": "SB", "dataset_type": "polsess", **extra},
        }
        torch.save(
            {"model_state_dict": model.state_dict(), "config": config,
             "epoch": 3, "val_sisdr": 1.5},
            checkpoint_path,
        )
        return checkpoint_path, model.eval()

    def test_bundle_reproduces_the_checkpoint(self, tmp_path):
        checkpoint_path, model = self._make_checkpoint(tmp_path, sample_rate=16000)
        bundle = tmp_path / "bundle"

        meta = export(checkpoint_path, bundle, device="cpu")

        assert sorted(p.name for p in bundle.iterdir()) == [BUNDLE_META, BUNDLE_WEIGHTS]
        assert meta["sample_rate"] == 16000 and meta["n_src"] == 2
        assert meta["model_params"] == self.PARAMS
        assert meta["source_checkpoint"]["name"] == "run_name/convtasnet_SB_best.pt"
        assert meta["val_metric"] == "si_sdr" and meta["exported_with"]["torch"] == torch.__version__
        assert json.loads((bundle / BUNDLE_META).read_text()) == meta

        loaded, loaded_meta = load_separator(str(bundle), device="cpu")
        assert loaded_meta == meta and not loaded.training
        x = torch.randn(1, 4000)
        with torch.no_grad():
            assert torch.equal(loaded(x), model(x))

    def test_checkpoint_without_sample_rate_is_8k(self, tmp_path):
        checkpoint_path, _ = self._make_checkpoint(tmp_path)
        assert export(checkpoint_path, tmp_path / "bundle", device="cpu")["sample_rate"] == 8000

    def test_directory_that_is_not_a_bundle(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Not a separator bundle"):
            load_separator(str(tmp_path), device="cpu")

    def test_unknown_format_version(self, tmp_path):
        checkpoint_path, _ = self._make_checkpoint(tmp_path)
        bundle = tmp_path / "bundle"
        meta = export(checkpoint_path, bundle, device="cpu")
        (bundle / BUNDLE_META).write_text(json.dumps({**meta, "format_version": 99}))
        with pytest.raises(ValueError, match="bundle format 99"):
            load_separator(str(bundle), device="cpu")

    def test_unknown_architecture_names_both_versions(self, tmp_path):
        checkpoint_path, _ = self._make_checkpoint(tmp_path)
        bundle = tmp_path / "bundle"
        meta = export(checkpoint_path, bundle, device="cpu")
        (bundle / BUNDLE_META).write_text(
            json.dumps({**meta, "model_type": "from_the_future", "polsess_models_version": "9.0.0"})
        )
        with pytest.raises(ValueError, match=r"Unknown model type.*9\.0\.0.*installed"):
            load_separator(str(bundle), device="cpu")

    def test_weights_that_do_not_fit_the_architecture(self, tmp_path):
        checkpoint_path, _ = self._make_checkpoint(tmp_path)
        bundle = tmp_path / "bundle"
        meta = export(checkpoint_path, bundle, device="cpu")
        (bundle / BUNDLE_META).write_text(
            json.dumps({**meta, "model_params": {**self.PARAMS, "N": 32}})
        )
        with pytest.raises(RuntimeError, match="size mismatch"):
            load_separator(str(bundle), device="cpu")

    def test_kwargs_the_installed_version_does_not_know(self, tmp_path):
        checkpoint_path, _ = self._make_checkpoint(tmp_path)
        bundle = tmp_path / "bundle"
        meta = export(checkpoint_path, bundle, device="cpu")
        (bundle / BUNDLE_META).write_text(json.dumps({
            **meta, "polsess_models_version": "9.0.0",
            "model_params": {**self.PARAMS, "knob_from_the_future": 1},
        }))
        with pytest.raises(TypeError, match=r"knob_from_the_future.*9\.0\.0.*installed"):
            load_separator(str(bundle), device="cpu")

    def test_metadata_without_required_keys(self, tmp_path):
        checkpoint_path, _ = self._make_checkpoint(tmp_path)
        bundle = tmp_path / "bundle"
        meta = export(checkpoint_path, bundle, device="cpu")
        del meta["model_type"]
        (bundle / BUNDLE_META).write_text(json.dumps(meta))
        with pytest.raises(ValueError, match=r"missing required keys \['model_type'\]"):
            load_separator(str(bundle), device="cpu")

    def test_per_variant_runs_record_their_metric(self, tmp_path):
        checkpoint_path, _ = self._make_checkpoint(tmp_path)
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        checkpoint["config"]["training"] = {"per_variant_validation": True}
        torch.save(checkpoint, checkpoint_path)
        meta = export(checkpoint_path, tmp_path / "bundle", device="cpu")
        assert meta["val_metric"] == "si_sdri_mean_over_variants"

    def test_failed_export_leaves_no_files(self, tmp_path, monkeypatch):
        """Any failure after the first write, not only a round-trip mismatch."""
        checkpoint_path, _ = self._make_checkpoint(tmp_path)
        bundle = tmp_path / "bundle"

        def boom(*args, **kwargs):
            raise RuntimeError("disk full")
        monkeypatch.setattr(json, "dump", boom)
        with pytest.raises(RuntimeError, match="disk full"):
            export(checkpoint_path, bundle, device="cpu")
        assert not list(bundle.iterdir())

    @pytest.mark.parametrize("model_type", MAMBA_MODELS)
    def test_mamba_checkpoints_are_refused(self, tmp_path, monkeypatch, model_type):
        """No polsess-models wheel ships the family, so the bundle would load nowhere."""
        import scripts.export_separator as exporter

        config = {"model": {"model_type": model_type, model_type: {}}, "data": {}}
        monkeypatch.setattr(
            exporter, "load_model_for_inference",
            lambda *a, **k: (torch.nn.Identity(), {"config": config}),
        )
        with pytest.raises(SystemExit, match="Mamba-family"):
            exporter.export(tmp_path / "x.pt", tmp_path / "bundle", device="cpu")
        assert not (tmp_path / "bundle").exists()

    def test_corrupt_metadata_names_the_file(self, tmp_path):
        checkpoint_path, _ = self._make_checkpoint(tmp_path)
        bundle = tmp_path / "bundle"
        export(checkpoint_path, bundle, device="cpu")
        (bundle / BUNDLE_META).write_text("{not json")
        with pytest.raises(ValueError, match=r"separator\.json: not valid JSON"):
            load_separator(str(bundle), device="cpu")

    def test_truncated_weights_name_the_file(self, tmp_path):
        """A badly copied bundle: safetensors' own exception type, wrapped."""
        checkpoint_path, _ = self._make_checkpoint(tmp_path)
        bundle = tmp_path / "bundle"
        export(checkpoint_path, bundle, device="cpu")
        data = (bundle / BUNDLE_WEIGHTS).read_bytes()
        (bundle / BUNDLE_WEIGHTS).write_bytes(data[: len(data) // 2])
        with pytest.raises(RuntimeError, match=r"weights\.safetensors: cannot be read.*truncated or corrupt"):
            load_separator(str(bundle), device="cpu")
