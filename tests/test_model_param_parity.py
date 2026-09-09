"""Dataclass <-> constructor parity guard (Work Package C, item C1 / gap 10).

`models/factory.py:create_model_from_config` calls `model_class(**vars(params))`
— it unpacks a config dataclass (e.g. `ConvTasNetParams`) directly into the
model constructor. If a dataclass field doesn't exist as a constructor
parameter, that only explodes at run start, and for the Mamba-family models
that only happens on the GPU training box.

This test catches that drift on CPU in well under a second: for every model
registered in `models.MODELS`, assert the corresponding config dataclass's
field names are a subset of `inspect.signature(model_class.__init__)`'s
parameters. Pure signature inspection — no model is ever instantiated, so
this is safe to run with no GPU and no mamba-ssm CUDA kernels touched.
"""

import dataclasses
import inspect

import pytest

try:
    import models
except Exception as exc:  # pragma: no cover - environment-dependent
    pytest.skip(f"models package failed to import: {exc}", allow_module_level=True)

import config as config_module

# model_type (models.MODELS key) -> name of the matching Params dataclass in
# config.py. Mirrors the same mapping config.py itself hardcodes in
# ModelConfig.__post_init__ / load_config_from_dict / Config.summary.
MODEL_TYPE_TO_PARAMS_CLASS_NAME = {
    "convtasnet": "ConvTasNetParams",
    "sepformer": "SepFormerParams",
    "mossformer2": "MossFormer2Params",
    "tf_mossformer": "TFMossFormerParams",
    "dprnn": "DPRNNParams",
    "spmamba": "SPMambaParams",
    "mamba_tasnet": "MambaTasNetParams",
    "dpmamba": "DPMambaParams",
}


@pytest.mark.parametrize("model_type", sorted(models.MODELS.keys()))
def test_params_dataclass_fields_subset_of_constructor_signature(model_type):
    """Every {Params dataclass} field must be an accepted constructor kwarg."""
    params_class_name = MODEL_TYPE_TO_PARAMS_CLASS_NAME.get(model_type)
    if params_class_name is None:
        pytest.skip(
            f"No known config dataclass mapping for model_type={model_type!r}; "
            "add one to MODEL_TYPE_TO_PARAMS_CLASS_NAME in this test if a new "
            "model was registered."
        )

    params_class = getattr(config_module, params_class_name, None)
    if params_class is None:
        pytest.skip(f"config.py has no {params_class_name} (moved/renamed?)")

    model_class = models.MODELS[model_type]

    try:
        signature = inspect.signature(model_class.__init__)
    except (TypeError, ValueError) as exc:
        pytest.skip(f"Could not inspect {model_class.__name__}.__init__: {exc}")

    ctor_params = set(signature.parameters) - {"self"}
    dataclass_fields = {f.name for f in dataclasses.fields(params_class)}

    missing = dataclass_fields - ctor_params
    assert not missing, (
        f"{params_class_name} field(s) {sorted(missing)} are not accepted by "
        f"{model_class.__name__}.__init__ — models/factory.py's "
        f"model_class(**vars(params)) would raise TypeError at model creation."
    )


def test_registry_covers_all_mapped_model_types():
    """Sanity check: the mapping above shouldn't silently go stale.

    If a model is removed from models.MODELS (or the whole Mamba family is
    unavailable on this machine), the parametrized test above simply has
    fewer cases — this test instead checks the mapping doesn't reference a
    model_type that no longer exists in the registry, which would mean this
    file itself has drifted from models/__init__.py.
    """
    stale = set(MODEL_TYPE_TO_PARAMS_CLASS_NAME) - set(models.MODELS)
    # Mamba-family entries are expected to be "missing" from the registry on
    # machines without mamba-ssm — that's not staleness, just unavailability.
    stale -= set(getattr(models, "MAMBA_MODELS", ()))
    assert not stale, f"Mapping references unknown model_type(s): {sorted(stale)}"
