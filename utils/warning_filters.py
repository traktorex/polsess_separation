"""Canonical warning filters, applied on import.

Some of these warnings fire at *import time* inside third-party libraries
(notably SpeechBrain's ``check_torchaudio_backend()``, which runs at module load
in ``stoi_loss.py`` and ``dataio.py`` and calls the deprecated
``torchaudio.list_audio_backends()``). Registering the filters from inside a
function that runs in ``main()`` is too late — speechbrain is already imported by
then. So entry-point scripts import this module *first*, before any import that
pulls in speechbrain, to register the filters early:

    from utils import warning_filters  # noqa: F401  (must precede speechbrain imports)

``utils/__init__`` applies these filters as its first statement, so importing the
``utils`` package in any way registers them before the package goes on to import
speechbrain-backed submodules. This module deliberately has no heavy imports so
it can be loaded before everything else; ``filterwarnings`` is idempotent, so the
re-application by ``utils.common.setup_warnings()`` is harmless.
"""

import warnings


def apply() -> None:
    """Register the project's warning filters (idempotent)."""
    # TorchAudio/torio deprecations from the TorchCodec migration. These all
    # share the "transition TorchAudio into a maintenance phase" boilerplate;
    # matching it suppresses the whole family at once: list_audio_backends (fires
    # at import via speechbrain's check_torchaudio_backend), StreamingMediaDecoder
    # / StreamReader (fire at audio-decode time inside dataloader workers), etc.
    warnings.filterwarnings(
        "ignore",
        message=".*transition TorchAudio into a maintenance phase.*",
        category=UserWarning,
    )

    # TorchAudio load deprecation (transition to TorchCodec) — different message,
    # no "maintenance phase" boilerplate, so it needs its own filter.
    warnings.filterwarnings(
        "ignore",
        message=".*this function's implementation will be changed.*torchcodec.*",
        category=UserWarning,
    )

    # TorchMetrics pkg_resources deprecation (setuptools migration)
    warnings.filterwarnings(
        "ignore",
        message=".*pkg_resources is deprecated.*",
        category=UserWarning,
    )

    # Torch dynamo warnings for pybind functions (SPMamba selective scan)
    warnings.filterwarnings(
        "ignore",
        message=".*Dynamo does not know how to trace.*selective_scan_cuda.*",
        category=UserWarning,
    )

    # ComplexHalf experimental support (used in SPMamba STFT)
    warnings.filterwarnings(
        "ignore",
        message=".*ComplexHalf support is experimental.*",
        category=UserWarning,
    )

    # Suppress pybind deprecation warnings from frozen importlib
    warnings.filterwarnings(
        "ignore",
        message=".*SwigPy.*has no __module__ attribute.*",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*swigvarlink.*has no __module__ attribute.*",
        category=DeprecationWarning,
    )
    warnings.filterwarnings("ignore", category=UserWarning, module="inspect")
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
    warnings.filterwarnings("ignore", message=".*speechbrain.pretrained.*")


apply()
