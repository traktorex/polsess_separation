"""Config utility functions for serialization and conversion."""

from dataclasses import is_dataclass, asdict
from typing import Any


def dataclass_to_dict(obj: Any) -> dict:
    """Convert dataclass or SimpleNamespace to dict recursively."""
    if is_dataclass(obj):
        return asdict(obj)
    elif hasattr(obj, "__dict__"):
        return vars(obj)
    else:
        return obj
