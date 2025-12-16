"""Tests for config utility functions."""

import pytest
from dataclasses import dataclass
from types import SimpleNamespace
from utils.config_utils import dataclass_to_dict


@dataclass
class SampleDataclass:
    """Sample dataclass for testing."""
    name: str
    value: int
    flag: bool = True


@dataclass
class NestedDataclass:
    """Nested dataclass for testing."""
    inner: SampleDataclass
    count: int


def test_dataclass_to_dict_simple():
    """Test converting simple dataclass to dict."""
    obj = SampleDataclass(name="test", value=42, flag=False)
    
    result = dataclass_to_dict(obj)
    
    assert isinstance(result, dict)
    assert result["name"] == "test"
    assert result["value"] == 42
    assert result["flag"] is False


def test_dataclass_to_dict_with_defaults():
    """Test dataclass with default values."""
    obj = SampleDataclass(name="default_test", value=100)
    
    result = dataclass_to_dict(obj)
    
    assert result["name"] == "default_test"
    assert result["value"] == 100
    assert result["flag"] is True  # Default value


def test_dataclass_to_dict_nested():
    """Test converting nested dataclass."""
    inner = SampleDataclass(name="inner", value=10)
    outer = NestedDataclass(inner=inner, count=5)
    
    result = dataclass_to_dict(outer)
    
    assert isinstance(result, dict)
    assert result["count"] == 5
    assert isinstance(result["inner"], dict)
    assert result["inner"]["name"] == "inner"
    assert result["inner"]["value"] == 10


def test_dataclass_to_dict_with_config_params():
    """Test with actual config dataclasses."""
    from config import ConvTasNetParams
    
    params = ConvTasNetParams(N=128, B=256, H=512)
    
    result = dataclass_to_dict(params)
    
    assert result["N"] == 128
    assert result["B"] == 256
    assert result["H"] == 512
    assert result["P"] == 3  # Default value


def test_dataclass_to_dict_simplenamespace():
    """Test converting SimpleNamespace to dict."""
    obj = SimpleNamespace(name="test", value=42, flag=True)
    
    result = dataclass_to_dict(obj)
    
    assert isinstance(result, dict)
    assert result["name"] == "test"
    assert result["value"] == 42
    assert result["flag"] is True


def test_dataclass_to_dict_non_dataclass_non_namespace():
    """Test that non-dataclass/non-namespace objects are returned as-is."""
    obj = "regular_string"
    
    result = dataclass_to_dict(obj)
    
    assert result == "regular_string"


def test_dataclass_to_dict_dict_passthrough():
    """Test that dicts are returned as-is."""
    obj = {"key": "value", "number": 123}
    
    result = dataclass_to_dict(obj)
    
    assert result == obj


def test_dataclass_to_dict_preserves_types():
    """Test that value types are preserved."""
    obj = SampleDataclass(name="type_test", value=999, flag=False)
    
    result = dataclass_to_dict(obj)
    
    assert isinstance(result["name"], str)
    assert isinstance(result["value"], int)
    assert isinstance(result["flag"], bool)
