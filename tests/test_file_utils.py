"""Tests for file utility functions."""

import pytest
from pathlib import Path
from utils.file_utils import ensure_dir


def test_ensure_dir_creates_directory(tmp_path):
    """Test ensure_dir creates directory."""
    new_dir = tmp_path / "test" / "nested" / "path"
    
    assert not new_dir.exists()
    
    result = ensure_dir(new_dir)
    
    assert new_dir.exists()
    assert new_dir.is_dir()
    assert result == new_dir


def test_ensure_dir_creates_parent_directories(tmp_path):
    """Test ensure_dir creates parent directories."""
    nested_dir = tmp_path / "level1" / "level2" / "level3"
    
    result = ensure_dir(nested_dir)
    
    assert (tmp_path / "level1").exists()
    assert (tmp_path / "level1" / "level2").exists()
    assert nested_dir.exists()
    assert result == nested_dir


def test_ensure_dir_existing_directory_no_error(tmp_path):
    """Test ensure_dir handles existing directory without error."""
    existing_dir = tmp_path / "existing"
    existing_dir.mkdir()
    
    # Should not raise error
    result = ensure_dir(existing_dir)
    
    assert result == existing_dir
    assert existing_dir.exists()


def test_ensure_dir_returns_path_object(tmp_path):
    """Test ensure_dir returns Path object."""
    new_dir = tmp_path / "test"
    
    result = ensure_dir(new_dir)
    
    assert isinstance(result, Path)


def test_ensure_dir_accepts_string_path(tmp_path):
    """Test ensure_dir accepts string paths."""
    new_dir = tmp_path / "string_test"
    new_dir_str = str(new_dir)
    
    result = ensure_dir(new_dir_str)
    
    assert new_dir.exists()
    assert isinstance(result, Path)
    assert result == new_dir


def test_ensure_dir_idempotent(tmp_path):
    """Test ensure_dir can be called multiple times safely."""
    new_dir = tmp_path / "idempotent"
    
    result1 = ensure_dir(new_dir)
    result2 = ensure_dir(new_dir)
    result3 = ensure_dir(new_dir)
    
    assert result1 == result2 == result3
    assert new_dir.exists()
