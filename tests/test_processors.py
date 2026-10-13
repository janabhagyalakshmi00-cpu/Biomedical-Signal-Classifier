"""Tests for processor configurations."""

import pytest
from src.processors import load_processor, validate_processor, get_coupling_map


def test_load_processor_a():
    cfg = load_processor("A")
    assert cfg["physical_qubit_budget"] == 5
    assert len(cfg["coupling_map"]) == 4


def test_load_processor_b():
    cfg = load_processor("B")
    assert cfg["physical_qubit_budget"] == 9
    assert len(cfg["coupling_map"]) == 12


def test_coupling_map_object():
    cfg = load_processor("A")
    cmap = get_coupling_map(cfg)
    assert cmap.size() == 5
