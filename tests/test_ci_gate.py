"""
Unit Tests for CI Accuracy Gate Logic (Milestone 3).
"""

import os
import sys
import pytest

# Ensure parent directory is on sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from ml.ci_accuracy_gate import run_accuracy_gate


def test_accuracy_gate_structure_and_metrics():
    # Run gate with a threshold that will pass
    result = run_accuracy_gate(threshold=0.50)

    assert "model_name" in result
    assert "threshold" in result
    assert "actual_accuracy" in result
    assert "actual_precision" in result
    assert "actual_recall" in result
    assert "actual_f1" in result
    assert "passed" in result
    assert "test_samples" in result

    # Accuracy must be a valid probability
    assert 0.0 <= result["actual_accuracy"] <= 1.0
    assert result["test_samples"] > 0
    assert result["passed"] is True


def test_accuracy_gate_failing_condition():
    # If threshold is higher than actual maximum accuracy (e.g. 1.05), passed must be False
    result = run_accuracy_gate(threshold=1.05)
    assert result["passed"] is False
    assert result["actual_accuracy"] < 1.05


def test_accuracy_gate_realistic_threshold():
    # Test standard production threshold of 80%
    result = run_accuracy_gate(threshold=0.80)
    assert result["actual_accuracy"] >= 0.80
    assert result["passed"] is True
