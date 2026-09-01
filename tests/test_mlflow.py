"""
Unit and Integration Tests for MLflow Experiment Tracking & Model Registry (Milestone 3).
"""

import os
import sys
import tempfile
import pytest

# Ensure parent directory is on sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import mlflow
from ml.mlflow_integration import (
    setup_mlflow,
    log_model_run_to_mlflow,
    track_and_register_all_models,
    EXPERIMENT_NAME,
    REGISTERED_MODEL_NAME
)
from ml.model_comparison import get_best_model_artifacts


def test_setup_mlflow_local_tracking():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        db_path = os.path.join(temp_dir, "test_mlflow.db").replace(os.sep, "/")
        ml = setup_mlflow(tracking_uri=f"sqlite:///{db_path}")
        assert ml is not None
        exp = ml.get_experiment_by_name(EXPERIMENT_NAME)
        assert exp is not None
        assert exp.name == EXPERIMENT_NAME


def test_log_model_run_to_mlflow():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        db_path = os.path.join(temp_dir, "test_mlflow.db").replace(os.sep, "/")
        setup_mlflow(tracking_uri=f"sqlite:///{db_path}")
        model, vectorizer, label_encoder, model_name = get_best_model_artifacts()

        params = {"n_estimators": 50, "max_depth": 10, "random_state": 42}
        metrics = {"accuracy": 0.88, "precision": 0.87, "recall": 0.88, "f1_score": 0.875}

        run_result = log_model_run_to_mlflow(
            model_name="Test_Model_Run",
            model_key="random_forest",
            params=params,
            metrics=metrics,
            model_obj=model,
            vectorizer_obj=vectorizer,
            label_encoder_obj=label_encoder,
            register_as_best=False
        )

        assert "run_id" in run_result
        assert run_result["model_name"] == "Test_Model_Run"
        assert run_result["metrics"]["accuracy"] == 0.88


def test_track_and_register_pipeline():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        db_path = os.path.join(temp_dir, "test_mlflow.db").replace(os.sep, "/")
        setup_mlflow(tracking_uri=f"sqlite:///{db_path}")
        summary = track_and_register_all_models()

        assert summary["status"] == "success"
        assert summary["experiment_name"] == EXPERIMENT_NAME
        assert summary["registered_model_name"] == REGISTERED_MODEL_NAME
        assert summary["total_runs_tracked"] >= 1
