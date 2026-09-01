"""
MLflow Experiment Tracking and Model Registry Integration Module.
Logs hyperparameters, evaluation metrics, artifacts, and registers the winning model
as 'CareerPredictionModel' in the local MLflow Model Registry.
Milestone 3: Skill Gap Analysis, API & CI Integration.
"""

import os
import sys
import json
import joblib
from typing import Dict, Any, Optional

# Set path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from ml.config import MODELS_DIR, RF_MODEL_DIR, XGB_MODEL_DIR, RESULTS_DIR, REPORTS_DIR
from ml.model_comparison import run_model_comparison, get_best_model_artifacts


EXPERIMENT_NAME = "Career_Prediction_Experiment"
REGISTERED_MODEL_NAME = "CareerPredictionModel"


def setup_mlflow(tracking_uri: Optional[str] = None):
    """
    Configures MLflow tracking URI and initializes experiment.
    Uses local backend suitable for local development without credentials.
    """
    import mlflow
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    if tracking_uri is None:
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        if not tracking_uri:
            db_path = os.path.join(parent_dir, "mlflow.db").replace(os.sep, "/")
            tracking_uri = f"sqlite:///{db_path}"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return mlflow


def log_model_run_to_mlflow(
    model_name: str,
    model_key: str,
    params: Dict[str, Any],
    metrics: Dict[str, Any],
    model_obj: Any,
    vectorizer_obj: Any,
    label_encoder_obj: Any,
    artifacts_dir: Optional[str] = None,
    register_as_best: bool = False
) -> Dict[str, Any]:
    """
    Logs an individual model training run into MLflow with parameters, metrics,
    serialized artifacts, and optionally registers the model in the Model Registry.
    """
    import mlflow
    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run(run_name=model_name) as run:
        run_id = run.info.run_id
        print(f"[MLflow] Started Run: {model_name} (Run ID: {run_id})")

        # 1. Log Hyperparameters
        if params:
            # Flatten or format parameters for MLflow logging
            cleaned_params = {str(k): str(v) for k, v in params.items()}
            mlflow.log_params(cleaned_params)

        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_type", model_key)
        mlflow.log_param("vectorizer_type", "TfidfVectorizer")
        mlflow.log_param("vectorizer_ngram_range", "(1, 2)")
        mlflow.log_param("vectorizer_max_features", 2500)

        # 2. Log Evaluation Metrics (at least accuracy, precision, recall, f1_score)
        mlflow.log_metric("accuracy", float(metrics.get("accuracy", 0.0)))
        mlflow.log_metric("precision", float(metrics.get("precision", 0.0)))
        mlflow.log_metric("recall", float(metrics.get("recall", 0.0)))
        mlflow.log_metric("f1_score", float(metrics.get("f1_score", 0.0)))

        if "cv_best_score" in metrics:
            mlflow.log_metric("cv_best_score", float(metrics["cv_best_score"]))
        if "coverage" in metrics:
            mlflow.log_metric("coverage", float(metrics["coverage"]))

        # 3. Log Artifacts if available
        if artifacts_dir and os.path.exists(artifacts_dir):
            for fname in os.listdir(artifacts_dir):
                fpath = os.path.join(artifacts_dir, fname)
                if os.path.isfile(fpath) and fname.endswith((".json", ".png", ".txt", ".pkl")):
                    mlflow.log_artifact(fpath, artifact_path="evaluation_artifacts")

        # Also log shared reports folder artifacts (e.g. confusion matrix)
        cm_path = os.path.join(REPORTS_DIR, "confusion_matrix.png")
        if os.path.exists(cm_path):
            mlflow.log_artifact(cm_path, artifact_path="visualizations")

        # 4. Log the Trained Model
        try:
            if model_key == "xgboost":
                import mlflow.xgboost
                mlflow.xgboost.log_model(
                    xgb_model=model_obj,
                    artifact_path="model"
                )
            else:
                mlflow.sklearn.log_model(
                    sk_model=model_obj,
                    artifact_path="model"
                )
        except Exception as e:
            print(f"[MLflow] Note on model logging flavor: {e}")
            mlflow.sklearn.log_model(
                sk_model=model_obj,
                artifact_path="model"
            )

        # 5. Register in MLflow Model Registry if flagged as winning model
        registered_version = None
        if register_as_best:
            try:
                model_uri = f"runs:/{run_id}/model"
                reg_model = mlflow.register_model(
                    model_uri=model_uri,
                    name=REGISTERED_MODEL_NAME
                )
                registered_version = reg_model.version
                print(f"[MLflow] Successfully registered model '{REGISTERED_MODEL_NAME}' version {registered_version}!")
            except Exception as reg_err:
                print(f"[MLflow] Registry registration note: {reg_err}")

        print(f"[MLflow] Finished logging run {model_name} [Accuracy: {metrics.get('accuracy', 0.0):.4f}, F1: {metrics.get('f1_score', 0.0):.4f}]")

        return {
            "run_id": run_id,
            "model_name": model_name,
            "metrics": metrics,
            "registered_model": REGISTERED_MODEL_NAME if register_as_best else None,
            "registered_version": registered_version
        }


def track_and_register_all_models() -> Dict[str, Any]:
    """
    Executes or loads model comparison, logs each candidate model into MLflow,
    and registers the best-performing model into the MLflow Model Registry.
    """
    mlflow = setup_mlflow()

    # Ensure model comparison results are up-to-date
    comp_results = run_model_comparison(force_retrain=False)
    best_model_key = comp_results.get("best_model_key", "random_forest")

    run_summaries = []

    # 1. Track Logistic Regression Baseline
    lr_metrics_path = os.path.join(REPORTS_DIR, "metrics.json")
    if os.path.exists(lr_metrics_path):
        with open(lr_metrics_path, "r", encoding="utf-8") as f:
            lr_metrics = json.load(f)
        lr_model = joblib.load(os.path.join(MODELS_DIR, "logistic_regression_model.pkl"))
        lr_vec = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
        lr_le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))

        lr_run = log_model_run_to_mlflow(
            model_name="Logistic_Regression_Baseline",
            model_key="logistic_regression",
            params={"C": "1.0", "max_iter": "1000", "solver": "lbfgs"},
            metrics=lr_metrics,
            model_obj=lr_model,
            vectorizer_obj=lr_vec,
            label_encoder_obj=lr_le,
            artifacts_dir=REPORTS_DIR,
            register_as_best=(best_model_key == "logistic_regression")
        )
        run_summaries.append(lr_run)

    # 2. Track Random Forest
    rf_metrics_path = os.path.join(RF_MODEL_DIR, "rf_metrics.json")
    if os.path.exists(rf_metrics_path):
        with open(rf_metrics_path, "r", encoding="utf-8") as f:
            rf_metrics = json.load(f)
        rf_model = joblib.load(os.path.join(RF_MODEL_DIR, "random_forest_model.pkl"))
        rf_vec = joblib.load(os.path.join(RF_MODEL_DIR, "tfidf_vectorizer.pkl"))
        rf_le = joblib.load(os.path.join(RF_MODEL_DIR, "label_encoder.pkl"))

        rf_run = log_model_run_to_mlflow(
            model_name="Random_Forest_Classifier",
            model_key="random_forest",
            params=rf_metrics.get("best_params", {}),
            metrics=rf_metrics,
            model_obj=rf_model,
            vectorizer_obj=rf_vec,
            label_encoder_obj=rf_le,
            artifacts_dir=RF_MODEL_DIR,
            register_as_best=(best_model_key == "random_forest")
        )
        run_summaries.append(rf_run)

    # 3. Track XGBoost
    xgb_metrics_path = os.path.join(XGB_MODEL_DIR, "xgb_metrics.json")
    if os.path.exists(xgb_metrics_path):
        with open(xgb_metrics_path, "r", encoding="utf-8") as f:
            xgb_metrics = json.load(f)
        xgb_model = joblib.load(os.path.join(XGB_MODEL_DIR, "xgboost_model.pkl"))
        xgb_vec = joblib.load(os.path.join(XGB_MODEL_DIR, "tfidf_vectorizer.pkl"))
        xgb_le = joblib.load(os.path.join(XGB_MODEL_DIR, "label_encoder.pkl"))

        xgb_run = log_model_run_to_mlflow(
            model_name="XGBoost_Classifier",
            model_key="xgboost",
            params=xgb_metrics.get("best_params", {}),
            metrics=xgb_metrics,
            model_obj=xgb_model,
            vectorizer_obj=xgb_vec,
            label_encoder_obj=xgb_le,
            artifacts_dir=XGB_MODEL_DIR,
            register_as_best=(best_model_key == "xgboost")
        )
        run_summaries.append(xgb_run)

    summary = {
        "status": "success",
        "experiment_name": EXPERIMENT_NAME,
        "registered_model_name": REGISTERED_MODEL_NAME,
        "best_model_selected": comp_results.get("best_model"),
        "best_model_key": best_model_key,
        "total_runs_tracked": len(run_summaries),
        "runs": run_summaries
    }

    # Save tracking summary in results/
    mlflow_summary_path = os.path.join(RESULTS_DIR, "mlflow_summary.json")
    with open(mlflow_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    return summary


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("RUNNING MLFLOW EXPERIMENT TRACKING & MODEL REGISTRATION")
    print("=" * 65)
    res = track_and_register_all_models()
    print("\nMLflow Tracking Summary:")
    print(f"  - Experiment: {res['experiment_name']}")
    print(f"  - Registered Model: {res['registered_model_name']}")
    print(f"  - Winning Model: {res['best_model_selected']}")
    print(f"  - Runs Tracked: {res['total_runs_tracked']}")
    print("=" * 65 + "\n")
