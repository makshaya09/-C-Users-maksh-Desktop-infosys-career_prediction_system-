"""
Main Flask Application for AI-Based Career Prediction and Recommendation System.
Milestone 1 & Milestone 2: Advanced ML & Recommendation Engine.
"""

import json
import os
import re
import sys
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_from_directory,
    jsonify
)

# Project base directory configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from nlp.resume_parser import parse_resume
from ml.predict import predict_career
from ml.recommendation_engine import recommend_careers
from ml.model_comparison import run_model_comparison, get_best_model_artifacts
from ml.evaluation import run_full_evaluation_suite
from ml.config import DEFAULT_TOP_K

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "career-prediction-major-project-secret-key")

# Configuration
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
REPORTS_FOLDER = os.path.join(BASE_DIR, "reports")
MODELS_FOLDER = os.path.join(BASE_DIR, "models")
RESULTS_FOLDER = os.path.join(BASE_DIR, "results")
ALLOWED_EXTENSIONS = {"pdf", "txt"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORTS_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)


def allowed_file(filename: str) -> bool:
    """Checks if the uploaded file has a supported extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def validate_profile_data(data: dict):
    """
    Validates user profile input fields.
    Returns (is_valid: bool, errors: list)
    """
    errors = []

    name = str(data.get("name", "") or "").strip()
    if not name:
        errors.append("Full Name cannot be empty.")

    email = str(data.get("email", "") or "").strip()
    email_regex = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
    if not email or not re.match(email_regex, email):
        errors.append("Please provide a valid email address.")

    education = data.get("education", "")
    if isinstance(education, list):
        edu_str = ", ".join(str(e).strip() for e in education if str(e).strip())
    else:
        edu_str = str(education or "").strip()

    if not edu_str:
        errors.append("Education Degree must be selected.")

    skills = data.get("skills", "")
    if isinstance(skills, list):
        skills_clean = [str(s).strip() for s in skills if str(s).strip()]
        if not skills_clean:
            errors.append("At least one technical skill is required.")
    else:
        if not str(skills or "").strip():
            errors.append("At least one technical skill is required.")

    experience = data.get("experience", 0)
    try:
        exp_val = float(experience)
        if exp_val < 0:
            errors.append("Years of experience cannot be negative.")
        elif exp_val > 60:
            errors.append("Years of experience exceeds realistic range.")
    except (ValueError, TypeError):
        errors.append("Years of experience must be a numeric value.")

    return len(errors) == 0, errors


def ensure_models_and_benchmarks_ready():
    """Ensures models and comparison results exist."""
    comp_file = os.path.join(RESULTS_FOLDER, "model_comparison.json")
    if not os.path.exists(comp_file):
        print("[App] Initializing models and comparison benchmark...")
        run_model_comparison()
        run_full_evaluation_suite()


@app.route("/")
def index():
    """Home landing page."""
    return render_template("index.html")


@app.route("/upload", methods=["GET"])
def upload_page():
    """Resume upload page."""
    return render_template("upload_resume.html")


@app.route("/parse", methods=["GET", "POST"])
def parse_resume_route():
    """Handles resume upload and SpaCy parsing."""
    if request.method == "GET":
        return redirect(url_for("upload_page"))

    if "resume" not in request.files:
        flash("No file part provided in the upload request.", "error")
        return redirect(url_for("upload_page"))

    file = request.files["resume"]
    if file.filename == "":
        flash("No file selected. Please select a resume file to upload.", "error")
        return redirect(url_for("upload_page"))

    if not allowed_file(file.filename):
        flash("Unsupported file format. Please upload a PDF or TXT resume.", "error")
        return redirect(url_for("upload_page"))

    try:
        # Extract and parse using SpaCy NLP parser
        parsed_data = parse_resume(file.stream, filename=file.filename)

        # Prepare prefilled profile data
        skills_str = ", ".join(parsed_data.get("skills", []))
        certifications_str = ", ".join(parsed_data.get("certifications", []))
        projects_str = "; ".join(parsed_data.get("projects", []))

        profile_data = {
            "name": "",
            "email": "",
            "education": parsed_data.get("primary_education", "B.Tech"),
            "degree_major": "Computer Science",
            "skills": skills_str,
            "experience": parsed_data.get("experience", 0),
            "certifications": certifications_str,
            "projects": projects_str,
            "preferred_career": ""
        }

        flash("Resume successfully parsed with SpaCy NER! Review and confirm your profile details.", "success")
        return render_template(
            "profile.html",
            profile_data=profile_data,
            parsed_from_resume=True,
            extracted_entities=parsed_data
        )

    except Exception as e:
        flash(f"Error parsing resume: {str(e)}", "error")
        return redirect(url_for("upload_page"))


@app.route("/profile", methods=["GET"])
def profile_page():
    """Manual profile entry page."""
    default_profile = {
        "name": "",
        "email": "",
        "education": "",
        "degree_major": "",
        "skills": "",
        "experience": 0,
        "certifications": "",
        "projects": "",
        "preferred_career": ""
    }
    return render_template("profile.html", profile_data=default_profile, parsed_from_resume=False)


@app.route("/predict", methods=["GET", "POST"])
def predict_route():
    """
    Validates profile and generates Top-K recommendations using the recommendation engine
    (combining best ML model, Sentence-BERT semantic similarity, and skill alignment).
    """
    if request.method == "GET":
        return redirect(url_for("profile_page"))
    form_data = {
        "name": request.form.get("name", "").strip(),
        "email": request.form.get("email", "").strip(),
        "education": request.form.get("education", "").strip(),
        "degree_major": request.form.get("degree_major", "").strip(),
        "skills": request.form.get("skills", "").strip(),
        "experience": request.form.get("experience", "0").strip(),
        "certifications": request.form.get("certifications", "").strip(),
        "projects": request.form.get("projects", "").strip(),
        "preferred_career": request.form.get("preferred_career", "").strip()
    }

    # Validate inputs
    is_valid, errors = validate_profile_data(form_data)
    if not is_valid:
        return render_template(
            "profile.html",
            profile_data=form_data,
            form_errors=errors,
            parsed_from_resume=False
        )

    try:
        # Run recommendation engine (Top 5 recommendations)
        result = recommend_careers(form_data, top_k=DEFAULT_TOP_K)
        return render_template("result.html", result=result, profile=form_data)
    except Exception as e:
        flash(f"Recommendation engine error: {str(e)}", "error")
        return render_template(
            "profile.html",
            profile_data=form_data,
            form_errors=[f"Inference Error: {str(e)}"],
            parsed_from_resume=False
        )


@app.route("/report", methods=["GET"])
def report_page():
    """Model evaluation and benchmark performance report page."""
    ensure_models_and_benchmarks_ready()

    metrics_path = os.path.join(REPORTS_FOLDER, "metrics.json")
    report_text_path = os.path.join(REPORTS_FOLDER, "classification_report.txt")
    comp_file = os.path.join(RESULTS_FOLDER, "model_comparison.json")
    eval_file = os.path.join(RESULTS_FOLDER, "evaluation_report.json")

    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    raw_report = ""
    if os.path.exists(report_text_path):
        with open(report_text_path, "r", encoding="utf-8") as f:
            raw_report = f.read()

    model_comparison = {}
    if os.path.exists(comp_file):
        with open(comp_file, "r", encoding="utf-8") as f:
            model_comparison = json.load(f)

    eval_report = {}
    if os.path.exists(eval_file):
        with open(eval_file, "r", encoding="utf-8") as f:
            eval_report = json.load(f)

    return render_template(
        "report.html",
        metrics=metrics,
        raw_classification_report=raw_report,
        model_comparison=model_comparison,
        eval_report=eval_report
    )


@app.route("/reports/<path:filename>")
def get_report_file(filename):
    """Serves report files like confusion_matrix.png."""
    return send_from_directory(REPORTS_FOLDER, filename)


@app.route("/retrain", methods=["GET", "POST"])
def retrain_model_route():
    """Retrains all ML models (RF, XGB, LR) and updates benchmark metrics."""
    if request.method == "GET":
        return redirect(url_for("report_page"))

    try:
        run_model_comparison(force_retrain=True)
        run_full_evaluation_suite()
        flash("All models successfully retrained, compared, and benchmark reports refreshed!", "success")
    except Exception as e:
        flash(f"Retraining failed: {str(e)}", "error")
    return redirect(url_for("report_page"))


# API Endpoints
@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    """
    Milestone 2 Top-K Career Recommendation API endpoint.
    Expects JSON:
    {
        "skills": ["Python", "Machine Learning", "SQL"],
        "experience": 2,
        "education": "B.Tech",
        "top_k": 5
    }
    """
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json()
    is_valid, errors = validate_profile_data(data)
    if not is_valid:
        return jsonify({"error": "Validation failed", "details": errors}), 422

    top_k = int(data.get("top_k", DEFAULT_TOP_K))

    try:
        res = recommend_careers(data, top_k=top_k)
        return jsonify({
            "status": "success",
            "prediction": res["prediction"],
            "confidence_score": res["confidence_score"],
            "skill_alignment": res["skill_alignment"],
            "semantic_similarity": res["semantic_similarity"],
            "overall_score": res["overall_score"],
            "active_model": res["active_model"],
            "recommendations": res["recommendations"]
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Milestone 1 API endpoint for baseline career prediction."""
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 400

    data = request.get_json()
    is_valid, errors = validate_profile_data(data)
    if not is_valid:
        return jsonify({"error": "Validation failed", "details": errors}), 422

    try:
        res = predict_career(data, top_k=3)
        return jsonify({"status": "success", "data": res}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/model-comparison", methods=["GET"])
def api_model_comparison():
    """Returns model comparison results as JSON."""
    comp_file = os.path.join(RESULTS_FOLDER, "model_comparison.json")
    if not os.path.exists(comp_file):
        run_model_comparison()

    with open(comp_file, "r", encoding="utf-8") as f:
        comp_data = json.load(f)

    return jsonify(comp_data), 200


@app.errorhandler(404)
def page_not_found(e):
    return render_template("index.html"), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template("index.html"), 500


if __name__ == "__main__":
    ensure_models_and_benchmarks_ready()
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    print("\n" + "=" * 60)
    print("AI Career Prediction & Recommendation System (CareerAI)")
    print(f"Starting Flask Web Server at http://{host}:{port}")
    print("=" * 60 + "\n")
    app.run(host=host, port=port, debug=debug)
