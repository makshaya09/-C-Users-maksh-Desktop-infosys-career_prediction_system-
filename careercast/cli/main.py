"""
CareerCast Command Line Interface (CLI).
Provides CLI subcommands for resume parsing, career prediction, recommendations, skill gap analysis, and reports.
"""

import argparse
import json
import os
import sys
from typing import Dict, Any, Optional

# Ensure project root is available on sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
pkg_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(pkg_dir)
for p in [root_dir, pkg_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from nlp.resume_parser import parse_resume
from ml.predict import predict_career
from ml.recommendation_engine import recommend_careers, parse_user_skills, CAREER_REQUIRED_SKILLS
from skill_gap.analyzer import analyze_skill_gap
from careercast.reports.generator import generate_markdown_report, generate_json_report, generate_pdf_report
from careercast.analytics.cohort import compute_career_comparison

VERSION = "1.0.0"


def _build_profile_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Builds a candidate profile dictionary from CLI arguments or parsed resume file."""
    profile: Dict[str, Any] = {
        "name": getattr(args, "name", "Candidate") or "Candidate",
        "email": getattr(args, "email", "candidate@example.com") or "candidate@example.com",
        "education": getattr(args, "education", "B.Tech") or "B.Tech",
        "skills": getattr(args, "skills", "") or "",
        "experience": float(getattr(args, "experience", 1.0) or 1.0),
        "certifications": getattr(args, "certifications", "") or "",
        "projects": getattr(args, "projects", "") or ""
    }

    # If resume file is provided, parse it and overlay extracted fields
    file_path = getattr(args, "file", None)
    if file_path:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Resume file not found: {file_path}")
        parsed = parse_resume(file_path, filename=os.path.basename(file_path))
        if not profile["skills"]:
            profile["skills"] = ", ".join(parsed.get("skills", []))
        if getattr(args, "education", None) is None and parsed.get("primary_education"):
            profile["education"] = parsed["primary_education"]
        if getattr(args, "experience", None) is None and parsed.get("experience"):
            profile["experience"] = float(parsed["experience"])
        if not profile["certifications"] and parsed.get("certifications"):
            profile["certifications"] = ", ".join(parsed["certifications"])
        if not profile["projects"] and parsed.get("projects"):
            profile["projects"] = "; ".join(parsed["projects"])

    # Validate that skills are not completely empty
    if not str(profile["skills"]).strip():
        raise ValueError("Candidate skills must be provided via --skills '...' or --file <resume_path>.")

    return profile


def handle_parse(args: argparse.Namespace) -> int:
    """Handles the 'parse' CLI subcommand."""
    file_path = args.file
    if not file_path:
        print("Error: --file argument is required for resume parsing.", file=sys.stderr)
        return 1

    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        return 1

    try:
        parsed = parse_resume(file_path, filename=os.path.basename(file_path))
        if getattr(args, "json", False):
            print(json.dumps(parsed, indent=2))
        else:
            print("=" * 60)
            print(" RESUME PARSING RESULT")
            print("=" * 60)
            print(f"File: {file_path}")
            print(f"Primary Education: {parsed.get('primary_education', 'N/A')}")
            print(f"Experience: {parsed.get('experience', 0)} years")
            print(f"Skills Found ({len(parsed.get('skills', []))}): {', '.join(parsed.get('skills', []))}")
            print(f"Roles: {', '.join(parsed.get('roles', [])) or 'None'}")
            print(f"Certifications: {', '.join(parsed.get('certifications', [])) or 'None'}")
            print(f"Projects: {'; '.join(parsed.get('projects', [])) or 'None'}")
            print("=" * 60)
        return 0
    except Exception as e:
        print(f"Error parsing resume: {str(e)}", file=sys.stderr)
        return 1


def handle_predict(args: argparse.Namespace) -> int:
    """Handles the 'predict' CLI subcommand."""
    try:
        profile = _build_profile_from_args(args)
    except Exception as e:
        print(f"Validation Error: {str(e)}", file=sys.stderr)
        return 1

    top_k = getattr(args, "top_k", 3) or 3
    try:
        res = predict_career(profile, top_k=top_k)
        if getattr(args, "json", False):
            print(json.dumps(res, indent=2))
        else:
            print("=" * 60)
            print(" CAREER PREDICTION RESULT")
            print("=" * 60)
            print(f"Top Predicted Career: {res['prediction']}")
            print(f"Confidence Score:     {res['confidence_percentage']}% (Probability: {res['confidence']})")
            print("\nRanked Predictions:")
            for idx, r in enumerate(res.get("recommendations", []), 1):
                print(f"  {idx}. {r['career']:<26} {r['percentage']:>5.1f}%  (prob: {r['probability']})")
            print("=" * 60)
        return 0
    except Exception as e:
        print(f"Inference Error: {str(e)}", file=sys.stderr)
        return 1


def handle_recommend(args: argparse.Namespace) -> int:
    """Handles the 'recommend' CLI subcommand."""
    try:
        profile = _build_profile_from_args(args)
    except Exception as e:
        print(f"Validation Error: {str(e)}", file=sys.stderr)
        return 1

    top_k = getattr(args, "top_k", 5) or 5
    try:
        res = recommend_careers(profile, top_k=top_k)
        if getattr(args, "json", False):
            print(json.dumps(res, indent=2))
        else:
            print("=" * 60)
            print(" MULTI-MODAL CAREER RECOMMENDATIONS")
            print("=" * 60)
            print(f"Top Recommendation: {res['prediction']}")
            print(f"Overall Match Score: {res['overall_score']}%")
            print(f"Active Classifier:  {res['active_model']}")
            print(f"Parsed Skills:      {', '.join(res['user_skills'])}")
            print("\nRanked Recommendations:")
            for r in res.get("recommendations", []):
                print(
                    f"  #{r['rank']} {r['career']:<24} "
                    f"Overall: {r['overall_score']:>5.1f}% | "
                    f"ML: {r['confidence_percentage']:>5.1f}% | "
                    f"Skill Align: {r['skill_alignment']:>5.1f}% | "
                    f"Matched: {r['matched_count']}/{r['total_required_count']}"
                )
            print("=" * 60)
        return 0
    except Exception as e:
        print(f"Recommendation Error: {str(e)}", file=sys.stderr)
        return 1


def handle_gap(args: argparse.Namespace) -> int:
    """Handles the 'gap' CLI subcommand."""
    try:
        profile = _build_profile_from_args(args)
    except Exception as e:
        print(f"Validation Error: {str(e)}", file=sys.stderr)
        return 1

    target_career = getattr(args, "career", None)
    try:
        res = analyze_skill_gap(
            candidate_skills=profile["skills"],
            target_career=target_career,
            profile=profile
        )
        if getattr(args, "json", False):
            print(json.dumps(res, indent=2))
        else:
            print("=" * 60)
            print(f" SKILL GAP ANALYSIS: {res['target_career']}")
            print("=" * 60)
            print(f"Competency Match Score: {res['competency_match_score']}%")
            print(f"Skill Gap Percentage:   {res['skill_gap_percentage']}%")
            print(f"Matched Skills ({res['matched_count']}/{res['total_required_count']}): {', '.join(res['matched_skills']) or 'None'}")
            print(f"Missing Skills ({res['missing_count']}): {', '.join(res['missing_skills']) or 'None'}")
            print("\nActionable Improvement Suggestions:")
            for idx, sug in enumerate(res.get("improvement_suggestions", []), 1):
                print(f"  {idx}. [{sug['priority']} Priority] {sug['skill']}: {sug['recommended_action']}")
                print(f"     Project: {sug['practical_project']} (Est: {sug['estimated_time']})")
            print("=" * 60)
        return 0
    except Exception as e:
        print(f"Skill Gap Error: {str(e)}", file=sys.stderr)
        return 1


def handle_report(args: argparse.Namespace) -> int:
    """Handles the 'report' CLI subcommand."""
    try:
        profile = _build_profile_from_args(args)
    except Exception as e:
        print(f"Validation Error: {str(e)}", file=sys.stderr)
        return 1

    target_career = getattr(args, "career", None)
    output_format = getattr(args, "format", "md").lower()
    output_file = getattr(args, "output", None)

    try:
        rec_res = recommend_careers(profile, top_k=5)
        resolved_target = target_career or rec_res.get("prediction", "Data Scientist")
        gap_res = analyze_skill_gap(candidate_skills=profile["skills"], target_career=resolved_target, profile=profile)
        comp_res = compute_career_comparison(candidate_skills=profile["skills"], profile=profile)

        if output_format == "json":
            report_str = generate_json_report(profile, rec_res, gap_res, comparison_results=comp_res)
            if output_file:
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(report_str)
                print(f"JSON report saved to: {output_file}")
            else:
                print(report_str)

        elif output_format == "pdf":
            default_pdf = output_file or "careercast_report.pdf"
            generate_pdf_report(profile, rec_res, gap_res, output_path=default_pdf, comparison_results=comp_res)
            print(f"[OK] Professional PDF report generated and saved to: {default_pdf}")

        else:  # markdown
            report_str = generate_markdown_report(profile, rec_res, gap_res, comparison_results=comp_res)
            if output_file:
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(report_str)
                print(f"Markdown report saved to: {output_file}")
            else:
                print(report_str)

        return 0
    except Exception as e:
        print(f"Report Generation Error: {str(e)}", file=sys.stderr)
        return 1



def handle_serve(args: argparse.Namespace) -> int:
    """Starts the FastAPI REST server."""
    host = getattr(args, "host", "127.0.0.1") or "127.0.0.1"
    port = int(getattr(args, "port", 8000) or 8000)
    print(f"Starting CareerCast FastAPI REST Server on http://{host}:{port} ...")
    print(f"Swagger API Docs: http://{host}:{port}/docs")
    try:
        import uvicorn
        from api.main import app as fastapi_app
        uvicorn.run(fastapi_app, host=host, port=port)
        return 0
    except Exception as e:
        print(f"Failed to start FastAPI server: {str(e)}", file=sys.stderr)
        return 1


def handle_ui(args: argparse.Namespace) -> int:
    """Launches the Streamlit Review UI."""
    port = int(getattr(args, "port", 8501) or 8501)
    print(f"Launching CareerCast Streamlit Review UI on port {port} ...")
    try:
        import subprocess
        streamlit_script = os.path.join(root_dir, "streamlit_app.py")
        subprocess.run([sys.executable, "-m", "streamlit", "run", streamlit_script, "--server.port", str(port)])
        return 0
    except Exception as e:
        print(f"Failed to launch Streamlit UI: {str(e)}", file=sys.stderr)
        return 1


def create_parser() -> argparse.ArgumentParser:
    """Constructs the top-level argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="careercast",
        description="CareerCast — AI-Powered Career Prediction, Recommendation & Skill Gap Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  careercast parse --file resume.pdf
  careercast predict --skills "Python, Machine Learning, SQL" --experience 2.0
  careercast recommend --skills "AWS, Docker, Kubernetes, Linux" --top-k 3
  careercast gap --skills "Python, SQL" --career "Data Scientist"
  careercast report --file resume.pdf --format pdf --output assessment.pdf
  careercast serve --port 8000
  careercast ui --port 8501
"""
    )
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {VERSION}")

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # 1. Parse command
    p_parse = subparsers.add_parser("parse", help="Parse PDF or TXT resume and extract entities")
    p_parse.add_argument("-f", "--file", required=True, help="Path to PDF or TXT resume file")
    p_parse.add_argument("--json", action="store_true", help="Output results in JSON format")

    # 2. Predict command
    p_predict = subparsers.add_parser("predict", help="Predict career domain and top probabilities")
    p_predict.add_argument("-s", "--skills", help="Candidate skills (comma-separated)")
    p_predict.add_argument("-e", "--education", default="B.Tech", help="Candidate education degree")
    p_predict.add_argument("-x", "--experience", type=float, default=1.0, help="Years of experience")
    p_predict.add_argument("-f", "--file", help="Path to resume file (skills will be parsed)")
    p_predict.add_argument("-k", "--top-k", type=int, default=3, help="Number of top predictions to return")
    p_predict.add_argument("--name", default="Candidate", help="Candidate name")
    p_predict.add_argument("--certifications", default="", help="Certifications")
    p_predict.add_argument("--projects", default="", help="Key projects")
    p_predict.add_argument("--json", action="store_true", help="Output results in JSON format")

    # 3. Recommend command
    p_rec = subparsers.add_parser("recommend", help="Generate multi-modal ranked career recommendations")
    p_rec.add_argument("-s", "--skills", help="Candidate skills (comma-separated)")
    p_rec.add_argument("-e", "--education", default="B.Tech", help="Candidate education degree")
    p_rec.add_argument("-x", "--experience", type=float, default=1.0, help="Years of experience")
    p_rec.add_argument("-f", "--file", help="Path to resume file")
    p_rec.add_argument("-k", "--top-k", type=int, default=5, help="Number of recommendations to return")
    p_rec.add_argument("--name", default="Candidate", help="Candidate name")
    p_rec.add_argument("--certifications", default="", help="Certifications")
    p_rec.add_argument("--projects", default="", help="Key projects")
    p_rec.add_argument("--json", action="store_true", help="Output results in JSON format")

    # 4. Gap command
    p_gap = subparsers.add_parser("gap", help="Perform skill gap analysis against target career")
    p_gap.add_argument("-s", "--skills", help="Candidate skills (comma-separated)")
    p_gap.add_argument("-c", "--career", help="Target career category (e.g., 'Data Scientist')")
    p_gap.add_argument("-f", "--file", help="Path to resume file")
    p_gap.add_argument("-e", "--education", default="B.Tech", help="Education degree")
    p_gap.add_argument("-x", "--experience", type=float, default=1.0, help="Years of experience")
    p_gap.add_argument("--json", action="store_true", help="Output results in JSON format")

    # 5. Report command
    p_rep = subparsers.add_parser("report", help="Generate assessment report in Markdown, JSON, or PDF")
    p_rep.add_argument("-s", "--skills", help="Candidate skills (comma-separated)")
    p_rep.add_argument("-f", "--file", help="Path to resume file")
    p_rep.add_argument("-c", "--career", help="Target career domain")
    p_rep.add_argument("-e", "--education", default="B.Tech", help="Education degree")
    p_rep.add_argument("-x", "--experience", type=float, default=1.0, help="Years of experience")
    p_rep.add_argument("--name", default="Candidate", help="Candidate name")
    p_rep.add_argument("--email", default="candidate@example.com", help="Candidate email")
    p_rep.add_argument("--certifications", default="", help="Certifications")
    p_rep.add_argument("--projects", default="", help="Key projects")
    p_rep.add_argument("--format", choices=["md", "json", "pdf"], default="md", help="Output format (md, json, pdf)")
    p_rep.add_argument("-o", "--output", help="Output destination file path")

    # 6. Serve command
    p_srv = subparsers.add_parser("serve", help="Start FastAPI REST service")
    p_srv.add_argument("--host", default="127.0.0.1", help="Host address to bind")
    p_srv.add_argument("-p", "--port", type=int, default=8000, help="Port to bind")

    # 7. UI command
    p_ui = subparsers.add_parser("ui", help="Launch Streamlit Review UI")
    p_ui.add_argument("-p", "--port", type=int, default=8501, help="Port to bind")

    return parser


def main(args: Optional[List[str]] = None) -> int:
    """Main CLI entrypoint."""
    parser = create_parser()
    parsed_args = parser.parse_args(args)

    if not parsed_args.command:
        parser.print_help()
        return 0

    command_handlers = {
        "parse": handle_parse,
        "predict": handle_predict,
        "recommend": handle_recommend,
        "gap": handle_gap,
        "report": handle_report,
        "serve": handle_serve,
        "ui": handle_ui
    }

    handler = command_handlers.get(parsed_args.command)
    if handler:
        return handler(parsed_args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
