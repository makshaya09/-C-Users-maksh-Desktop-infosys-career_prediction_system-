"""
CareerCast — Streamlit Review UI & Analytics Platform
Milestone 3 & Public Release Enhancement.
Interactive web dashboard for career prediction, real ML probability visualization,
multi-modal career recommendations, skill gap analysis, multi-career comparison,
cohort analytics, and multi-format report export (Markdown, JSON, PDF).
"""

import os
import sys
import json
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np

# Ensure project root is on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml.predict import predict_career
from ml.recommendation_engine import recommend_careers, CAREER_REQUIRED_SKILLS, parse_user_skills
from ml.model_comparison import get_best_model_artifacts
from skill_gap.analyzer import analyze_skill_gap
from nlp.resume_parser import parse_resume
from careercast.reports.generator import (
    generate_markdown_report,
    generate_json_report,
    generate_pdf_report
)
from careercast.analytics.cohort import (
    compute_cohort_analytics,
    compute_career_comparison,
    get_sample_cohort
)

# Page Configuration
st.set_page_config(
    page_title="CareerCast — AI Career Prediction & Skill Gap Review",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px;
        color: #f8fafc;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-subtitle {
        font-size: 0.8rem;
        color: #cbd5e1;
        margin-top: 3px;
    }
    
    /* Skill Badge Styling */
    .badge-matched {
        display: inline-block;
        background-color: #064e3b;
        color: #6ee7b7;
        border: 1px solid #059669;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 3px;
    }
    .badge-missing {
        display: inline-block;
        background-color: #451a03;
        color: #fdba74;
        border: 1px solid #d97706;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 3px;
    }
    
    /* Recommendation Card */
    .rec-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 10px;
    }
    .rec-rank {
        background: #0284c7;
        color: white;
        font-weight: bold;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        display: inline-block;
        margin-bottom: 5px;
    }
    .comp-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px;
        height: 100%;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State for Cohort Analytics History
if "cohort_records" not in st.session_state:
    st.session_state["cohort_records"] = get_sample_cohort()


# Presets for quick evaluation
PRESET_PROFILES = {
    "Data Scientist Profile": {
        "name": "Arjun Sharma",
        "email": "arjun.sharma@example.com",
        "education": "M.Tech in Artificial Intelligence",
        "skills": "Python, Machine Learning, Deep Learning, TensorFlow, Pandas, Scikit-learn, SQL",
        "experience": 3.0,
        "certifications": "TensorFlow Developer Certificate, DeepLearning.AI Specialization",
        "projects": "Predictive Churn Model, Transformer NLP Classifier"
    },
    "Web Developer Profile": {
        "name": "Priya Patel",
        "email": "priya.patel@example.com",
        "education": "B.Tech in Computer Engineering",
        "skills": "HTML, CSS, JavaScript, React, Node.js, Express, MongoDB, REST API",
        "experience": 2.0,
        "certifications": "Meta Front-End Developer Professional Certificate",
        "projects": "E-Commerce Web Portal, Real-time Collaborative Task Manager"
    },
    "Cloud Engineer Profile": {
        "name": "Rohan Verma",
        "email": "rohan.verma@example.com",
        "education": "B.Tech in Information Technology",
        "skills": "AWS, Docker, Kubernetes, Linux, Terraform, CI/CD, Git",
        "experience": 2.5,
        "certifications": "AWS Certified Solutions Architect Associate",
        "projects": "Multi-Region VPC Infrastructure, Kubernetes Microservices Deployment"
    },
    "Cybersecurity Analyst Profile": {
        "name": "Neha Gupta",
        "email": "neha.gupta@example.com",
        "education": "B.Tech in Computer Science",
        "skills": "Network Security, Wireshark, Linux, Ethical Hacking, Nmap, Firewall, SIEM",
        "experience": 3.0,
        "certifications": "CompTIA Security+, Certified Ethical Hacker (CEH)",
        "projects": "Enterprise Network Vulnerability Assessment, SIEM SOC Ingestion"
    }
}


def render_sidebar():
    """Renders profile input controls and presets in sidebar."""
    st.sidebar.image("https://img.icons8.com/isometric/100/artificial-intelligence.png", width=55)
    st.sidebar.title("Candidate Profile")
    st.sidebar.caption("CareerCast AI — Prediction, Recommendation & Analytics")

    # Mode selection
    input_mode = st.sidebar.radio(
        "Input Method:",
        ["Quick Preset", "Manual Form Entry", "Upload Resume (PDF/TXT)"],
        index=0
    )

    profile = {}

    if input_mode == "Quick Preset":
        selected_preset = st.sidebar.selectbox("Select Candidate Preset:", list(PRESET_PROFILES.keys()))
        preset_data = PRESET_PROFILES[selected_preset]
        profile = {
            "name": st.sidebar.text_input("Candidate Name", preset_data["name"]),
            "email": st.sidebar.text_input("Candidate Email", preset_data["email"]),
            "education": st.sidebar.text_input("Education Degree", preset_data["education"]),
            "skills": st.sidebar.text_area("Technical Skills (comma-separated)", preset_data["skills"], height=100),
            "experience": st.sidebar.number_input("Years of Experience", min_value=0.0, max_value=50.0, value=preset_data["experience"], step=0.5),
            "certifications": st.sidebar.text_input("Certifications", preset_data["certifications"]),
            "projects": st.sidebar.text_input("Key Projects", preset_data["projects"]),
            "preferred_career": st.sidebar.text_input("Preferred Career (Optional)", "")
        }

    elif input_mode == "Manual Form Entry":
        profile = {
            "name": st.sidebar.text_input("Candidate Name", "John Doe"),
            "email": st.sidebar.text_input("Candidate Email", "john.doe@example.com"),
            "education": st.sidebar.selectbox("Education Degree", ["B.Tech", "M.Tech", "BCA", "MCA", "B.Sc CS", "M.Sc Data Science", "MBA"]),
            "skills": st.sidebar.text_area("Technical Skills (comma-separated)", "Python, SQL, Machine Learning, Pandas", height=100),
            "experience": st.sidebar.number_input("Years of Experience", min_value=0.0, max_value=50.0, value=1.5, step=0.5),
            "certifications": st.sidebar.text_input("Certifications", ""),
            "projects": st.sidebar.text_input("Key Projects", ""),
            "preferred_career": st.sidebar.text_input("Preferred Career (Optional)", "")
        }

    else:
        uploaded_file = st.sidebar.file_uploader("Upload Resume File", type=["pdf", "txt"])
        if uploaded_file is not None:
            try:
                parsed = parse_resume(uploaded_file, filename=uploaded_file.name)
                st.sidebar.success(f"Resume Parsed: {len(parsed.get('skills', []))} skills found!")
                profile = {
                    "name": st.sidebar.text_input("Candidate Name", "Uploaded Candidate"),
                    "email": st.sidebar.text_input("Candidate Email", "candidate@example.com"),
                    "education": st.sidebar.text_input("Education Degree", parsed.get("primary_education", "B.Tech")),
                    "skills": st.sidebar.text_area("Technical Skills", ", ".join(parsed.get("skills", [])), height=100),
                    "experience": st.sidebar.number_input("Years of Experience", min_value=0.0, max_value=50.0, value=float(parsed.get("experience", 1.0)), step=0.5),
                    "certifications": st.sidebar.text_input("Certifications", ", ".join(parsed.get("certifications", []))),
                    "projects": st.sidebar.text_input("Key Projects", "; ".join(parsed.get("projects", []))),
                    "preferred_career": ""
                }
            except Exception as e:
                st.sidebar.error(f"Error parsing resume: {str(e)}")
                profile = {
                    "name": "", "email": "", "education": "B.Tech",
                    "skills": "Python, SQL", "experience": 1.0,
                    "certifications": "", "projects": "", "preferred_career": ""
                }
        else:
            st.sidebar.info("Upload a resume or select a preset to proceed.")
            profile = {
                "name": "Candidate", "email": "candidate@example.com",
                "education": "B.Tech", "skills": "Python, SQL, Machine Learning",
                "experience": 1.0, "certifications": "", "projects": "", "preferred_career": ""
            }

    top_k = st.sidebar.slider("Top Recommendations Count (K)", min_value=3, max_value=7, value=5)
    return profile, top_k


def render_assessment_tab(profile: dict, top_k: int, active_model_name: str):
    """Renders Individual Candidate Assessment & Skill Gap Tab."""
    skills_raw = str(profile.get("skills", "")).strip()
    if not skills_raw:
        st.warning("👈 Please enter or upload candidate skills in the sidebar to run predictions.")
        return

    # Execute ML recommendation and prediction engine
    with st.spinner("Analyzing candidate profile against ML models and semantic embeddings..."):
        try:
            rec_result = recommend_careers(profile, top_k=top_k)
            pred_result = predict_career(profile, top_k=7)  # Get all career probabilities
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return

    top_career = rec_result.get("prediction", "N/A")
    top_score = rec_result.get("overall_score", 0.0)
    top_conf = rec_result.get("confidence_percentage", 0.0)

    # Section 1: Executive Summary Metrics
    st.markdown("### 📊 1. Career Classification & Probability Distribution")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Top Predicted Career</div>
            <div class="metric-value">{top_career}</div>
            <div class="metric-subtitle">Active Model: {active_model_name}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Overall Match Score</div>
            <div class="metric-value">{top_score}%</div>
            <div class="metric-subtitle">Multi-Modal Scoring</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">ML Classification Confidence</div>
            <div class="metric-value">{top_conf}%</div>
            <div class="metric-subtitle">predict_proba() Distribution</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Candidate Skills Extracted</div>
            <div class="metric-value">{len(rec_result.get('user_skills', []))}</div>
            <div class="metric-subtitle">Normalized Competencies</div>
        </div>
        """, unsafe_allow_html=True)

    # Real ML Model Probability Distribution Chart
    col_chart, col_recs = st.columns([1.1, 0.9])

    with col_chart:
        st.markdown("#### Real ML Probability Distribution (All Careers)")
        prob_data = []
        for r in pred_result.get("recommendations", []):
            prob_data.append({
                "Career": r["career"],
                "Probability (%)": r["percentage"],
                "Probability": r["probability"]
            })
        df_probs = pd.DataFrame(prob_data)
        df_probs = df_probs.sort_values(by="Probability (%)", ascending=True)

        st.bar_chart(
            df_probs.set_index("Career")["Probability (%)"],
            horizontal=True,
            color="#0284c7"
        )
        st.caption("🔍 Visualizes actual model probabilities generated from the winning classifier.")

    with col_recs:
        st.markdown(f"#### Top {top_k} Multi-Modal Recommendations")
        for r in rec_result.get("recommendations", []):
            st.markdown(f"""
            <div class="rec-card">
                <span class="rec-rank">Rank #{r['rank']}</span>
                <strong style="font-size: 1.05rem; color: #f8fafc; margin-left: 8px;">{r['career']}</strong>
                <div style="margin-top: 5px; font-size: 0.85rem; color: #94a3b8;">
                    <strong>Overall: <span style="color: #38bdf8;">{r['overall_score']}%</span></strong> | 
                    ML: {r['confidence_percentage']}% | 
                    Skill Align: {r['skill_alignment']}% | 
                    Semantic: {r['semantic_percentage']}%
                </div>
                <div style="margin-top: 3px; font-size: 0.8rem; color: #cbd5e1;">
                    Matched: <span style="color: #6ee7b7;">{r['matched_count']}/{r['total_required_count']} skills</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Section 2: Skill Gap Analysis & Competency Roadmap
    st.markdown("### 🎯 2. Skill Gap Analysis & Actionable Competency Roadmap")

    careers_list = list(CAREER_REQUIRED_SKILLS.keys())
    default_idx = careers_list.index(top_career) if top_career in careers_list else 0

    selected_target_career = st.selectbox(
        "Select Target Career for Skill Gap Breakdown:",
        careers_list,
        index=default_idx
    )

    gap_result = analyze_skill_gap(
        candidate_skills=profile["skills"],
        target_career=selected_target_career,
        profile=profile
    )

    # Save to session cohort records if user clicks 'Add to Cohort History'
    col_g1, col_g2, col_g3, col_g4 = st.columns(4)
    with col_g1:
        st.metric("Competency Match", f"{gap_result['competency_match_score']}%")
    with col_g2:
        st.metric("Skill Gap", f"{gap_result['skill_gap_percentage']}%")
    with col_g3:
        st.metric("Matched Skills", f"{gap_result['matched_count']} / {gap_result['total_required_count']}")
    with col_g4:
        st.metric("Missing Skills", f"{gap_result['missing_count']}")

    st.progress(float(gap_result["competency_match_score"]) / 100.0)

    # Skills Pill Breakdown
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown(f"#### ✅ Matched Skills ({gap_result['matched_count']})")
        if gap_result["matched_skills"]:
            matched_html = "".join(f'<span class="badge-matched">✓ {s}</span>' for s in gap_result["matched_skills"])
            st.markdown(matched_html, unsafe_allow_html=True)
        else:
            st.info("No matching canonical skills found for this career profile.")

    with col_p2:
        st.markdown(f"#### ⚠️ Missing Skills ({gap_result['missing_count']})")
        if gap_result["missing_skills"]:
            missing_html = "".join(f'<span class="badge-missing">✕ {s}</span>' for s in gap_result["missing_skills"])
            st.markdown(missing_html, unsafe_allow_html=True)
        else:
            st.success("🎉 Outstanding! You possess all canonical required skills for this career!")

    # Actionable Improvement Suggestions Roadmap
    st.markdown("#### 🛠️ Actionable Competency Improvement Roadmap")
    suggestions = gap_result.get("improvement_suggestions", [])
    if suggestions:
        for idx, sug in enumerate(suggestions, start=1):
            priority_color = "🔴" if sug["priority"] == "High" else ("🟡" if sug["priority"] == "Medium" else "🟢")
            with st.expander(f"{priority_color} #{idx} {sug['skill']} ({sug['priority']} Priority — {sug['category']}) — Est. {sug['estimated_time']}", expanded=(idx <= 2)):
                st.markdown(f"**Recommended Action**: {sug['recommended_action']}")
                st.markdown(f"**Practical Project Idea**: `{sug['practical_project']}`")
                st.markdown("**Curated Learning Resources & Documentation**:")
                for r in sug.get("learning_resources", []):
                    st.markdown(f"- 🔗 {r}")
    else:
        st.info("No skill gaps identified! The candidate meets all canonical requirements.")

    st.markdown("---")

    # Section 3: Export & Download Career Assessment Reports
    st.markdown("### 📥 3. Export Career Assessment Reports")

    comparison_results = compute_career_comparison(candidate_skills=profile["skills"], profile=profile)
    json_report_str = generate_json_report(profile, rec_result, gap_result, active_model_name, comparison_results=comparison_results)
    md_report_str = generate_markdown_report(profile, rec_result, gap_result, active_model_name, comparison_results=comparison_results)

    col_exp1, col_exp2, col_exp3 = st.columns(3)

    with col_exp1:
        st.download_button(
            label="📄 Download Markdown Report (.md)",
            data=md_report_str,
            file_name=f"career_assessment_{profile.get('name', 'candidate').lower().replace(' ', '_')}.md",
            mime="text/markdown",
            use_container_width=True
        )

    with col_exp2:
        st.download_button(
            label="📦 Download JSON Report (.json)",
            data=json_report_str,
            file_name=f"career_assessment_{profile.get('name', 'candidate').lower().replace(' ', '_')}.json",
            mime="application/json",
            use_container_width=True
        )

    with col_exp3:
        try:
            pdf_bytes = generate_pdf_report(
                profile,
                rec_result,
                gap_result,
                model_name=active_model_name,
                comparison_results=comparison_results
            )
            st.download_button(
                label="📑 Download PDF Report (.pdf)",
                data=pdf_bytes,
                file_name=f"career_assessment_{profile.get('name', 'candidate').lower().replace(' ', '_')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        except Exception as pdf_err:
            st.warning(f"PDF export unavailable: {str(pdf_err)}")

    # Add to cohort history button
    if st.button("➕ Add Current Assessment to Cohort History"):
        new_record = {
            "name": profile.get("name", "Candidate"),
            "education": profile.get("education", "B.Tech"),
            "experience": float(profile.get("experience", 1.0)),
            "skills": profile.get("skills", ""),
            "prediction": top_career,
            "confidence_percentage": top_conf,
            "competency_match_score": gap_result.get("competency_match_score", 0.0),
            "missing_skills": gap_result.get("missing_skills", [])
        }
        st.session_state["cohort_records"].append(new_record)
        st.success(f"Added {profile.get('name', 'Candidate')} to Cohort Analytics! (Total: {len(st.session_state['cohort_records'])})")


def render_comparison_tab(profile: dict):
    """Renders Multi-Career Comparison View."""
    st.markdown("### ⚖️ Multi-Career Comparative Evaluation")
    st.caption("Compare candidate competency, ML classification probabilities, matching skills, and missing gaps across multiple target career paths.")

    skills_raw = str(profile.get("skills", "")).strip()
    if not skills_raw:
        st.info("👈 Please enter or upload candidate skills in the sidebar to compare careers.")
        return

    all_careers = list(CAREER_REQUIRED_SKILLS.keys())
    selected_careers = st.multiselect(
        "Select Careers to Compare:",
        options=all_careers,
        default=all_careers[:4]
    )

    if not selected_careers:
        st.warning("Please select at least one career to view comparison.")
        return

    with st.spinner("Computing multi-career alignment and probability scores..."):
        comparisons = compute_career_comparison(
            candidate_skills=profile["skills"],
            target_careers=selected_careers,
            profile=profile
        )

    # 1. Comparative Bar Charts
    df_comp = pd.DataFrame([
        {
            "Career": c["career"],
            "Overall Score (%)": c["overall_score"],
            "ML Probability (%)": c["probability_percentage"],
            "Skill Alignment (%)": c["alignment_score"],
            "Competency Match (%)": c["competency_match_score"],
            "Matched Skills": f"{c['matched_count']}/{c['total_required_count']}"
        }
        for c in comparisons
    ])

    st.markdown("#### Overall Score vs Skill Alignment Comparison")
    st.bar_chart(
        df_comp.set_index("Career")[["Overall Score (%)", "Skill Alignment (%)", "ML Probability (%)"]],
        horizontal=False
    )

    # 2. Side-by-Side Detail Cards
    st.markdown("#### Side-by-Side Detailed Breakdown")
    cols = st.columns(len(comparisons))
    for idx, c in enumerate(comparisons):
        with cols[idx]:
            st.markdown(f"""
            <div class="comp-card">
                <h4 style="color: #38bdf8; margin-bottom: 4px;">{c['career']}</h4>
                <div style="font-size: 0.85rem; color: #94a3b8; margin-bottom: 8px;">Rank #{c['rank']} in comparison</div>
                <hr style="border-color: #334155; margin: 8px 0;"/>
                <p><b>Overall Match:</b> <span style="color: #38bdf8; font-size: 1.1rem; font-weight: bold;">{c['overall_score']}%</span></p>
                <p><b>ML Prob:</b> {c['probability_percentage']}%</p>
                <p><b>Alignment:</b> {c['alignment_score']}%</p>
                <p><b>Competency Match:</b> {c['competency_match_score']}%</p>
                <hr style="border-color: #334155; margin: 8px 0;"/>
                <p style="font-size: 0.85rem; color: #6ee7b7;"><b>✓ Matched ({c['matched_count']}):</b><br/>{', '.join(c['matched_skills']) if c['matched_skills'] else 'None'}</p>
                <p style="font-size: 0.85rem; color: #fdba74;"><b>✕ Missing ({c['missing_count']}):</b><br/>{', '.join(c['missing_skills']) if c['missing_skills'] else 'None'}</p>
            </div>
            """, unsafe_allow_html=True)

    # 3. Comparison Table
    st.markdown("#### Summary Comparison Table")
    st.dataframe(df_comp, use_container_width=True)


def render_cohort_tab():
    """Renders Cohort Analytics & Candidate Distribution Tab."""
    st.markdown("### 📈 Cohort Analytics & Candidate Insights")
    st.caption("Aggregated statistical overview, career distributions, skill frequencies, and competency bottleneck analysis across candidate cohort.")

    cohort_records = st.session_state.get("cohort_records", [])

    # Action buttons
    col_act1, col_act2, col_act3 = st.columns([1, 1, 2])
    with col_act1:
        if st.button("🔄 Reset to Default Sample Cohort"):
            st.session_state["cohort_records"] = get_sample_cohort()
            st.rerun()
    with col_act2:
        if st.button("🗑️ Clear Cohort History"):
            st.session_state["cohort_records"] = []
            st.rerun()

    analytics = compute_cohort_analytics(cohort_records)

    # Handle empty dataset gracefully
    if analytics["total_candidates"] == 0:
        st.info("ℹ️ No candidates currently in the cohort dataset. Add candidates from the Assessment tab or click 'Reset to Default Sample Cohort' above.")
        return

    # 1. Top-Level Summary Metrics
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Candidates</div>
            <div class="metric-value">{analytics['total_candidates']}</div>
            <div class="metric-subtitle">Assessed Profiles</div>
        </div>
        """, unsafe_allow_html=True)
    with col_s2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Avg. Experience</div>
            <div class="metric-value">{analytics['average_experience']} <font size='4'>yrs</font></div>
            <div class="metric-subtitle">Across All Domains</div>
        </div>
        """, unsafe_allow_html=True)
    with col_s3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Avg. ML Confidence</div>
            <div class="metric-value">{analytics['average_confidence']}%</div>
            <div class="metric-subtitle">Model Certainty</div>
        </div>
        """, unsafe_allow_html=True)
    with col_s4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Avg. Competency Match</div>
            <div class="metric-value">{analytics['average_competency_score']}%</div>
            <div class="metric-subtitle">Skill Alignment</div>
        </div>
        """, unsafe_allow_html=True)

    # 2. Charts Section
    col_c1, col_c2 = st.columns(2)

    with col_c1:
        st.markdown("#### 🎯 Career Distribution Across Cohort")
        if analytics["career_distribution"]:
            df_careers = pd.DataFrame(
                list(analytics["career_distribution"].items()),
                columns=["Career", "Candidates Count"]
            ).set_index("Career")
            st.bar_chart(df_careers["Candidates Count"], color="#0284c7")
        else:
            st.info("No career distribution data available.")

    with col_c2:
        st.markdown("#### 📊 Confidence Score Distribution")
        df_conf = pd.DataFrame(
            list(analytics["confidence_bins"].items()),
            columns=["Confidence Range", "Count"]
        ).set_index("Confidence Range")
        st.bar_chart(df_conf["Count"], color="#059669")

    # 3. Skills Analysis
    col_sk1, col_sk2 = st.columns(2)
    with col_sk1:
        st.markdown("#### 🛠️ Most Frequent Candidate Skills")
        if analytics["top_candidate_skills"]:
            df_skills = pd.DataFrame(
                list(analytics["top_candidate_skills"].items()),
                columns=["Skill", "Count"]
            ).set_index("Skill")
            st.bar_chart(df_skills["Count"], horizontal=True, color="#38bdf8")
        else:
            st.info("No skill frequency data available.")

    with col_sk2:
        st.markdown("#### ⚠️ Top Missing Skill Bottlenecks")
        if analytics["top_missing_skills"]:
            df_missing = pd.DataFrame(
                list(analytics["top_missing_skills"].items()),
                columns=["Missing Skill", "Frequency"]
            ).set_index("Missing Skill")
            st.bar_chart(df_missing["Frequency"], horizontal=True, color="#d97706")
        else:
            st.info("No missing skill bottleneck data available.")

    # 4. Candidate Summary Table
    st.markdown("#### 📋 Candidate Cohort Records")
    if analytics["records_summary"]:
        df_summary = pd.DataFrame(analytics["records_summary"])
        st.dataframe(df_summary, use_container_width=True)
        csv_data = df_summary.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Cohort Summary (CSV)",
            data=csv_data,
            file_name="careercast_cohort_summary.csv",
            mime="text/csv"
        )


def main():
    # Header
    st.title("🚀 CareerCast — AI Career Prediction, Recommendation & Analytics Platform")
    st.caption("Production AI Platform: Machine Learning Classifiers, Semantic Sentence-BERT, Skill Gap Engine, Cohort Analytics & PDF Export")

    # Load active model name
    try:
        _, _, _, active_model_name = get_best_model_artifacts()
    except Exception:
        active_model_name = "Random Forest / Ensemble"

    # Sidebar inputs
    profile, top_k = render_sidebar()

    # Main Navigation Tabs
    tab_assess, tab_compare, tab_cohort = st.tabs([
        "🎯 Candidate Assessment",
        "⚖️ Multi-Career Comparison",
        "📈 Cohort Analytics"
    ])

    with tab_assess:
        render_assessment_tab(profile, top_k, active_model_name)

    with tab_compare:
        render_comparison_tab(profile)

    with tab_cohort:
        render_cohort_tab()


if __name__ == "__main__":
    main()
