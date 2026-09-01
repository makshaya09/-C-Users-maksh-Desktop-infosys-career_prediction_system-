"""
Report Generation Module for CareerCast.
Generates comprehensive career assessment reports in Markdown, JSON, and professional PDF formats.
"""

import io
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List, Union

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.colors import HexColor
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        KeepTogether,
        HRFlowable
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    canvas = None



def generate_markdown_report(
    profile: Dict[str, Any],
    rec_result: Dict[str, Any],
    gap_result: Dict[str, Any],
    model_name: Optional[str] = None,
    comparison_results: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Generates a formatted Markdown report of the career assessment,
    including candidate profile, predictions, recommendations, skill gap analysis,
    and optional multi-career comparison.

    Parameters:
    - profile: Candidate profile dictionary (name, email, education, experience, skills, etc.)
    - rec_result: Recommendation engine result dictionary
    - gap_result: Skill gap analysis result dictionary
    - model_name: Name of active classifier model
    - comparison_results: Optional list of career comparison results

    Returns:
    - Formatted Markdown string.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    active_model = model_name or rec_result.get("active_model", "Ensemble Classifier (Random Forest / XGBoost)")

    report = f"""# AI Career Assessment & Skill Gap Analysis Report
**Generated On**: {timestamp}  
**Active Classifier**: {active_model}

---

## 1. Candidate Information
- **Name**: {profile.get('name', 'Candidate')}
- **Email**: {profile.get('email', 'N/A')}
- **Education**: {profile.get('education', 'N/A')}
- **Experience**: {profile.get('experience', 0)} years
- **Certifications**: {profile.get('certifications', 'None')}
- **Projects**: {profile.get('projects', 'None')}
- **Candidate Skills**: {', '.join(gap_result.get('candidate_skills', [])) if gap_result.get('candidate_skills') else 'None'}

---

## 2. Career Classification & Recommendations
- **Top Predicted Career**: **{rec_result.get('prediction', 'N/A')}**
- **Classification Confidence**: {rec_result.get('confidence_percentage', 0.0)}%
- **Overall Multi-Modal Score**: {rec_result.get('overall_score', 0.0)}%

### Top-K Ranked Career Recommendations
"""
    for r in rec_result.get("recommendations", []):
        report += (
            f"- **Rank #{r.get('rank', '-')} {r.get('career', '')}**: "
            f"Overall Score {r.get('overall_score', 0)}% "
            f"(ML Confidence: {r.get('confidence_percentage', 0)}%, "
            f"Skill Alignment: {r.get('skill_alignment', 0)}%, "
            f"Semantic: {r.get('semantic_percentage', 0)}%, "
            f"Matched: {r.get('matched_count', 0)}/{r.get('total_required_count', 0)} skills)\n"
        )

    report += f"""
---

## 3. Skill Gap Analysis for Target Domain: {gap_result.get('target_career', 'Target Career')}
- **Competency Match Score**: {gap_result.get('competency_match_score', 0)}%
- **Skill Gap Percentage**: {gap_result.get('skill_gap_percentage', 0)}%
- **Matched Skills ({gap_result.get('matched_count', 0)}/{gap_result.get('total_required_count', 0)})**: {', '.join(gap_result.get('matched_skills', [])) if gap_result.get('matched_skills') else 'None'}
- **Missing Skills ({gap_result.get('missing_count', 0)})**: {', '.join(gap_result.get('missing_skills', [])) if gap_result.get('missing_skills') else 'None'}

---

## 4. Actionable Competency Improvement Roadmap
"""
    suggestions = gap_result.get("improvement_suggestions", [])
    if suggestions:
        for idx, sug in enumerate(suggestions, 1):
            report += f"""
### {idx}. {sug.get('skill', '')} [{sug.get('priority', 'Medium')} Priority — {sug.get('category', 'Technical')}]
- **Recommended Action**: {sug.get('recommended_action', 'Learn core principles')}
- **Estimated Timeline**: {sug.get('estimated_time', '2-4 weeks')}
- **Practical Project Idea**: {sug.get('practical_project', 'Build hands-on implementation')}
- **Curated Learning Resources**:
"""
            for res in sug.get("learning_resources", []):
                report += f"  * {res}\n"
    else:
        report += "\n*No skill gaps identified! The candidate meets all canonical requirements.*\n"

    if comparison_results:
        report += """
---

## 5. Multi-Career Comparative Analysis
| Career | Overall Score | ML Prob | Alignment | Matched Skills | Missing Skills |
|---|---|---|---|---|---|
"""
        for comp in comparison_results:
            c_name = comp.get("career", "")
            ovr = comp.get("overall_score", 0)
            ml_p = comp.get("probability_percentage", comp.get("confidence_percentage", 0))
            align = comp.get("alignment_score", 0)
            matched_str = ", ".join(comp.get("matched_skills", [])) or "None"
            missing_str = ", ".join(comp.get("missing_skills", [])) or "None"
            report += f"| **{c_name}** | {ovr}% | {ml_p}% | {align}% | {matched_str} | {missing_str} |\n"

    report += """
---
*Report generated by CareerCast AI Career Prediction & Skill Gap Recommendation System.*
"""
    return report


def generate_json_report(
    profile: Dict[str, Any],
    rec_result: Dict[str, Any],
    gap_result: Dict[str, Any],
    model_name: Optional[str] = None,
    comparison_results: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Generates a structured JSON string containing full career assessment metadata,
    candidate profile, prediction results, skill gap breakdown, and recommendations.
    """
    active_model = model_name or rec_result.get("active_model", "Ensemble Classifier")
    report_dict = {
        "report_metadata": {
            "system": "CareerCast Career Prediction & Skill Gap System",
            "version": "1.0.0",
            "generated_at": datetime.now().isoformat(),
            "active_model": active_model
        },
        "candidate_profile": profile,
        "recommendation_results": rec_result,
        "skill_gap_analysis": gap_result,
        "career_comparisons": comparison_results or []
    }
    return json.dumps(report_dict, indent=4)


if REPORTLAB_AVAILABLE and canvas is not None:
    class NumberedCanvas(canvas.Canvas):
        """Custom canvas to add dynamic running header and page numbers."""
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._saved_page_states = []

        def showPage(self):
            self._saved_page_states.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            num_pages = len(self._saved_page_states)
            for state in self._saved_page_states:
                self.__dict__.update(state)
                self.draw_page_decorations(num_pages)
                super().showPage()
            super().save()

        def draw_page_decorations(self, page_count: int):
            self.saveState()
            self.setFont("Helvetica", 8)
            self.setFillColor(HexColor("#64748b"))

            # Running header (pages > 1)
            if self._pageNumber > 1:
                self.drawString(40, 760, "CareerCast — AI Career Assessment & Skill Gap Report")
                self.setStrokeColor(HexColor("#e2e8f0"))
                self.setLineWidth(0.5)
                self.line(40, 752, 572, 752)

            # Running footer
            footer_text = f"Page {self._pageNumber} of {page_count}"
            self.drawRightString(572, 30, footer_text)
            self.drawString(40, 30, "Confidential — Generated by CareerCast AI System")
            self.setStrokeColor(HexColor("#e2e8f0"))
            self.setLineWidth(0.5)
            self.line(40, 42, 572, 42)
            self.restoreState()
else:
    NumberedCanvas = None


def generate_pdf_report(
    profile: Dict[str, Any],
    rec_result: Dict[str, Any],
    gap_result: Dict[str, Any],
    model_name: Optional[str] = None,
    output_path: Optional[str] = None,
    comparison_results: Optional[List[Dict[str, Any]]] = None
) -> bytes:
    """
    Generates a professional, beautifully styled multi-page PDF career assessment report.


    Parameters:
    - profile: Candidate profile dictionary
    - rec_result: Recommendations result dictionary
    - gap_result: Skill gap result dictionary
    - model_name: Active model name
    - output_path: Optional path to save the PDF file
    - comparison_results: Optional career comparison list

    Returns:
    - PDF document as bytes.
    """
    if not REPORTLAB_AVAILABLE:
        raise ImportError("ReportLab is required for PDF report generation. Run 'pip install reportlab'.")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=50,
        bottomMargin=50
    )

    styles = getSampleStyleSheet()
    primary_color = HexColor("#0f172a")     # Deep navy slate
    brand_blue = HexColor("#0284c7")        # Sky blue
    success_color = HexColor("#059669")     # Emerald green
    warning_color = HexColor("#d97706")     # Amber
    dark_gray = HexColor("#334155")         # Slate 700
    light_bg = HexColor("#f8fafc")          # Slate 50
    border_color = HexColor("#cbd5e1")      # Slate 300

    # Custom typography styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=primary_color,
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        textColor=HexColor("#64748b"),
        spaceAfter=12
    )
    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=primary_color,
        spaceBefore=10,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=dark_gray
    )
    body_bold = ParagraphStyle(
        "ReportBodyBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=primary_color
    )
    table_cell = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=dark_gray
    )
    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=primary_color
    )
    badge_matched = ParagraphStyle(
        "BadgeMatched",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=success_color
    )
    badge_missing = ParagraphStyle(
        "BadgeMissing",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=warning_color
    )

    story = []

    # 1. Header Banner
    story.append(Paragraph("🚀 CareerCast — AI Career Assessment Report", title_style))
    active_model = model_name or rec_result.get("active_model", "Ensemble Classifier")
    timestamp = datetime.now().strftime("%B %d, %Y • %I:%M %p")
    story.append(Paragraph(f"Generated: {timestamp} | Active Model: <b>{active_model}</b>", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=brand_blue, spaceBefore=0, spaceAfter=10))

    # 2. Candidate Information Card
    story.append(Paragraph("1. Candidate Profile Overview", h2_style))
    cand_info = [
        [
            Paragraph("<b>Candidate Name:</b>", table_cell_bold),
            Paragraph(str(profile.get("name", "Candidate")), table_cell),
            Paragraph("<b>Email:</b>", table_cell_bold),
            Paragraph(str(profile.get("email", "N/A")), table_cell),
        ],
        [
            Paragraph("<b>Education:</b>", table_cell_bold),
            Paragraph(str(profile.get("education", "N/A")), table_cell),
            Paragraph("<b>Experience:</b>", table_cell_bold),
            Paragraph(f"{profile.get('experience', 0)} Years", table_cell),
        ],
        [
            Paragraph("<b>Certifications:</b>", table_cell_bold),
            Paragraph(str(profile.get("certifications", "None")), table_cell),
            Paragraph("<b>Key Projects:</b>", table_cell_bold),
            Paragraph(str(profile.get("projects", "None")), table_cell),
        ],
        [
            Paragraph("<b>Candidate Skills:</b>", table_cell_bold),
            Paragraph(", ".join(gap_result.get("candidate_skills", [])) or str(profile.get("skills", "None")), table_cell),
            Paragraph("", table_cell),
            Paragraph("", table_cell),
        ]
    ]
    cand_table = Table(cand_info, colWidths=[100, 166, 90, 176])
    cand_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), light_bg),
        ('BOX', (0, 0), (-1, -1), 0.5, border_color),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, HexColor("#e2e8f0")),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('SPAN', (1, 3), (3, 3)),
    ]))
    story.append(cand_table)
    story.append(Spacer(1, 10))

    # 3. Top Prediction Summary Metrics
    story.append(Paragraph("2. Career Classification & Top Recommendations", h2_style))
    top_career = rec_result.get("prediction", "N/A")
    top_score = rec_result.get("overall_score", 0.0)
    top_conf = rec_result.get("confidence_percentage", 0.0)

    summary_metrics = [
        [
            Paragraph(f"<font size='7' color='#64748b'>TOP PREDICTED CAREER</font><br/><b><font size='13' color='#0284c7'>{top_career}</font></b>", styles["Normal"]),
            Paragraph(f"<font size='7' color='#64748b'>OVERALL MATCH SCORE</font><br/><b><font size='13' color='#059669'>{top_score}%</font></b>", styles["Normal"]),
            Paragraph(f"<font size='7' color='#64748b'>ML CONFIDENCE</font><br/><b><font size='13' color='#0f172a'>{top_conf}%</font></b>", styles["Normal"]),
        ]
    ]
    summary_table = Table(summary_metrics, colWidths=[180, 176, 176])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), light_bg),
        ('BOX', (0, 0), (-1, -1), 1, brand_blue),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 8))

    # Recommendations Table
    rec_rows = [
        [
            Paragraph("Rank", table_cell_bold),
            Paragraph("Career Domain", table_cell_bold),
            Paragraph("Overall Score", table_cell_bold),
            Paragraph("ML Conf", table_cell_bold),
            Paragraph("Skill Align", table_cell_bold),
            Paragraph("Semantic", table_cell_bold),
            Paragraph("Matched Skills", table_cell_bold),
        ]
    ]
    for r in rec_result.get("recommendations", []):
        rec_rows.append([
            Paragraph(f"#{r.get('rank', '-')}", table_cell),
            Paragraph(f"<b>{r.get('career', '')}</b>", table_cell),
            Paragraph(f"<b>{r.get('overall_score', 0)}%</b>", table_cell_bold),
            Paragraph(f"{r.get('confidence_percentage', 0)}%", table_cell),
            Paragraph(f"{r.get('skill_alignment', 0)}%", table_cell),
            Paragraph(f"{r.get('semantic_percentage', 0)}%", table_cell),
            Paragraph(f"{r.get('matched_count', 0)}/{r.get('total_required_count', 0)}", table_cell),
        ])

    rec_table = Table(rec_rows, colWidths=[40, 150, 75, 60, 65, 65, 77])
    rec_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), HexColor("#e2e8f0")),
        ('BOX', (0, 0), (-1, -1), 0.5, border_color),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, HexColor("#e2e8f0")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor("#ffffff"), light_bg]),
    ]))
    story.append(rec_table)
    story.append(Spacer(1, 10))

    # 4. Skill Gap Analysis Section
    target_c = gap_result.get("target_career", top_career)
    story.append(Paragraph(f"3. Skill Gap Analysis (Target: {target_c})", h2_style))

    gap_metrics = [
        [
            Paragraph(f"<b>Competency Match:</b> {gap_result.get('competency_match_score', 0)}%", table_cell_bold),
            Paragraph(f"<b>Skill Gap:</b> {gap_result.get('skill_gap_percentage', 0)}%", table_cell_bold),
            Paragraph(f"<b>Matched:</b> {gap_result.get('matched_count', 0)} of {gap_result.get('total_required_count', 0)}", table_cell),
            Paragraph(f"<b>Missing:</b> {gap_result.get('missing_count', 0)} skills", table_cell),
        ]
    ]
    gap_metric_table = Table(gap_metrics, colWidths=[130, 130, 136, 136])
    gap_metric_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), light_bg),
        ('BOX', (0, 0), (-1, -1), 0.5, border_color),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(gap_metric_table)
    story.append(Spacer(1, 6))

    # Matched vs Missing Skills breakdown
    matched_list = gap_result.get("matched_skills", [])
    missing_list = gap_result.get("missing_skills", [])

    matched_text = ", ".join([f"✓ {s}" for s in matched_list]) if matched_list else "None identified"
    missing_text = ", ".join([f"✕ {s}" for s in missing_list]) if missing_list else "None! Full competency match achieved."

    skills_breakdown = [
        [
            Paragraph("<b>✅ Matched Competencies:</b>", table_cell_bold),
            Paragraph(matched_text, badge_matched),
        ],
        [
            Paragraph("<b>⚠️ Missing / Skill Gaps:</b>", table_cell_bold),
            Paragraph(missing_text, badge_missing),
        ]
    ]
    breakdown_table = Table(skills_breakdown, colWidths=[140, 392])
    breakdown_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), HexColor("#f1f5f9")),
        ('BOX', (0, 0), (-1, -1), 0.5, border_color),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, HexColor("#e2e8f0")),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(breakdown_table)
    story.append(Spacer(1, 10))

    # 5. Actionable Competency Improvement Roadmap
    story.append(Paragraph("4. Actionable Competency Improvement Roadmap", h2_style))
    suggestions = gap_result.get("improvement_suggestions", [])
    if suggestions:
        sug_rows = [
            [
                Paragraph("Skill & Priority", table_cell_bold),
                Paragraph("Recommended Action", table_cell_bold),
                Paragraph("Timeline", table_cell_bold),
                Paragraph("Hands-On Project", table_cell_bold),
            ]
        ]
        for sug in suggestions:
            priority = sug.get("priority", "Medium")
            color_hex = "#dc2626" if priority == "High" else ("#d97706" if priority == "Medium" else "#059669")
            sug_rows.append([
                Paragraph(f"<b>{sug.get('skill', '')}</b><br/><font color='{color_hex}'><b>[{priority} Priority]</b></font>", table_cell),
                Paragraph(str(sug.get("recommended_action", "")), table_cell),
                Paragraph(str(sug.get("estimated_time", "2-4 wks")), table_cell),
                Paragraph(str(sug.get("practical_project", "")), table_cell),
            ])

        sug_table = Table(sug_rows, colWidths=[110, 190, 72, 160])
        sug_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor("#e2e8f0")),
            ('BOX', (0, 0), (-1, -1), 0.5, border_color),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, HexColor("#e2e8f0")),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor("#ffffff"), light_bg]),
        ]))
        story.append(KeepTogether([sug_table]))
    else:
        story.append(Paragraph("<i>No skill gaps identified. Candidate is fully prepared for target role.</i>", body_style))

    # 6. Optional Multi-Career Comparison Section
    if comparison_results:
        story.append(Spacer(1, 10))
        story.append(Paragraph("5. Multi-Career Comparative Overview", h2_style))
        comp_rows = [
            [
                Paragraph("Career", table_cell_bold),
                Paragraph("Overall Score", table_cell_bold),
                Paragraph("ML Prob", table_cell_bold),
                Paragraph("Alignment", table_cell_bold),
                Paragraph("Matched Skills", table_cell_bold),
                Paragraph("Missing Skills", table_cell_bold),
            ]
        ]
        for comp in comparison_results:
            c_name = comp.get("career", "")
            ovr = comp.get("overall_score", 0)
            ml_p = comp.get("probability_percentage", comp.get("confidence_percentage", 0))
            align = comp.get("alignment_score", 0)
            matched_s = ", ".join(comp.get("matched_skills", [])) or "None"
            missing_s = ", ".join(comp.get("missing_skills", [])) or "None"
            comp_rows.append([
                Paragraph(f"<b>{c_name}</b>", table_cell),
                Paragraph(f"<b>{ovr}%</b>", table_cell_bold),
                Paragraph(f"{ml_p}%", table_cell),
                Paragraph(f"{align}%", table_cell),
                Paragraph(matched_s, table_cell),
                Paragraph(missing_s, table_cell),
            ])

        comp_table = Table(comp_rows, colWidths=[100, 60, 50, 55, 135, 132])
        comp_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor("#e2e8f0")),
            ('BOX', (0, 0), (-1, -1), 0.5, border_color),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, HexColor("#e2e8f0")),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [HexColor("#ffffff"), light_bg]),
        ]))
        story.append(KeepTogether([comp_table]))

    # Build Document with running headers/footers
    doc.build(story, canvasmaker=NumberedCanvas)
    pdf_bytes = buffer.getvalue()
    buffer.close()

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)

    return pdf_bytes
