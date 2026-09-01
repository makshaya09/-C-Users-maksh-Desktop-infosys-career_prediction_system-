# Streamlit Review UI & Analytics Guide — CareerCast

The CareerCast Streamlit Review UI is an interactive, multi-view dashboard designed for career counselors, hiring managers, candidates, and AI evaluators.

---

## 1. Launching the Streamlit Application

```bash
# Via CareerCast CLI
careercast ui --port 8501

# Or via Streamlit directly
streamlit run streamlit_app.py
```

The web dashboard is hosted at `http://localhost:8501`.

---

## 2. Main Dashboard Views

### Tab 1: 🎯 Candidate Assessment & Skill Gap Review
- **Sidebar Profile Entry**:
  - **Quick Presets**: Pre-loaded profiles across Data Science, Web Development, Cloud Engineering, Cybersecurity.
  - **Manual Form Entry**: Custom entry of Name, Email, Degree, Skills, Experience, Certifications, Projects.
  - **Resume File Upload**: Instant parsing of `.pdf` and `.txt` files with automatic skill and entity extraction.
- **Real ML Probability Distribution**: Horizontal bar chart visualizing actual classifier probability outputs across all 7 career categories.
- **Top-K Ranked Recommendations**: Cards displaying multi-modal composite score, ML confidence %, skill alignment %, and matched skill counts.
- **Interactive Skill Gap Breakdown**: Target career selector, competency match score %, skill gap %, green matched skill badges (`✓ Python`), orange missing skill badges (`✕ TensorFlow`).
- **Actionable Competency Improvement Roadmap**: Accordion cards featuring priority level, curated learning resources, practical project ideas, and estimated timelines.
- **Multi-Format Export**: One-click download buttons for **Markdown (.md)**, **JSON (.json)**, and professional **Vector PDF (.pdf)** reports.
- **Add to Cohort History**: Append current assessment directly into cohort memory.

---

### Tab 2: ⚖️ Multi-Career Comparison
- **Career Selector**: Select 2 to 7 career domains for side-by-side comparative analysis.
- **Comparative Metrics Chart**: Grouped bar chart comparing Overall Match Score, Skill Alignment Score, and ML Probability across chosen careers.
- **Side-by-Side Breakdown Cards**: Individual career summary cards showing match %, alignment %, and matching vs missing skill lists.
- **Summary Comparison Table**: Full dataframe table for tabular review.

---

### Tab 3: 📈 Cohort Analytics & Insights
- **Aggregate KPIs**: Total assessed candidates, average experience in years, average model confidence %, average competency match score %.
- **Career Distribution Chart**: Frequency of candidate classifications across domains.
- **Confidence Score Distribution**: Histogram/bins (`< 50%`, `50-70%`, `70-90%`, `90-100%`).
- **Skill Frequency Analysis**: Bar chart of the top technical skills present in the candidate cohort.
- **Missing Skill Bottlenecks**: Highlights the most frequently lacking skills across the candidate pool.
- **Cohort Records Table**: Interactive dataframe of all assessed candidates with search and sort.
- **CSV Export**: One-click download of the complete cohort summary table as `.csv`.
- **Dataset Controls**: "Reset to Default Sample Cohort" and "Clear Cohort History" buttons for testing empty states.
