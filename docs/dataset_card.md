# Dataset Card — CareerCast

---

## 1. Dataset Summary

The CareerCast dataset collection comprises synthetic and benchmark candidate profile records curated to train, validate, and benchmark AI models for career domain classification, semantic skill matching, and transition recommendations.

---

## 2. Dataset Purpose

The primary purpose of the dataset is to enable supervised classification of technical candidate profiles into standard career domains, calculate semantic skill similarity, identify competency gaps, and benchmark recommendation accuracy against historical transition patterns.

---

## 3. Data Sources

1. **`data/career_dataset.csv`**:
   - Primary multi-class training and evaluation dataset (840 structured records, 120 balanced samples across each of the 7 canonical domains).
   - Generated reproducibly via `data/generate_dataset.py` using domain-specific skill distributions, degree requirements, experience distributions, and project templates.
2. **`data/linkedin_transitions_sample.csv`**:
   - Curated career transition dataset tracking historical career movements (source career $\rightarrow$ target career) with associated skills.
3. **`data/semeval_career_benchmark_sample.csv`**:
   - Standard career benchmark dataset used for evaluating Top-K ranking, Mean Reciprocal Rank (MRR), and Normalized Discounted Cumulative Gain (NDCG@5).

---

## 4. Features & Attributes

| Feature Name | Data Type | Description | Example |
|---|---|---|---|
| `Education` | String | Academic degree qualification | `B.Tech`, `M.Tech in AI`, `BCA`, `MCA`, `B.Sc CS` |
| `Skills` | String | Comma-separated list of technical competencies | `Python, Machine Learning, SQL, Pandas, Scikit-learn` |
| `Experience` | Numeric (Float) | Number of years of relevant work experience | `2.5`, `3.0`, `1.0` |
| `Certifications` | String | Professional certifications earned | `AWS Certified Solutions Architect`, `None` |
| `Projects` | String | Key academic or industry projects | `Customer Churn Prediction System` |
| `Career` *(Target)* | Categorical | Target career domain label | `Data Scientist`, `Cloud Engineer`, etc. |

---

## 5. Target Labels & Domain Classes

The dataset covers 7 balanced canonical career domains:

1. **Data Scientist**: Machine learning, statistical modeling, exploratory data analysis, deep learning, Python, SQL.
2. **Data Analyst**: Business intelligence, SQL queries, spreadsheet modeling, Tableau/Power BI, descriptive statistics.
3. **Software Developer**: Core Java/C++, data structures, algorithms, object-oriented design, REST APIs, Git.
4. **Web Developer**: Frontend/backend web engineering, HTML, CSS, JavaScript, React, Node.js, Express, MongoDB.
5. **Machine Learning Engineer**: Scaled ML pipelines, PyTorch, TensorFlow, MLOps, deep learning, Docker, model serving.
6. **Cloud Engineer**: Cloud infrastructure, AWS/Azure, Docker, Kubernetes, Linux, Terraform, CI/CD automation.
7. **Cybersecurity Analyst**: Network defense, vulnerability assessments, Wireshark, ethical hacking, SIEM, firewalls, Linux.

---

## 6. Preprocessing & Feature Engineering

1. **Text Normalization**: Lowercasing, punctuation stripping, whitespace normalization, and removal of special characters.
2. **Feature Concatenation**: Candidate attributes are combined into a serialized text representation matching the format:
   ```
   Education: {edu} | Skills: {skills} | Experience: {exp} years | Certifications: {cert} | Projects: {proj}
   ```
3. **Vectorization**: TF-IDF (Term Frequency-Inverse Document Frequency) unigram and bigram tokenization with sublinear term-frequency scaling.
4. **Label Encoding**: Categorical string labels mapped to integers ($0 \dots 6$) via Scikit-Learn `LabelEncoder`.

---

## 7. Limitations

- **Domain Scope**: The dataset focuses primarily on Information Technology and Computer Science disciplines (7 target domains). Non-software engineering disciplines (e.g., Civil, Mechanical, Healthcare) are not currently represented.
- **Synthesized Data**: The primary dataset was synthetically generated using realistic probabilistic distributions; edge cases in non-standard resumes may require domain fine-tuning.

---

## 8. Bias and Ethical Considerations

- **Demographic Neutrality**: The dataset contains no personal demographic identifiers (e.g., gender, race, ethnicity, age, religion, or location) to prevent demographic bias in classification.
- **Merit-Based Features**: Predictions are strictly driven by technical skills, educational degrees, experience duration, projects, and certifications.
- **Advisory Role**: The dataset and models are intended solely as assistive guidance tools and should not be used as the sole determinant for employment or hiring decisions.

---

## 9. Intended Use

- Academic research, career counseling prototypes, and developer education.
- Automated resume skill parsing and technical competency gap analysis.
- Evaluating multi-modal recommendation systems blending ML classification with semantic embeddings.

---

## 10. Out-of-Scope Use

- Automated high-stakes employment rejection or autonomous candidate screening without human review.
- Assessing non-technical job roles outside the 7 trained IT domains without retraining.
