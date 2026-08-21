"""
Sentence-BERT Skill Embeddings Module.
Generates semantic embeddings for user skills and career profiles,
and computes semantic cosine similarity.
Milestone 2: Advanced ML & Recommendation Engine.
"""

import os
import sys
from typing import List, Union, Dict, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Central config
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from ml.config import SBERT_MODEL_NAME


# Career Knowledge Base with detailed descriptions and canonical skill requirements
CAREER_DESCRIPTIONS = {
    "Data Scientist": (
        "Statistical analysis, machine learning algorithms, deep learning neural networks, predictive modeling, "
        "data cleaning with pandas and numpy, feature engineering, hypothesis testing, SQL database querying, "
        "data visualization with matplotlib, seaborn and tableau, big data analytics, and Python programming."
    ),
    "Data Analyst": (
        "Business intelligence, data visualization dashboards, exploratory data analysis, advanced Excel spreadsheets, "
        "Power BI reports, Tableau dashboards, SQL relational database queries, metric reporting, KPI tracking, "
        "and data wrangling with pandas."
    ),
    "Software Developer": (
        "Object-oriented programming, data structures, algorithms, system design, Java and C++ application development, "
        "RESTful API services, backend architecture, unit testing, Git version control, multithreading, and debugging."
    ),
    "Web Developer": (
        "Frontend user interfaces with HTML5, CSS3, modern JavaScript, TypeScript, React components, state management, "
        "responsive design, CSS frameworks like Bootstrap and Tailwind, backend web servers with Node.js, Express, Flask, "
        "and RESTful web service integration."
    ),
    "Machine Learning Engineer": (
        "Deep learning architectures, computer vision, natural language processing (NLP), PyTorch, TensorFlow, "
        "MLOps pipelines, containerization with Docker, Kubernetes model deployment, low latency inference, "
        "and model optimization."
    ),
    "Cloud Engineer": (
        "Cloud architecture on AWS, Microsoft Azure, and GCP, infrastructure as code with Terraform, Docker containers, "
        "Kubernetes cluster orchestration, CI/CD automation pipelines with Jenkins and GitLab, Linux server administration, "
        "and cloud network security."
    ),
    "Cybersecurity Analyst": (
        "Network security protocols, penetration testing, vulnerability assessment, ethical hacking, Wireshark packet capture, "
        "Nmap scanning, Metasploit exploitation, firewall configurations, SIEM log monitoring, SOC operations, "
        "and incident response."
    )
}

# Singleton holders for Sentence-BERT model
_sbert_model = None
_sbert_available = None
_fallback_vectorizer = None


def get_sbert_model():
    """
    Loads SentenceTransformer model ('all-MiniLM-L6-v2') if available locally,
    or falls back seamlessly to the semantic TF-IDF cosine vectorizer.
    """
    global _sbert_model, _sbert_available

    if _sbert_available is False:
        return None

    if _sbert_model is not None:
        return _sbert_model

    # Check if a fine-tuned or downloaded model exists in local model dir
    local_model_path = os.path.join(parent_dir, "models", "sentence_transformer")
    model_to_load = local_model_path if os.path.exists(os.path.join(local_model_path, "modules.json")) else SBERT_MODEL_NAME

    try:
        from sentence_transformers import SentenceTransformer
        # Use local_files_only if available locally or try loading
        if os.path.exists(os.path.join(local_model_path, "modules.json")):
            _sbert_model = SentenceTransformer(local_model_path)
            _sbert_available = True
            return _sbert_model
        else:
            # Check environment flag or attempt quick load
            if os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1":
                _sbert_available = False
                return None

            import socket
            # Quick check if huggingface.co is reachable in 1 second
            try:
                socket.create_connection(("huggingface.co", 443), timeout=1.0)
                _sbert_model = SentenceTransformer(SBERT_MODEL_NAME)
                _sbert_available = True
                return _sbert_model
            except Exception:
                _sbert_available = False
                return None
    except Exception as e:
        _sbert_available = False
        return None


def get_fallback_vectorizer():
    """
    Initializes TF-IDF vectorizer over the career descriptions corpus as fallback.
    """
    global _fallback_vectorizer
    if _fallback_vectorizer is None:
        _fallback_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        _fallback_vectorizer.fit(list(CAREER_DESCRIPTIONS.values()))
    return _fallback_vectorizer


def encode_skills(skills: Union[str, List[str]]) -> np.ndarray:
    """
    Encodes user skills into a dense semantic embedding vector.
    Handles strings, lists, empty inputs, and synonyms.
    """
    if isinstance(skills, list):
        skill_text = ", ".join(str(s).strip() for s in skills if str(s).strip())
    else:
        skill_text = str(skills or "").strip()

    if not skill_text:
        # Return zero vector if no skills provided
        model = get_sbert_model()
        dim = 384 if model is not None else 100
        return np.zeros((1, dim))

    model = get_sbert_model()
    if model is not None:
        embedding = model.encode([skill_text], show_progress_bar=False, normalize_embeddings=True)
        return np.array(embedding)
    else:
        # Fallback to TF-IDF vectorization
        vec = get_fallback_vectorizer()
        emb = vec.transform([skill_text]).toarray()
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb


def encode_job_description(description: str) -> np.ndarray:
    """
    Encodes a career or job description into a semantic embedding vector.
    """
    desc_text = str(description or "").strip()
    if not desc_text:
        model = get_sbert_model()
        dim = 384 if model is not None else 100
        return np.zeros((1, dim))

    model = get_sbert_model()
    if model is not None:
        embedding = model.encode([desc_text], show_progress_bar=False, normalize_embeddings=True)
        return np.array(embedding)
    else:
        vec = get_fallback_vectorizer()
        emb = vec.transform([desc_text]).toarray()
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb


def calculate_skill_similarity(user_embedding: np.ndarray, career_embedding: np.ndarray) -> float:
    """
    Calculates cosine similarity between user skill embedding and career embedding.
    Returns float score in range [0.0, 1.0].
    """
    if user_embedding.shape[1] != career_embedding.shape[1]:
        # Dimension mismatch guard
        return 0.0

    # Ensure 2D
    if len(user_embedding.shape) == 1:
        user_embedding = user_embedding.reshape(1, -1)
    if len(career_embedding.shape) == 1:
        career_embedding = career_embedding.reshape(1, -1)

    sim = float(cosine_similarity(user_embedding, career_embedding)[0][0])
    # Bound between 0.0 and 1.0
    return float(np.clip(sim, 0.0, 1.0))


def get_career_semantic_similarity(user_skills: Union[str, List[str]], career_name: str) -> float:
    """
    Calculates semantic similarity between user skills and a specific career profile.
    """
    career_desc = CAREER_DESCRIPTIONS.get(career_name, career_name)
    user_emb = encode_skills(user_skills)
    career_emb = encode_job_description(career_desc)
    return calculate_skill_similarity(user_emb, career_emb)


def get_all_career_similarities(user_skills: Union[str, List[str]]) -> Dict[str, float]:
    """
    Computes semantic similarity for all known career categories.
    """
    user_emb = encode_skills(user_skills)
    similarities = {}
    for career, desc in CAREER_DESCRIPTIONS.items():
        career_emb = encode_job_description(desc)
        sim = calculate_skill_similarity(user_emb, career_emb)
        similarities[career] = round(sim, 4)
    return similarities


if __name__ == "__main__":
    test_skills = ["Python", "Machine Learning", "SQL", "Pandas"]
    print(f"Testing Semantic Similarities for skills: {test_skills}")
    sims = get_all_career_similarities(test_skills)
    for c, s in sorted(sims.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {c:<28}: {s*100:.1f}%")
