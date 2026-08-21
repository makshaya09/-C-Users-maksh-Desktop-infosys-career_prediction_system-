"""
Resume Parsing Module using SpaCy and PyPDF2.
Extracts Skills, Education, Roles, Experience, Certifications, and Projects from PDF and TXT resumes.
"""

import io
import os
import re
from typing import Dict, List, Any, Union
import spacy
from spacy.pipeline import EntityRuler
import PyPDF2

from .preprocessing import clean_text


# Predefined knowledge bases for skills, education, and roles
KNOWN_SKILLS = [
    # Programming Languages
    "python", "java", "c++", "c#", "c", "javascript", "typescript", "php", "ruby", "go", "rust", "scala", "kotlin", "swift", "r", "matlab", "bash", "shell",
    # Web & Frameworks
    "html", "html5", "css", "css3", "bootstrap", "tailwind", "react", "react.js", "angular", "vue.js", "vue", "node.js", "express", "express.js",
    "flask", "django", "fastapi", "spring", "spring boot", "asp.net", "laravel", "jquery", "rest api", "graphql",
    # Data Science & Machine Learning
    "machine learning", "deep learning", "nlp", "natural language processing", "computer vision",
    "pandas", "numpy", "scikit-learn", "sklearn", "tensorflow", "keras", "pytorch", "matplotlib", "seaborn",
    "data analysis", "data visualization", "statistics", "tableau", "power bi", "excel", "big data", "hadoop", "spark",
    # Cloud & DevOps
    "aws", "amazon web services", "azure", "microsoft azure", "gcp", "google cloud", "google cloud platform",
    "docker", "kubernetes", "jenkins", "ci/cd", "terraform", "ansible", "linux", "git", "github", "gitlab",
    # Databases
    "sql", "mysql", "postgresql", "sqlite", "mongodb", "redis", "oracle", "nosql", "dynamodb",
    # Cybersecurity & Networking
    "cybersecurity", "cyber security", "network security", "penetration testing", "ethical hacking",
    "wireshark", "nmap", "metasploit", "cryptography", "firewall", "siem", "soc", "vulnerability assessment"
]

KNOWN_EDUCATION = [
    "b.tech", "btech", "b.e", "be", "m.tech", "mtech", "m.e", "me",
    "bca", "mca", "b.sc", "bsc", "m.sc", "msc", "b.s", "bs", "m.s", "ms",
    "b.com", "bcom", "m.com", "mcom", "bba", "mba", "ph.d", "phd", "diploma",
    "bachelor of technology", "master of technology", "bachelor of engineering",
    "master of engineering", "bachelor of science", "master of science",
    "bachelor of computer applications", "master of computer applications",
    "bachelor of business administration", "master of business administration"
]

KNOWN_ROLES = [
    "software developer", "software engineer", "junior software developer", "senior software engineer",
    "web developer", "frontend developer", "backend developer", "full stack developer", "fullstack engineer",
    "data scientist", "junior data scientist", "senior data scientist",
    "data analyst", "junior data analyst", "business analyst",
    "machine learning engineer", "ml engineer", "ai engineer", "deep learning engineer",
    "cloud engineer", "cloud architect", "devops engineer", "site reliability engineer",
    "cybersecurity analyst", "security analyst", "information security analyst", "soc analyst",
    "penetration tester", "security engineer", "database administrator", "qa engineer", "system analyst"
]

KNOWN_CERTIFICATIONS = [
    "aws certified", "aws certified solutions architect", "aws certified developer", "aws certified cloud practitioner",
    "azure fundamentals", "azure solutions architect", "google associate cloud engineer", "gcp certified",
    "comptia security+", "comptia network+", "comptia a+", "ceh", "certified ethical hacker", "cissp", "ccna",
    "google data analytics", "ibm data science", "tensorflow developer certificate", "pmp", "scrum master"
]


# Singleton SpaCy NLP model loader
_nlp = None


def get_nlp_model():
    """
    Loads or initializes the SpaCy NLP pipeline with EntityRuler for custom domain entities.
    """
    global _nlp
    if _nlp is not None:
        return _nlp

    try:
        _nlp = spacy.load("en_core_web_sm")
    except Exception:
        # Fallback to blank model if model package is not yet installed
        _nlp = spacy.blank("en")

    # Add entity ruler if not present
    if "entity_ruler" not in _nlp.pipe_names:
        ruler = _nlp.add_pipe("entity_ruler", before="ner" if "ner" in _nlp.pipe_names else None)

        patterns = []
        for skill in KNOWN_SKILLS:
            patterns.append({"label": "SKILL", "pattern": skill})
            patterns.append({"label": "SKILL", "pattern": skill.title()})
            patterns.append({"label": "SKILL", "pattern": skill.upper()})

        for edu in KNOWN_EDUCATION:
            patterns.append({"label": "EDUCATION", "pattern": edu})
            patterns.append({"label": "EDUCATION", "pattern": edu.title()})
            patterns.append({"label": "EDUCATION", "pattern": edu.upper()})

        for role in KNOWN_ROLES:
            patterns.append({"label": "ROLE", "pattern": role})
            patterns.append({"label": "ROLE", "pattern": role.title()})

        for cert in KNOWN_CERTIFICATIONS:
            patterns.append({"label": "CERTIFICATION", "pattern": cert})
            patterns.append({"label": "CERTIFICATION", "pattern": cert.title()})

        ruler.add_patterns(patterns)

    return _nlp


def extract_text_from_pdf(file_input: Union[str, bytes, io.BytesIO]) -> str:
    """
    Extracts plain text from a PDF file path, bytes, or BytesIO stream using PyPDF2.
    """
    text = ""
    try:
        if isinstance(file_input, str):
            with open(file_input, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        elif isinstance(file_input, bytes):
            stream = io.BytesIO(file_input)
            reader = PyPDF2.PdfReader(stream)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        elif hasattr(file_input, "read"):
            reader = PyPDF2.PdfReader(file_input)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        raise ValueError(f"Failed to extract text from PDF: {str(e)}")

    return text.strip()


def extract_text_from_txt(file_input: Union[str, bytes, io.BytesIO]) -> str:
    """
    Extracts text from a TXT file path, bytes, or BytesIO stream.
    """
    try:
        if isinstance(file_input, str):
            with open(file_input, "r", encoding="utf-8", errors="ignore") as f:
                return f.read().strip()
        elif isinstance(file_input, bytes):
            return file_input.decode("utf-8", errors="ignore").strip()
        elif hasattr(file_input, "read"):
            content = file_input.read()
            if isinstance(content, bytes):
                return content.decode("utf-8", errors="ignore").strip()
            return str(content).strip()
    except Exception as e:
        raise ValueError(f"Failed to read TXT file: {str(e)}")

    return ""


def extract_text(file_input: Union[str, bytes, io.BytesIO], filename: str = "") -> str:
    """
    Extracts text from an uploaded resume file (supports PDF and TXT).
    """
    if isinstance(file_input, str):
        ext = os.path.splitext(file_input)[1].lower()
    else:
        ext = os.path.splitext(filename)[1].lower() if filename else ""

    if ext == ".pdf":
        return extract_text_from_pdf(file_input)
    elif ext in [".txt", ".text"]:
        return extract_text_from_txt(file_input)
    else:
        # Try PDF first, then fallback to TXT
        try:
            return extract_text_from_pdf(file_input)
        except Exception:
            if hasattr(file_input, "seek"):
                file_input.seek(0)
            return extract_text_from_txt(file_input)


def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Processes resume text with SpaCy NER and EntityRuler to extract entities.
    """
    nlp = get_nlp_model()
    doc = nlp(text)

    entities: Dict[str, List[str]] = {
        "SKILL": [],
        "EDUCATION": [],
        "ROLE": [],
        "CERTIFICATION": []
    }

    # Extract from doc.ents
    for ent in doc.ents:
        label = ent.label_
        val = ent.text.strip()
        if label in entities:
            # Normalize display capitalization
            val_clean = val.title() if not val.isupper() else val
            if val_clean.lower() not in [item.lower() for item in entities[label]]:
                entities[label].append(val_clean)

    # Fallback regex / keyword search for any high-priority skills missed by tokenization
    text_lower = text.lower()
    for skill in KNOWN_SKILLS:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower, re.IGNORECASE):
            disp = skill.upper() if len(skill) <= 3 and skill not in ["git", "vue", "php", "aws", "gcp"] else skill.title()
            if skill == "c++":
                disp = "C++"
            elif skill == "c#":
                disp = "C#"
            elif skill == "aws":
                disp = "AWS"
            elif skill == "gcp":
                disp = "GCP"
            elif skill == "sql":
                disp = "SQL"
            elif skill == "html" or skill == "html5":
                disp = "HTML5"
            elif skill == "css" or skill == "css3":
                disp = "CSS3"

            if disp.lower() not in [s.lower() for s in entities["SKILL"]]:
                entities["SKILL"].append(disp)

    for edu in KNOWN_EDUCATION:
        pattern = r"\b" + re.escape(edu) + r"\b"
        if re.search(pattern, text_lower, re.IGNORECASE):
            disp = edu.upper() if len(edu) <= 5 else edu.title()
            if disp.lower() not in [e.lower() for e in entities["EDUCATION"]]:
                entities["EDUCATION"].append(disp)

    for role in KNOWN_ROLES:
        pattern = r"\b" + re.escape(role) + r"\b"
        if re.search(pattern, text_lower, re.IGNORECASE):
            disp = role.title()
            if disp.lower() not in [r.lower() for r in entities["ROLE"]]:
                entities["ROLE"].append(disp)

    return entities


def extract_skills(text_or_entities: Union[str, Dict]) -> List[str]:
    """Extracts unique list of skills."""
    if isinstance(text_or_entities, dict):
        return text_or_entities.get("SKILL", [])
    return extract_entities(str(text_or_entities)).get("SKILL", [])


def extract_education(text_or_entities: Union[str, Dict]) -> List[str]:
    """Extracts unique list of education degrees."""
    if isinstance(text_or_entities, dict):
        return text_or_entities.get("EDUCATION", [])
    return extract_entities(str(text_or_entities)).get("EDUCATION", [])


def extract_roles(text_or_entities: Union[str, Dict]) -> List[str]:
    """Extracts unique list of previous roles."""
    if isinstance(text_or_entities, dict):
        return text_or_entities.get("ROLE", [])
    return extract_entities(str(text_or_entities)).get("ROLE", [])


def extract_experience_years(text: str) -> float:
    """
    Extracts numeric years of experience using regular expression heuristics.
    e.g., '3 years of experience', '5+ yrs exp', '2.5 years'
    """
    patterns = [
        r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)(?:\s+of)?\s*(?:experience|exp)?",
        r"(?:experience|exp)\s*:\s*(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                years = float(match.group(1))
                if 0 <= years <= 50:
                    return years
            except ValueError:
                pass
    return 0.0


def extract_projects(text: str) -> List[str]:
    """
    Extracts candidate project mentions or section bullet items.
    """
    projects = []
    # Search for project section headers
    match = re.search(r"(?:projects|academic projects|key projects)[:\n](.*?)(?:\n\n|\n[A-Z]{3,}|\Z)", text, re.IGNORECASE | re.DOTALL)
    if match:
        section_text = match.group(1)
        lines = [line.strip("•-* \t\r") for line in section_text.split("\n") if len(line.strip("•-* \t\r")) > 5]
        projects = lines[:3]
    return projects


def parse_resume(file_input: Union[str, bytes, io.BytesIO], filename: str = "") -> Dict[str, Any]:
    """
    Complete resume parsing pipeline:
    1. Extract raw text from file
    2. Clean text
    3. Extract skills, education, roles, experience, certifications, and projects
    4. Return structured dictionary
    """
    raw_text = extract_text(file_input, filename=filename)
    if not raw_text or not raw_text.strip():
        raise ValueError("The uploaded resume is empty or no text could be extracted.")

    cleaned = clean_text(raw_text)
    entities = extract_entities(cleaned)
    experience = extract_experience_years(cleaned)
    projects = extract_projects(cleaned)

    # Pick top degree as primary education if found
    primary_education = entities.get("EDUCATION", ["B.Tech"])[0] if entities.get("EDUCATION") else "B.Tech"

    result = {
        "skills": entities.get("SKILL", []),
        "education": entities.get("EDUCATION", []),
        "primary_education": primary_education,
        "roles": entities.get("ROLE", []),
        "certifications": entities.get("CERTIFICATION", []),
        "experience": experience,
        "projects": projects,
        "raw_text_preview": cleaned[:300] + ("..." if len(cleaned) > 300 else "")
    }

    return result
