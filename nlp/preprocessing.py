"""
Text Preprocessing Module for Career Prediction System.
Handles text cleaning, token normalization, and feature combination.
"""

import re
import string


def clean_text(text: str) -> str:
    """
    Cleans raw input text by removing URLs, non-printable characters,
    excessive whitespace, and standardizing casing.
    """
    if not isinstance(text, str):
        return ""

    # Replace newlines and tabs with spaces
    text = re.sub(r"[\r\n\t]+", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Remove email addresses from feature text if present
    text = re.sub(r"\S+@\S+", " ", text)

    # Normalize special characters (keep punctuation relevant to skills like C++, C#, .NET)
    # Replace weird unicode bullets and symbols with space
    text = re.sub(r"[•●▪■◆★\u2022\u2023\u25E6\u2043\u2219]", " ", text)

    # Replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_text(text: str) -> str:
    """
    Converts text to lowercase and strips trailing whitespace.
    """
    return clean_text(text).lower()


def combine_profile_features(profile: dict) -> str:
    """
    Combines structured profile fields into a single unified text string
    used for TF-IDF feature extraction.

    Expected profile keys:
    - education: str or list
    - skills: str or list
    - experience: int or float or str (years)
    - certifications: str or list
    - projects: str or list
    """
    if not isinstance(profile, dict):
        return ""

    def _format_field(val):
        if isinstance(val, list):
            return ", ".join(str(item).strip() for item in val if str(item).strip())
        return str(val).strip() if val is not None else ""

    education = _format_field(profile.get("education", ""))
    skills = _format_field(profile.get("skills", ""))
    experience = profile.get("experience", 0)
    certifications = _format_field(profile.get("certifications", ""))
    projects = _format_field(profile.get("projects", ""))

    # Format unified representation
    combined = (
        f"Education: {education} | "
        f"Skills: {skills} | "
        f"Experience: {experience} years | "
        f"Certifications: {certifications} | "
        f"Projects: {projects}"
    )

    return combined
