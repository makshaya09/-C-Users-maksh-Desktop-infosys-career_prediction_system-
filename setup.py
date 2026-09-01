"""
Setup configuration script for CareerCast Python library.
"""

from setuptools import setup, find_packages

setup(
    name="careercast",
    version="1.0.0",
    description="AI-Based Career Prediction, Recommendation & Skill Gap Analysis System",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "careercast=careercast.cli.main:main",
        ],
    },
)
