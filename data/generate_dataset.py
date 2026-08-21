"""
Dataset Generator for Career Prediction System.
Generates a representative, labeled synthetic/sample career dataset (700+ records)
for training the baseline Logistic Regression model across 7 career categories.
"""

import os
import random
import pandas as pd

random.seed(42)

CAREERS = [
    "Data Scientist",
    "Data Analyst",
    "Software Developer",
    "Web Developer",
    "Machine Learning Engineer",
    "Cloud Engineer",
    "Cybersecurity Analyst"
]

EDUCATIONS = ["B.Tech", "B.E", "M.Tech", "MCA", "B.Sc", "M.Sc", "BCA", "M.S"]

CAREER_PROFILES = {
    "Data Scientist": {
        "core_skills": ["Python", "Machine Learning", "Pandas", "NumPy", "Scikit-learn", "SQL", "Statistics"],
        "optional_skills": ["Deep Learning", "TensorFlow", "PyTorch", "NLP", "Matplotlib", "Seaborn", "R", "Tableau", "Big Data", "Spark"],
        "certifications": ["IBM Data Science Professional", "TensorFlow Developer Certificate", "Google Data Analytics", "None"],
        "projects": [
            "Customer Churn Prediction Model", "Credit Risk Scoring Algorithm", "Sentiment Analysis on Product Reviews",
            "House Price Forecasting Pipeline", "Healthcare Fraud Detection Engine", "Recommendation System for E-Commerce"
        ]
    },
    "Data Analyst": {
        "core_skills": ["SQL", "Excel", "Tableau", "Power BI", "Python", "Data Visualization", "Data Analysis"],
        "optional_skills": ["Pandas", "NumPy", "Statistics", "MySQL", "PostgreSQL", "Google Analytics", "R", "Business Intelligence"],
        "certifications": ["Google Data Analytics Certificate", "Microsoft Certified Data Analyst", "Tableau Desktop Specialist", "None"],
        "projects": [
            "Sales Performance Interactive Dashboard", "Supply Chain Inventory Analysis", "Customer Segmentation and Cohort Study",
            "Financial Quarterly KPI Visualization", "Marketing Campaign ROI Tracker", "HR Employee Attrition Insights"
        ]
    },
    "Software Developer": {
        "core_skills": ["Java", "C++", "Python", "Data Structures", "Algorithms", "Git", "OOP"],
        "optional_skills": ["C#", ".NET", "Spring Boot", "SQL", "MySQL", "REST API", "Linux", "Design Patterns", "Docker", "Unit Testing"],
        "certifications": ["Oracle Certified Java Professional", "AWS Certified Developer", "Microsoft Certified C# Developer", "None"],
        "projects": [
            "Hospital Management Desktop App", "Banking Transaction Core System", "Inventory and Billing Management Software",
            "Multi-threaded File Downloader", "Peer-to-Peer Chat Engine", "Distributed Cache Mechanism"
        ]
    },
    "Web Developer": {
        "core_skills": ["HTML", "CSS", "JavaScript", "React", "Node.js", "Express", "REST API"],
        "optional_skills": ["Bootstrap", "Tailwind", "TypeScript", "Vue.js", "Angular", "MongoDB", "PostgreSQL", "Next.js", "Flask", "Django"],
        "certifications": ["Meta Front-End Developer Certificate", "Full Stack Web Development Bootcamp", "JavaScript Specialist", "None"],
        "projects": [
            "E-Commerce Web Portal with Stripe", "Social Media Feed Web Application", "Real-time Collaborative Whiteboard",
            "Blog CMS with Markdown Support", "Restaurant Food Ordering Web App", "Portfolio Showcase with Dynamic Theming"
        ]
    },
    "Machine Learning Engineer": {
        "core_skills": ["Python", "TensorFlow", "PyTorch", "Deep Learning", "Machine Learning", "Docker", "Scikit-learn"],
        "optional_skills": ["Computer Vision", "NLP", "MLOps", "Kubernetes", "AWS", "FastAPI", "Pandas", "CUDA", "Model Deployment"],
        "certifications": ["AWS Certified Machine Learning Specialty", "DeepLearning.AI TensorFlow Developer", "GCP Professional ML Engineer", "None"],
        "projects": [
            "Real-time Object Detection with YOLO", "Automated Image Segmentation for Medical Scans", "Transformer-based Chatbot Engine",
            "Edge AI Device Defect Classifier", "End-to-End MLOps Pipeline on Kubernetes", "Autonomous Driving Lane Detection"
        ]
    },
    "Cloud Engineer": {
        "core_skills": ["AWS", "Docker", "Kubernetes", "Linux", "Terraform", "CI/CD", "Cloud Architecture"],
        "optional_skills": ["Azure", "GCP", "Ansible", "Jenkins", "Bash", "Python", "Networking", "GitLab CI", "Microservices", "Prometheus"],
        "certifications": ["AWS Solutions Architect Associate", "Google Associate Cloud Engineer", "Certified Kubernetes Administrator (CKA)", "Microsoft Azure Fundamentals"],
        "projects": [
            "Multi-Region High Availability AWS Infrastructure", "Automated CI/CD Pipeline with Jenkins and Docker", "Kubernetes Microservices Cluster Deployment",
            "Serverless Event-Driven ETL with AWS Lambda", "Cloud Cost Optimization & Monitoring Setup", "Disaster Recovery Infrastructure as Code"
        ]
    },
    "Cybersecurity Analyst": {
        "core_skills": ["Network Security", "Wireshark", "Linux", "Ethical Hacking", "Firewall", "Cybersecurity", "Nmap"],
        "optional_skills": ["Metasploit", "SIEM", "SOC", "Vulnerability Assessment", "Cryptography", "Python", "Bash", "Penetration Testing", "Burp Suite"],
        "certifications": ["CompTIA Security+", "Certified Ethical Hacker (CEH)", "Cisco CCNA Security", "CompTIA CySA+", "None"],
        "projects": [
            "Enterprise Network Vulnerability Assessment", "Intrusion Detection System with Snort", "Automated Log Analysis SIEM Dashboard",
            "Web Application Penetration Testing Report", "Phishing Simulation & Awareness Program", "Malware Reverse Engineering Sandbox"
        ]
    }
}


def generate_dataset(num_samples_per_career: int = 120) -> pd.DataFrame:
    """Generates synthetic career dataset with realistic distributions."""
    records = []

    for career, profile in CAREER_PROFILES.items():
        for _ in range(num_samples_per_career):
            edu = random.choice(EDUCATIONS)
            exp = round(random.uniform(0.0, 10.0), 1)

            # Sample 3-5 core skills + 1-3 optional skills
            num_core = min(len(profile["core_skills"]), random.randint(3, 5))
            num_opt = min(len(profile["optional_skills"]), random.randint(1, 3))

            skills_chosen = random.sample(profile["core_skills"], num_core) + random.sample(profile["optional_skills"], num_opt)
            random.shuffle(skills_chosen)
            skills_str = ", ".join(skills_chosen)

            cert_chosen = random.choice(profile["certifications"])
            proj_chosen = random.choice(profile["projects"])

            records.append({
                "education": edu,
                "skills": skills_str,
                "experience": exp,
                "certifications": cert_chosen,
                "projects": proj_chosen,
                "career": career
            })

    # Shuffle records
    random.shuffle(records)
    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    df = generate_dataset(num_samples_per_career=120)
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(output_dir, "career_dataset.csv")
    df.to_csv(output_file, index=False)
    print(f"Generated {len(df)} samples across {df['career'].nunique()} career categories.")
    print(f"Saved to: {output_file}")
