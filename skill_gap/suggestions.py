"""
Competency Improvement Suggestions Knowledge Base.
Provides curated, actionable learning pathways, official resources,
practical project ideas, priority levels, and estimated timeframes for skills.
"""

from typing import Dict, Any, List


# Structured skill suggestion database
SKILL_SUGGESTIONS_DB: Dict[str, Dict[str, Any]] = {
    # Core Languages & Fundamentals
    "Python": {
        "priority": "High",
        "category": "Core Language",
        "recommended_action": "Master Python fundamentals, OOP principles, data structures, and standard libraries (collections, itertools).",
        "learning_resources": [
            "Official Python Documentation (docs.python.org)",
            "Real Python Interactive Tutorials (realpython.com)",
            "Automate the Boring Stuff with Python"
        ],
        "practical_project": "Build an asynchronous CLI data processing pipeline with logging, unit tests, and typing.",
        "estimated_time": "3 - 4 weeks"
    },
    "Java": {
        "priority": "High",
        "category": "Core Language",
        "recommended_action": "Study Core Java, JVM architecture, multithreading, concurrency, and Spring Boot framework.",
        "learning_resources": [
            "Oracle Java Documentation & Tutorials",
            "Baeldung Spring & Java Guides",
            "Effective Java by Joshua Bloch"
        ],
        "practical_project": "Develop a microservices backend with Spring Boot, JPA/Hibernate, and Maven.",
        "estimated_time": "4 - 6 weeks"
    },
    "C++": {
        "priority": "High",
        "category": "Core Language",
        "recommended_action": "Master modern C++ (C++17/20), memory management (pointers/RAII), STL containers, and template metaprogramming.",
        "learning_resources": [
            "cppreference.com Reference",
            "learncpp.com In-Depth Tutorials",
            "Effective Modern C++ by Scott Meyers"
        ],
        "practical_project": "Implement a high-performance custom memory allocator or graph processing engine in C++.",
        "estimated_time": "5 - 6 weeks"
    },
    "JavaScript": {
        "priority": "High",
        "category": "Frontend / Fullstack",
        "recommended_action": "Learn modern ES6+ features, asynchronous programming (Promises, async/await), DOM manipulation, and Event Loop.",
        "learning_resources": [
            "MDN Web Docs: JavaScript Guide",
            "javascript.info Modern JavaScript Tutorial",
            "You Don't Know JS Book Series"
        ],
        "practical_project": "Build an interactive single-page application with modular state management and WebSockets.",
        "estimated_time": "3 - 4 weeks"
    },
    "TypeScript": {
        "priority": "Medium",
        "category": "Frontend / Fullstack",
        "recommended_action": "Understand static typing, generics, interfaces, union types, and TypeScript compiler configuration.",
        "learning_resources": [
            "Official TypeScript Handbook (typescriptlang.org)",
            "Total TypeScript by Matt Pocock",
            "TypeScript Deep Dive"
        ],
        "practical_project": "Refactor a vanilla JavaScript project into a strictly typed TypeScript application.",
        "estimated_time": "2 - 3 weeks"
    },

    # Web & UI Development
    "HTML": {
        "priority": "High",
        "category": "Web Development",
        "recommended_action": "Master semantic HTML5 markup, accessibility standards (ARIA, WCAG), and SEO best practices.",
        "learning_resources": [
            "MDN Web Docs: HTML Elements & Semantics",
            "web.dev Accessibility & Semantic HTML Courses"
        ],
        "practical_project": "Create a fully accessible, semantic multi-page portal scoring 100 on Google Lighthouse.",
        "estimated_time": "1 - 2 weeks"
    },
    "CSS": {
        "priority": "High",
        "category": "Web Development",
        "recommended_action": "Deep dive into CSS Flexbox, Grid layout, CSS custom properties (variables), media queries, and responsive animations.",
        "learning_resources": [
            "MDN CSS Layout Reference",
            "CSS-Tricks Complete Guide to Flexbox and Grid",
            "web.dev Responsive Design Guide"
        ],
        "practical_project": "Design a responsive, dark-mode-enabled dashboard layout from scratch without external frameworks.",
        "estimated_time": "2 - 3 weeks"
    },
    "React": {
        "priority": "High",
        "category": "Frontend Framework",
        "recommended_action": "Master React Hooks (useState, useEffect, useMemo, useCallback), Context API, component lifecycle, and state architecture.",
        "learning_resources": [
            "Official React Docs (react.dev)",
            "Epic React by Kent C. Dodds",
            "FreeCodeCamp React Certification"
        ],
        "practical_project": "Build a collaborative kanban board with drag-and-drop, optimistic UI updates, and local storage persistence.",
        "estimated_time": "3 - 5 weeks"
    },
    "Node.js": {
        "priority": "High",
        "category": "Backend Runtime",
        "recommended_action": "Understand event-driven non-blocking I/O, Node Streams, Buffer, file system operations, and npm package architecture.",
        "learning_resources": [
            "Node.js Official Documentation (nodejs.org)",
            "Node.js Design Patterns (Mario Casciaro)",
            "The Odin Project Node.js Course"
        ],
        "practical_project": "Build a streaming file processing server with rate limiting, cluster mode, and custom middleware.",
        "estimated_time": "3 - 4 weeks"
    },
    "Express": {
        "priority": "Medium",
        "category": "Backend Framework",
        "recommended_action": "Learn RESTful routing, middleware chains, error handling pipelines, authentication (JWT/OAuth), and validation.",
        "learning_resources": [
            "Express.js Guide (expressjs.com)",
            "REST API Design Tutorial on MDN",
            "Mozilla Developer Network Express Web Framework"
        ],
        "practical_project": "Construct a secure REST API with JWT authentication, rate limiting, helmet security headers, and Swagger documentation.",
        "estimated_time": "2 - 3 weeks"
    },
    "Bootstrap": {
        "priority": "Normal",
        "category": "CSS Framework",
        "recommended_action": "Learn Bootstrap 5 grid breakpoints, utility classes, modal/carousel components, and SASS variable overrides.",
        "learning_resources": [
            "Bootstrap Official Documentation (getbootstrap.com)",
            "Bootstrap 5 Responsive Grid Guide"
        ],
        "practical_project": "Recreate an enterprise SaaS landing page with responsive layouts, modal workflows, and custom theme variables.",
        "estimated_time": "1 - 2 weeks"
    },

    # Data Science & Machine Learning
    "Machine Learning": {
        "priority": "High",
        "category": "Data Science / AI",
        "recommended_action": "Study supervised and unsupervised learning algorithms (Linear/Logistic Regression, Decision Trees, SVM, Ensembles, Clustering).",
        "learning_resources": [
            "Scikit-Learn User Guide (scikit-learn.org)",
            "Machine Learning Specialization by Andrew Ng (Coursera)",
            "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (A. Géron)"
        ],
        "practical_project": "Build an end-to-end customer churn prediction pipeline with cross-validation, feature importance, and ROC evaluation.",
        "estimated_time": "4 - 6 weeks"
    },
    "Deep Learning": {
        "priority": "High",
        "category": "Data Science / AI",
        "recommended_action": "Learn neural network backpropagation, activation functions, regularization (Dropout/Batch Normalization), CNNs, RNNs, and Transformers.",
        "learning_resources": [
            "Deep Learning Specialization (deeplearning.ai)",
            "MIT 6.S191: Introduction to Deep Learning",
            "Deep Learning with Python by François Chollet"
        ],
        "practical_project": "Train a convolutional neural network for image classification with data augmentation and transfer learning.",
        "estimated_time": "5 - 7 weeks"
    },
    "TensorFlow": {
        "priority": "High",
        "category": "Deep Learning Framework",
        "recommended_action": "Master TensorFlow 2.x, Keras functional API, custom layers, tf.data pipelines, and model export formats (SavedModel/TFLite).",
        "learning_resources": [
            "TensorFlow Core Documentation (tensorflow.org)",
            "DeepLearning.AI TensorFlow Developer Certificate Course"
        ],
        "practical_project": "Deploy an image segmentation or text classification model using TensorFlow Serving or TFLite.",
        "estimated_time": "3 - 4 weeks"
    },
    "PyTorch": {
        "priority": "High",
        "category": "Deep Learning Framework",
        "recommended_action": "Understand dynamic computation graphs, autograd engine, custom nn.Module architectures, Dataset/DataLoader pipelines, and GPU training.",
        "learning_resources": [
            "Official PyTorch Tutorials (pytorch.org/tutorials)",
            "Deep Learning with PyTorch (Stevens et al.)",
            "Fast.ai Practical Deep Learning for Coders"
        ],
        "practical_project": "Implement a custom Transformer encoder or generative model in PyTorch and train on GPU.",
        "estimated_time": "4 - 5 weeks"
    },
    "Pandas": {
        "priority": "High",
        "category": "Data Manipulation",
        "recommended_action": "Master DataFrame operations, indexing (.loc/.iloc), groupby aggregations, merging/joining, reshaping, and handling time-series data.",
        "learning_resources": [
            "Pandas Documentation & User Guide (pandas.pydata.org)",
            "Python for Data Analysis by Wes McKinney"
        ],
        "practical_project": "Perform comprehensive exploratory data analysis (EDA) and data cleansing on a messy real-world multi-table dataset.",
        "estimated_time": "2 - 3 weeks"
    },
    "NumPy": {
        "priority": "Medium",
        "category": "Scientific Computing",
        "recommended_action": "Master N-dimensional array manipulation, vectorization, broadcasting rules, linear algebra routines (linalg), and matrix operations.",
        "learning_resources": [
            "NumPy User Guide & Tutorials (numpy.org/doc)",
            "From Python to NumPy by Nicolas P. Rougier"
        ],
        "practical_project": "Implement standard ML algorithms (Linear Regression, K-Means) from scratch using only pure NumPy vectorization.",
        "estimated_time": "1 - 2 weeks"
    },
    "Scikit-learn": {
        "priority": "High",
        "category": "Machine Learning",
        "recommended_action": "Master Scikit-Learn Pipelines, ColumnTransformer, GridSearchCV/RandomizedSearchCV, cross-validation strategies, and metric evaluation.",
        "learning_resources": [
            "Scikit-Learn API Reference & Examples",
            "Hands-On Machine Learning with Scikit-Learn"
        ],
        "practical_project": "Design a reusable, production-ready ML preprocessing and modeling pipeline with serialized artifact export.",
        "estimated_time": "2 - 3 weeks"
    },
    "Statistics": {
        "priority": "High",
        "category": "Mathematics & Analytics",
        "recommended_action": "Understand probability distributions, hypothesis testing (t-tests, ANOVA, Chi-Square), p-values, confidence intervals, and Bayesian inference.",
        "learning_resources": [
            "Khan Academy Statistics and Probability",
            "Practical Statistics for Data Scientists (Bruce & Bruce)",
            "StatQuest with Josh Starmer Video Series"
        ],
        "practical_project": "Design an A/B testing statistical framework calculating sample sizes, power, and statistical significance.",
        "estimated_time": "3 - 4 weeks"
    },
    "Data Visualization": {
        "priority": "Medium",
        "category": "Data Analytics",
        "recommended_action": "Learn visual design principles, storytelling with data, chart selection rules, and dashboard layouts.",
        "learning_resources": [
            "Storytelling with Data by Cole Nussbaumer Knaflic",
            "Information Dashboard Design by Stephen Few"
        ],
        "practical_project": "Create a multi-chart visual executive report illustrating sales growth, retention curves, and anomalies.",
        "estimated_time": "2 weeks"
    },
    "Matplotlib": {
        "priority": "Normal",
        "category": "Data Visualization",
        "recommended_action": "Master figure/axes object-oriented plotting, subplots, custom styling, annotations, and exporting high-resolution graphics.",
        "learning_resources": [
            "Matplotlib User Guide (matplotlib.org)",
            "Python Graph Gallery (python-graph-gallery.com)"
        ],
        "practical_project": "Build an automated statistical report generation script generating publication-grade plots and confusion matrices.",
        "estimated_time": "1 - 2 weeks"
    },
    "Data Analysis": {
        "priority": "High",
        "category": "Data Analytics",
        "recommended_action": "Develop business acumen, metric formulation, cohort analysis, funnel tracking, and structured problem-solving frameworks.",
        "learning_resources": [
            "Google Data Analytics Professional Certificate",
            "Lean Analytics (Croll & Yoskovitz)"
        ],
        "practical_project": "Perform a complete churn and customer lifetime value (LTV) cohort analysis on transactional logs.",
        "estimated_time": "3 - 4 weeks"
    },
    "Excel": {
        "priority": "Medium",
        "category": "Analytics Tool",
        "recommended_action": "Master XLOOKUP, INDEX/MATCH, Pivot Tables, Power Query data transformation, and financial/statistical modeling formulas.",
        "learning_resources": [
            "Microsoft Excel Video Training",
            "ExcelIsFun YouTube Channel Guides"
        ],
        "practical_project": "Build a dynamic financial forecasting model with automated Power Query ETL pipelines and sensitivity analysis.",
        "estimated_time": "2 weeks"
    },
    "Tableau": {
        "priority": "Medium",
        "category": "BI & Analytics",
        "recommended_action": "Master Tableau calculated fields, Level of Detail (LOD) expressions, parameters, dynamic dashboards, and dashboard actions.",
        "learning_resources": [
            "Tableau Free Training Videos (tableau.com/learn)",
            "Tableau Desktop Specialist Certification Guide"
        ],
        "practical_project": "Develop an interactive sales performance dashboard with drill-down geographic maps and dynamic KPIs.",
        "estimated_time": "3 - 4 weeks"
    },
    "Power BI": {
        "priority": "Medium",
        "category": "BI & Analytics",
        "recommended_action": "Learn DAX (Data Analysis Expressions), Power Query M language, star schema data modeling, and Power BI Service publishing.",
        "learning_resources": [
            "Microsoft Learn: Power BI Data Analyst (PL-300) Track",
            "SQLBI: The Definitive Guide to DAX (Marco Russo & Alberto Ferrari)"
        ],
        "practical_project": "Build a corporate BI dashboard connecting multi-source SQL databases with DAX time-intelligence metrics.",
        "estimated_time": "3 - 4 weeks"
    },

    # Software Engineering Core
    "Data Structures": {
        "priority": "High",
        "category": "Computer Science Core",
        "recommended_action": "Master arrays, linked lists, stacks, queues, hash tables, binary search trees, heaps, and graphs.",
        "learning_resources": [
            "NeetCode.io Data Structures Roadmap",
            "Introduction to Algorithms (CLRS)",
            "Grokking Algorithms by Aditya Bhargava"
        ],
        "practical_project": "Implement self-balancing AVL trees, min-heaps, and graph traversal algorithms with benchmarked runtime tests.",
        "estimated_time": "4 - 6 weeks"
    },
    "Algorithms": {
        "priority": "High",
        "category": "Computer Science Core",
        "recommended_action": "Master sorting/searching, dynamic programming, greedy algorithms, divide-and-conquer, graph algorithms (BFS/DFS, Dijkstra), and Big-O notation.",
        "learning_resources": [
            "LeetCode 75 / Blind 75 Problem Sets",
            "Coursera: Algorithms Part I & II (Princeton)",
            "MIT OpenCourseWare 6.006: Introduction to Algorithms"
        ],
        "practical_project": "Solve 100+ standard algorithmic challenges and benchmark Big-O time and space trade-offs.",
        "estimated_time": "6 - 8 weeks"
    },
    "OOP": {
        "priority": "High",
        "category": "Software Architecture",
        "recommended_action": "Master Encapsulation, Inheritance, Polymorphism, Abstraction, SOLID principles, and Gang of Four (GoF) design patterns.",
        "learning_resources": [
            "Refactoring Guru Design Patterns Guide (refactoring.guru)",
            "Clean Code by Robert C. Martin",
            "Head First Design Patterns"
        ],
        "practical_project": "Architect a modular e-commerce order processing system utilizing Factory, Strategy, and Observer design patterns.",
        "estimated_time": "3 - 4 weeks"
    },
    "REST API": {
        "priority": "High",
        "category": "Web Services",
        "recommended_action": "Understand HTTP verbs, status codes, idempotency, REST architectural constraints, HATEOAS, versioning, and OpenAPI specs.",
        "learning_resources": [
            "RESTful API Design Best Practices (RFC specs / restfulapi.net)",
            "Postman API Design & Testing Tutorials"
        ],
        "practical_project": "Design and document an enterprise RESTful API adhering strictly to RFC standards with automated Postman test suites.",
        "estimated_time": "2 weeks"
    },
    "Git": {
        "priority": "High",
        "category": "DevOps / Version Control",
        "recommended_action": "Master branching models (GitFlow, trunk-based), rebasing, cherry-picking, merge conflict resolution, and pull request workflows.",
        "learning_resources": [
            "Pro Git Book by Scott Chacon (git-scm.com/book)",
            "Learn Git Branching Interactive Tutorial (learngitbranching.js.org)"
        ],
        "practical_project": "Manage a collaborative multi-developer repository enforcing branch protection, semantic commits, and PR reviews.",
        "estimated_time": "1 - 2 weeks"
    },
    "Linux": {
        "priority": "High",
        "category": "Systems & OS",
        "recommended_action": "Master shell scripting (Bash), file permissions, process management (systemd, ps, top), networking tools (netstat, curl), and package managers.",
        "learning_resources": [
            "Linux Journey Interactive Guide (linuxjourney.com)",
            "The Linux Command Line by William Shotts",
            "OverTheWire Bandit Wargame"
        ],
        "practical_project": "Write Bash automation scripts for server health auditing, automated log rotation, and service watchdog restarts.",
        "estimated_time": "2 - 3 weeks"
    },

    # Databases & Storage
    "SQL": {
        "priority": "High",
        "category": "Database",
        "recommended_action": "Master multi-table joins, subqueries, CTEs (Common Table Expressions), window functions (ROW_NUMBER, RANK, LEAD/LAG), and query optimization.",
        "learning_resources": [
            "Mode Analytics SQL Tutorial for Data Analysis",
            "Use The Index, Luke! (SQL Indexing Guide)",
            "LeetCode Database Problem Set"
        ],
        "practical_project": "Optimize slow database queries on a 1M-row database using EXPLAIN ANALYZE, indexing strategies, and normalized schemas.",
        "estimated_time": "3 - 4 weeks"
    },
    "MySQL": {
        "priority": "Medium",
        "category": "Relational Database",
        "recommended_action": "Study InnoDB storage engine, transactions (ACID), isolation levels, locking mechanisms, indexing (B-Trees), and replication.",
        "learning_resources": [
            "MySQL Official Reference Manual",
            "High Performance MySQL (Baron Schwartz et al.)"
        ],
        "practical_project": "Design a normalized database schema with stored procedures, triggers, foreign keys, and connection pooling.",
        "estimated_time": "2 - 3 weeks"
    },
    "MongoDB": {
        "priority": "Medium",
        "category": "NoSQL Database",
        "recommended_action": "Master JSON/BSON document modeling, Aggregation Pipeline ($match, $group, $lookup), indexing, and replica sets.",
        "learning_resources": [
            "MongoDB University Free Courses (learn.mongodb.com)",
            "MongoDB The Definitive Guide"
        ],
        "practical_project": "Build a real-time analytics aggregation service leveraging MongoDB aggregation pipelines with compound indexes.",
        "estimated_time": "2 - 3 weeks"
    },

    # Cloud & DevOps
    "AWS": {
        "priority": "High",
        "category": "Cloud Computing",
        "recommended_action": "Master core services: IAM policies, EC2, S3, RDS, Lambda serverless, VPC networking, CloudWatch, and CloudFormation.",
        "learning_resources": [
            "AWS Certified Solutions Architect Associate (SAA-C03) Training",
            "AWS Skill Builder Free Digital Training",
            "Adrian Cantrill AWS Courses"
        ],
        "practical_project": "Architect a highly available, fault-tolerant web infrastructure in AWS spanning multiple availability zones with Auto Scaling.",
        "estimated_time": "5 - 7 weeks"
    },
    "Azure": {
        "priority": "Medium",
        "category": "Cloud Computing",
        "recommended_action": "Master Azure Resource Manager (ARM), Virtual Machines, App Services, Azure SQL, Blob Storage, Virtual Networks, and Azure Entra ID.",
        "learning_resources": [
            "Microsoft Learn: Azure Administrator (AZ-104) Path",
            "John Savill Azure Master Class on YouTube"
        ],
        "practical_project": "Deploy a containerized application onto Azure App Service with Azure SQL Database and Key Vault secret management.",
        "estimated_time": "4 - 5 weeks"
    },
    "Docker": {
        "priority": "High",
        "category": "Containerization",
        "recommended_action": "Master Dockerfile best practices (multi-stage builds, non-root users), image optimization, Docker Compose, volumes, and networking.",
        "learning_resources": [
            "Docker Official Documentation (docs.docker.com)",
            "Docker Mastery Course by Bret Fisher",
            "Play with Docker Hands-on Labs"
        ],
        "practical_project": "Containerize a full-stack multi-container application (Frontend + Backend + DB + Redis) using Docker Compose with health checks.",
        "estimated_time": "2 - 3 weeks"
    },
    "Kubernetes": {
        "priority": "High",
        "category": "Container Orchestration",
        "recommended_action": "Understand Pods, Deployments, Services, Ingress, ConfigMaps/Secrets, PersistentVolumes, Helm charts, and HPA (Horizontal Pod Autoscaler).",
        "learning_resources": [
            "Kubernetes Official Documentation (kubernetes.io)",
            "Certified Kubernetes Administrator (CKA) Study Guide (Mumshad Mannambeth)",
            "Kubernetes Up & Running (Kelsey Hightower et al.)"
        ],
        "practical_project": "Deploy a scalable microservice cluster to local Minikube/k3s with Ingress routing, Helm charts, and auto-scaling.",
        "estimated_time": "4 - 6 weeks"
    },
    "Terraform": {
        "priority": "High",
        "category": "Infrastructure as Code",
        "recommended_action": "Master HCL syntax, provider configuration, state management (remote backend, locking), modules, variables, and terraform plan/apply workflows.",
        "learning_resources": [
            "HashiCorp Terraform Associate Tutorials (developer.hashicorp.com)",
            "Terraform: Up and Running by Yevgeniy Brikman"
        ],
        "practical_project": "Write modular Terraform code to provision a complete VPC network, security groups, and EC2 instances on AWS with remote S3 state.",
        "estimated_time": "3 - 4 weeks"
    },
    "CI/CD": {
        "priority": "High",
        "category": "DevOps Automation",
        "recommended_action": "Master automated pipeline triggers, multi-stage test execution, artifact caching, container image build/push, and deployment strategies.",
        "learning_resources": [
            "GitHub Actions Documentation & Guides",
            "GitLab CI/CD Pipeline Architecture Tutorials",
            "Continuous Delivery by Jez Humble & David Farley"
        ],
        "practical_project": "Build an automated GitHub Actions pipeline executing linting, unit tests, code coverage, Docker build, and automated staging deployment.",
        "estimated_time": "2 - 3 weeks"
    },
    "Jenkins": {
        "priority": "Medium",
        "category": "DevOps Automation",
        "recommended_action": "Learn Jenkins Declarative Pipelines (Jenkinsfile), agent nodes, secret credentials management, and webhook integrations.",
        "learning_resources": [
            "Jenkins User Handbook (jenkins.io/doc)",
            "CloudBees Jenkins Pipeline Tutorials"
        ],
        "practical_project": "Construct a multi-branch Jenkins pipeline with automated Slack notifications, SonarQube quality gates, and automated deployment.",
        "estimated_time": "2 - 3 weeks"
    },
    "Cloud Architecture": {
        "priority": "High",
        "category": "Cloud Systems",
        "recommended_action": "Study Well-Architected Framework (Security, Reliability, Performance, Cost, Operational Excellence) and disaster recovery.",
        "learning_resources": [
            "AWS Well-Architected Framework Whitepapers",
            "Google Cloud Architecture Framework"
        ],
        "practical_project": "Draft an enterprise multi-region disaster recovery architecture document with RPO/RTO SLAs and cost estimations.",
        "estimated_time": "3 - 4 weeks"
    },

    # Advanced AI / MLOps
    "MLOps": {
        "priority": "High",
        "category": "Machine Learning Systems",
        "recommended_action": "Master experiment tracking (MLflow), model registries, data versioning (DVC), model serving pipelines, and data drift monitoring.",
        "learning_resources": [
            "MLOps Specialization by Andrew Ng (Coursera)",
            "MLflow Documentation & Quickstarts (mlflow.org)",
            "Made With ML by Goku Mohandas (madewithml.com)"
        ],
        "practical_project": "Deploy an automated CI/CD MLOps pipeline that trains models, logs metrics to MLflow, registers top models, and serves REST predictions.",
        "estimated_time": "4 - 5 weeks"
    },
    "Computer Vision": {
        "priority": "Medium",
        "category": "Specialized AI",
        "recommended_action": "Learn OpenCV image filtering, convolutional kernels, object detection (YOLO), semantic segmentation, and vision transformers.",
        "learning_resources": [
            "CS231n: Deep Learning for Computer Vision (Stanford)",
            "OpenCV Python Tutorials",
            "PyImageSearch Deep Learning Guides"
        ],
        "practical_project": "Build a real-time object detection and tracking pipeline using YOLO and OpenCV from a live video stream.",
        "estimated_time": "4 - 6 weeks"
    },
    "NLP": {
        "priority": "High",
        "category": "Specialized AI",
        "recommended_action": "Master tokenization, TF-IDF, Word2Vec, Transformer architectures (BERT, RoBERTa), HuggingFace library, and Named Entity Recognition (NER).",
        "learning_resources": [
            "Hugging Face NLP Course (huggingface.co/course)",
            "CS224N: Natural Language Processing with Deep Learning (Stanford)",
            "Speech and Language Processing (Jurafsky & Martin)"
        ],
        "practical_project": "Fine-tune a pretrained BERT model for sentiment analysis or resume information extraction using HuggingFace Transformers.",
        "estimated_time": "4 - 6 weeks"
    },

    # Cybersecurity
    "Network Security": {
        "priority": "High",
        "category": "Cybersecurity",
        "recommended_action": "Understand OSI model layers, TCP/IP handshake, DNS/DHCP protocols, TLS/SSL encryption, VPNs, and subnetting.",
        "learning_resources": [
            "CompTIA Security+ Study Guide",
            "Professor Messer's Security+ Training Course",
            "TCP/IP Illustrated by W. Richard Stevens"
        ],
        "practical_project": "Set up a segmented virtual network with pfSense firewall rules, NAT, and encrypted WireGuard VPN tunnels.",
        "estimated_time": "3 - 5 weeks"
    },
    "Wireshark": {
        "priority": "High",
        "category": "Cybersecurity Tooling",
        "recommended_action": "Learn packet capture filters, display filter syntax, TCP stream reconstruction, protocol dissectors, and anomaly detection.",
        "learning_resources": [
            "Wireshark Official User Guide (wireshark.org/docs)",
            "Wireshark Network Analysis by Laura Chappell"
        ],
        "practical_project": "Analyze PCAP packet captures from a simulated malware infection to extract exfiltrated data and command-and-control IPs.",
        "estimated_time": "2 weeks"
    },
    "Ethical Hacking": {
        "priority": "High",
        "category": "Cybersecurity",
        "recommended_action": "Learn penetration testing methodology (Reconnaissance, Scanning, Exploitation, Post-Exploitation, Reporting) and OWASP Top 10.",
        "learning_resources": [
            "TryHackMe Hands-on Security Pathways (tryhackme.com)",
            "HackTheBox Academy (hackthebox.com)",
            "PortSwigger Web Security Academy"
        ],
        "practical_project": "Complete 15+ vulnerable machine labs on TryHackMe/HackTheBox and write professional penetration testing reports.",
        "estimated_time": "6 - 8 weeks"
    },
    "Firewall": {
        "priority": "High",
        "category": "Cybersecurity",
        "recommended_action": "Master stateful vs stateless packet inspection, Next-Gen Firewalls (NGFW), iptables/nftables, and intrusion prevention policies.",
        "learning_resources": [
            "pfSense Documentation & Guides",
            "Linux iptables Pocket Reference"
        ],
        "practical_project": "Configure a Linux server with custom iptables/UFW rules defending against port scans, SYN floods, and brute-force SSH attacks.",
        "estimated_time": "2 weeks"
    },
    "Cybersecurity": {
        "priority": "High",
        "category": "Cybersecurity",
        "recommended_action": "Study CIA triad, defense in depth, zero-trust architecture, threat modeling, and incident response lifecycles (NIST framework).",
        "learning_resources": [
            "NIST Cybersecurity Framework (CSF) Documentation",
            "SANS Cyber Aces Online Free Tutorials"
        ],
        "practical_project": "Create a comprehensive threat model and risk assessment matrix for a cloud-hosted e-commerce application.",
        "estimated_time": "3 - 4 weeks"
    },
    "Nmap": {
        "priority": "High",
        "category": "Cybersecurity Tooling",
        "recommended_action": "Master TCP SYN scan (-sS), UDP scan (-sU), service version detection (-sV), OS fingerprinting (-O), and Nmap Scripting Engine (NSE).",
        "learning_resources": [
            "Nmap Network Scanning by Gordon Lyon (Fyodor)",
            "Nmap Official Documentation (nmap.org/book)"
        ],
        "practical_project": "Write custom Lua NSE scripts to automate the detection of outdated software headers and open vulnerable ports.",
        "estimated_time": "1 - 2 weeks"
    },
    "Metasploit": {
        "priority": "Medium",
        "category": "Cybersecurity Tooling",
        "recommended_action": "Understand Metasploit framework architecture (msfconsole, payloads, exploits, auxiliary modules, Meterpreter, msfvenom).",
        "learning_resources": [
            "Metasploit Unleashed Free Course (Offensive Security)",
            "Metasploit: The Penetration Tester's Guide"
        ],
        "practical_project": "Execute controlled exploitation and post-exploitation privilege escalation on an isolated vulnerable lab machine (Metasploitable).",
        "estimated_time": "2 - 3 weeks"
    },
    "SIEM": {
        "priority": "High",
        "category": "Cybersecurity Operations",
        "recommended_action": "Learn log ingestion, event correlation rules, alert tuning, Splunk / Elastic Security (ELK), and SOC triage workflows.",
        "learning_resources": [
            "Splunk Free Training & Fundamentals",
            "Elastic Security Documentation & SIEM Guides"
        ],
        "practical_project": "Set up a local Elastic Security / Wazuh SIEM pipeline ingesting Sysmon logs and trigger automated alert rules for suspicious PowerShell execution.",
        "estimated_time": "3 - 4 weeks"
    },
    "Penetration Testing": {
        "priority": "High",
        "category": "Cybersecurity",
        "recommended_action": "Master web, network, and active directory penetration testing techniques following PTES (Penetration Testing Execution Standard).",
        "learning_resources": [
            "The Hacker Playbook 3 by Peter Kim",
            "PortSwigger Web Security Academy Free Training"
        ],
        "practical_project": "Conduct an authorized penetration test on an OWASP Juice Shop instance, identifying and exploiting Top 10 vulnerabilities.",
        "estimated_time": "5 - 7 weeks"
    }
}


def get_suggestion_for_skill(skill_name: str, target_career: str = "") -> Dict[str, Any]:
    """
    Retrieves the structured improvement suggestion for a given skill.
    If the skill is not in the curated knowledge base, dynamically generates
    a meaningful, professional learning recommendation.
    """
    skill_clean = str(skill_name or "").strip()
    if not skill_clean:
        return {
            "skill": "General Technical Competency",
            "priority": "Normal",
            "recommended_action": "Strengthen domain fundamentals through structured coursework and project practice.",
            "learning_resources": ["Official Documentation", "Coursera / edX Technical Specializations"],
            "practical_project": "Implement a portfolio demonstration project applying core concepts.",
            "estimated_time": "2 - 4 weeks"
        }

    # Direct match in curated database
    if skill_clean in SKILL_SUGGESTIONS_DB:
        data = SKILL_SUGGESTIONS_DB[skill_clean].copy()
        data["skill"] = skill_clean
        return data

    # Case-insensitive exact match
    for k, v in SKILL_SUGGESTIONS_DB.items():
        if k.lower() == skill_clean.lower():
            data = v.copy()
            data["skill"] = k
            return data

    # Alias / Prefix variations (e.g. React.js -> React, Nodejs -> Node.js)
    clean_lower = skill_clean.lower().replace(".js", "").replace("js", "").strip()
    for k, v in SKILL_SUGGESTIONS_DB.items():
        k_lower = k.lower().replace(".js", "").replace("js", "").strip()
        if clean_lower == k_lower:
            data = v.copy()
            data["skill"] = skill_clean
            return data

    # Dynamic intelligent fallback for emerging or uncataloged skills
    career_ctx = f" for {target_career}" if target_career else ""
    return {
        "skill": skill_clean,
        "priority": "Medium",
        "category": "Domain Competency",
        "recommended_action": f"Acquire foundational and practical working knowledge of {skill_clean}{career_ctx} by studying industry documentation and practicing code examples.",
        "learning_resources": [
            f"Official {skill_clean} Documentation & Quickstart Guides",
            f"FreeCodeCamp / YouTube in-depth {skill_clean} tutorials",
            "GitHub open-source reference implementations"
        ],
        "practical_project": f"Build a focused prototype or module that integrates {skill_clean} into an end-to-end workflow.",
        "estimated_time": "2 - 3 weeks"
    }


def generate_improvement_suggestions(missing_skills: List[str], target_career: str = "") -> List[Dict[str, Any]]:
    """
    Generates a structured list of actionable competency improvement suggestions
    for all missing skills for a target career.
    """
    suggestions = []
    seen = set()
    for s in missing_skills:
        s_clean = str(s).strip()
        if s_clean and s_clean.lower() not in seen:
            seen.add(s_clean.lower())
            suggestion = get_suggestion_for_skill(s_clean, target_career)
            suggestions.append(suggestion)

    # Sort suggestions: High priority first, then Medium, then Normal
    priority_order = {"High": 0, "Medium": 1, "Normal": 2}
    suggestions.sort(key=lambda x: priority_order.get(x.get("priority", "Normal"), 3))
    return suggestions
