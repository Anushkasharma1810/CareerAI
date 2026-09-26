"""
skill_vocabulary.py
-------------------
Configurable technical skill vocabulary used for extraction and matching.
Organized by domain/category.  Not hard-coded for one resume — covers
a broad range of job families.

To extend: add entries to the SKILL_VOCABULARY dict.
"""

# ── Master skill vocabulary ───────────────────────────────────────────────────
# Keys are canonical skill names; values are alias lists for fuzzy matching.
SKILL_VOCABULARY: dict[str, list[str]] = {
    # ── Programming Languages ─────────────────────────────────────────────
    "Python": ["python3", "python 3", "py"],
    "R": ["r programming", "r language", "r stats"],
    "Java": ["java8", "java 11", "java 17", "java se", "java ee"],
    "JavaScript": ["js", "es6", "es2015", "ecmascript", "javascript/typescript", "java script"],
    "Artificial Intelligence": ["ai", "artificial-intelligence"],
    "TypeScript": ["ts", "typescript"],
    "C++": ["c plus plus", "cpp", "c/c++"],
    "C": ["c language", "c programming"],
    "C#": ["csharp", "c sharp", ".net c#"],
    "Go": ["golang"],
    "Rust": ["rust lang", "rust programming"],
    "Swift": ["swift ios"],
    "Kotlin": ["kotlin android"],
    "Scala": ["scala programming"],
    "PHP": ["php7", "php8"],
    "Ruby": ["ruby on rails", "ror"],
    "MATLAB": ["matlab r2023"],
    "Bash": ["shell scripting", "bash scripting", "sh script"],
    "SQL": ["sql queries", "structured query language", "t-sql", "pl/sql"],
    "NoSQL": ["nosql databases"],
    "Dart": ["dart flutter"],

    # ── ML / AI Frameworks ────────────────────────────────────────────────
    "TensorFlow": ["tensorflow 2", "tf2", "tensorflow keras"],
    "PyTorch": ["torch", "pytorch lightning"],
    "Keras": ["keras api"],
    "scikit-learn": ["sklearn", "scikit learn", "scikitlearn"],
    "Hugging Face": ["huggingface", "transformers library", "hf transformers"],
    "spaCy": ["spacy nlp"],
    "NLTK": ["natural language toolkit"],
    "XGBoost": ["xgb", "xgboost classifier"],
    "LightGBM": ["lgbm", "lightgbm"],
    "CatBoost": ["catboost"],
    "OpenCV": ["cv2", "opencv python"],
    "MLflow": ["ml flow"],
    "Weights & Biases": ["wandb", "weights and biases"],
    "Ray": ["ray tune", "ray serve"],
    "ONNX": ["onnx runtime"],

    # ── ML Concepts ───────────────────────────────────────────────────────
    "Machine Learning": ["ml", "supervised learning", "unsupervised learning", "machine-learning"],
    "Deep Learning": ["dl", "neural networks", "deep neural network", "deep-learning"],
    "NLP": ["natural language processing", "text analytics", "text mining", "natural-language-processing"],
    "Computer Vision": ["cv", "image recognition", "image classification"],
    "Reinforcement Learning": ["rl", "q-learning", "policy gradient"],
    "Transfer Learning": ["fine-tuning", "pretrained models"],
    "Generative AI": ["genai", "generative models", "llm", "large language model"],
    "RAG": ["retrieval augmented generation", "retrieval-augmented generation"],
    "Transformers": ["bert", "gpt", "t5", "roberta", "distilbert", "llama", "mistral"],
    "TF-IDF": ["tfidf", "tf idf"],
    "Word Embeddings": ["word2vec", "glove", "fasttext"],
    "Feature Engineering": ["feature extraction", "feature selection"],
    "Hyperparameter Tuning": ["hyperparameter optimization", "optuna", "grid search", "random search"],
    "Model Deployment": ["ml deployment", "model serving", "mlops"],
    "A/B Testing": ["ab testing", "split testing", "experimentation"],

    # ── Data Science & Analytics ─────────────────────────────────────────
    "pandas": ["pandas dataframe", "pd"],
    "NumPy": ["numpy", "np"],
    "SciPy": ["scipy"],
    "Matplotlib": ["matplotlib pyplot"],
    "Seaborn": ["sns"],
    "Plotly": ["plotly dash"],
    "Tableau": ["tableau desktop", "tableau server"],
    "Power BI": ["powerbi", "power bi desktop", "microsoft power bi"],
    "Looker": ["looker studio", "google looker"],
    "Statistical Analysis": ["statistics", "statistical modeling", "hypothesis testing"],
    "Data Visualization": ["data viz", "visualization"],
    "ETL": ["extract transform load", "data pipeline", "data ingestion"],
    "Data Warehousing": ["data warehouse", "edw"],
    "Jupyter": ["jupyter notebook", "jupyter lab", "ipynb"],
    "dbt": ["data build tool", "dbt core", "dbt cloud"],

    # ── Databases ─────────────────────────────────────────────────────────
    "PostgreSQL": ["postgres", "pgsql"],
    "MySQL": ["mysql database"],
    "SQL Server": ["mssql", "microsoft sql server", "t-sql"],
    "Oracle": ["oracle db", "oracle database"],
    "MongoDB": ["mongo", "mongodb atlas"],
    "Redis": ["redis cache"],
    "Cassandra": ["apache cassandra"],
    "DynamoDB": ["aws dynamodb"],
    "Elasticsearch": ["elastic search", "opensearch"],
    "Snowflake": ["snowflake data warehouse"],
    "BigQuery": ["google bigquery", "bq"],
    "Redshift": ["amazon redshift"],
    "SQLite": ["sqlite3"],
    "Neo4j": ["graph database", "neo4j graph"],

    # ── Cloud Platforms ───────────────────────────────────────────────────
    "AWS": ["amazon web services", "amazon aws", "aws cloud"],
    "Azure": ["microsoft azure", "azure cloud"],
    "GCP": ["google cloud platform", "google cloud", "google gcp"],
    "AWS Lambda": ["lambda functions", "serverless aws"],
    "AWS S3": ["amazon s3", "s3 bucket"],
    "AWS EC2": ["amazon ec2", "ec2 instance"],
    "AWS SageMaker": ["sagemaker", "amazon sagemaker"],
    "Azure ML": ["azure machine learning"],
    "Vertex AI": ["google vertex ai"],

    # ── DevOps & Infrastructure ───────────────────────────────────────────
    "Docker": ["docker container", "dockerfile", "docker compose"],
    "Kubernetes": ["k8s", "kubectl", "kubernetes cluster"],
    "Terraform": ["terraform iac", "hashicorp terraform"],
    "Ansible": ["ansible playbook"],
    "Jenkins": ["jenkins ci", "jenkins pipeline"],
    "GitHub Actions": ["github actions ci", "gha"],
    "GitLab CI": ["gitlab pipeline", "gitlab ci/cd"],
    "CI/CD": ["continuous integration", "continuous deployment", "continuous delivery", "devops pipeline"],
    "Helm": ["helm chart", "helm kubernetes"],
    "Prometheus": ["prometheus monitoring"],
    "Grafana": ["grafana dashboard"],
    "Linux": ["ubuntu", "centos", "debian", "linux server", "unix"],
    "Git": ["git version control", "github", "gitlab", "bitbucket"],
    "Nginx": ["nginx web server"],
    "Apache": ["apache httpd", "apache web server"],

    # ── Web Frameworks ────────────────────────────────────────────────────
    "React": ["reactjs", "react.js", "react hooks"],
    "Next.js": ["nextjs", "next js"],
    "Vue.js": ["vuejs", "vue 3", "vue 2"],
    "Angular": ["angularjs", "angular 2+"],
    "Svelte": ["svelte kit"],
    "Node.js": ["nodejs", "node js", "express.js", "expressjs"],
    "Django": ["django rest framework", "drf"],
    "Flask": ["flask python", "flask api"],
    "FastAPI": ["fast api", "fastapi python"],
    "Spring Boot": ["spring framework", "spring mvc"],
    "Laravel": ["laravel php"],
    "Rails": ["ruby on rails", "ror"],
    "GraphQL": ["graph ql", "apollo graphql"],
    "REST API": [
        "restful api", "rest apis", "restful services",
        "restful",                       # bare adjective (e.g. "RESTful endpoints")
        "rest api design", "rest-api", "rest endpoint", "rest endpoints",
        "representational state transfer",
    ],
    "gRPC": ["grpc protocol"],
    "WebSocket": ["websockets", "socket.io"],
    "HTML": ["html5", "hypertext markup language"],
    "CSS": ["css3", "cascading style sheets"],
    "Tailwind CSS": ["tailwindcss", "tailwind"],
    "Bootstrap": ["bootstrap 4", "bootstrap 5"],

    # ── Security ──────────────────────────────────────────────────────────
    "Cybersecurity": ["information security", "infosec", "cyber security"],
    "Penetration Testing": ["pentest", "pen testing", "ethical hacking"],
    "SIEM": ["security information event management", "splunk siem"],
    "OWASP": ["owasp top 10"],
    "Cryptography": ["encryption", "ssl/tls", "tls", "ssl", "pki"],
    "IAM": ["identity access management", "access control", "rbac"],
    "Zero Trust": ["zero trust network", "ztna"],
    "SOC": ["security operations center", "soc analyst"],
    "Vulnerability Assessment": ["vulnerability scanning", "cve", "nessus"],
    "Compliance": ["iso 27001", "soc 2", "hipaa", "pci dss", "gdpr", "nist"],

    # ── Soft / Cross-domain Skills ────────────────────────────────────────
    "Agile": ["agile methodology", "scrum", "kanban", "sprint"],
    "Communication": ["verbal communication", "written communication", "presentation"],
    "Leadership": ["team lead", "team leadership", "project leadership"],
    "Problem Solving": ["analytical thinking", "critical thinking"],
    "Collaboration": ["teamwork", "cross-functional"],
    "Project Management": ["pmp", "jira", "confluence", "project planning"],
}

# Build reverse-lookup: alias → canonical skill name
_ALIAS_TO_CANONICAL: dict[str, str] = {}
for canonical, aliases in SKILL_VOCABULARY.items():
    _ALIAS_TO_CANONICAL[canonical.lower()] = canonical
    for alias in aliases:
        _ALIAS_TO_CANONICAL[alias.lower()] = canonical


def normalize_skill(raw: str) -> str | None:
    """Map a raw skill string to its canonical form, or None if not found."""
    return _ALIAS_TO_CANONICAL.get(raw.lower().strip())


def get_all_canonical_skills() -> list[str]:
    """Return sorted list of all canonical skill names."""
    return sorted(SKILL_VOCABULARY.keys())


def get_skills_by_domain(domain_prefix: str) -> list[str]:
    """Helper to filter skills by rough domain (e.g. 'ML', 'Cloud')."""
    return [k for k in SKILL_VOCABULARY if domain_prefix.lower() in k.lower()]
