"""
dataset_builder.py
------------------
Builds the training dataset.

PRIMARY: resume-atlas real dataset (Apache 2.0)
  File:    data/raw/updated_resume_dataset.csv
  Source:  https://huggingface.co/datasets/ahmedheakl/resume-atlas
  License: Apache 2.0 (free for research and commercial use)
  Content: 4,295 real resume texts pre-processed from 13,389 Atlas samples
           covering AI/ML Engineer, Data Analyst, DevOps Engineer,
           Software Engineer, Web Developer.
  Note:    Non-tech categories (Education, Finance, Sales, etc.) excluded.
           Balanced to 500 samples/class for training.

FALLBACK: if the CSV is not present, generates a clearly-labelled synthetic
          dataset.  Metrics on synthetic data must NOT be reported as
          real-world performance.

The dataset is saved to data/processed/dataset.csv for reproducibility.
"""

import os
import random
import pandas as pd
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parents[2]
RAW_DIR       = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
ATLAS_CSV     = RAW_DIR / "updated_resume_dataset.csv"   # real dataset
KAGGLE_CSV    = RAW_DIR / "UpdatedResumeDataSet.csv"      # legacy Kaggle path
OUTPUT_CSV    = PROCESSED_DIR / "dataset.csv"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── canonical job categories ──────────────────────────────────────────────────
# The set used by the inference predictor and the frontend.
# Must remain consistent between training and inference.
JOB_CATEGORIES = [
    "AI/ML Engineer",
    "Data Analyst",
    "DevOps Engineer",
    "Software Engineer",
    "Web Developer",
]

# ── legacy Kaggle label map (used only if Kaggle CSV is present) ──────────────
KAGGLE_LABEL_MAP = {
    "Data Science":              "AI/ML Engineer",
    "Machine Learning":          "AI/ML Engineer",
    "Artificial Intelligence":   "AI/ML Engineer",
    "Python Developer":          "AI/ML Engineer",
    "ETL Developer":             "AI/ML Engineer",
    "Data Analyst":              "Data Analyst",
    "SQL Developer":             "Data Analyst",
    "Business Analyst":          "Data Analyst",
    "Database":                  "Data Analyst",
    "Java Developer":            "Software Engineer",
    "DotNet Developer":          "Software Engineer",
    "Testing":                   "Software Engineer",
    "Information Technology":    "Software Engineer",
    "SAP Developer":             "Software Engineer",
    "PMO":                       "Software Engineer",
    "Blockchain":                "Software Engineer",
    "Web Designing":             "Web Developer",
    "React Developer":           "Web Developer",
    "Digital Media":             "Web Developer",
    "DevOps Engineer":           "DevOps Engineer",
    "DevOps":                    "DevOps Engineer",
    "Network Security Engineer": "DevOps Engineer",
}

# ── synthetic data templates (fallback only) ──────────────────────────────────
SYNTHETIC_TEMPLATES = {
    "AI/ML Engineer": [
        "machine learning engineer with experience in deep learning neural networks TensorFlow PyTorch "
        "computer vision NLP transformers BERT GPT model training model deployment MLOps scikit-learn "
        "Python data preprocessing feature engineering hyperparameter tuning cross-validation",

        "AI researcher specializing in reinforcement learning natural language processing generative models "
        "GANs variational autoencoders Hugging Face transformers model optimization quantization "
        "A/B testing statistical analysis Python TensorFlow Keras PyTorch SQL",

        "senior ML engineer building recommendation systems fraud detection models real-time inference pipelines "
        "Spark distributed training Kubernetes Docker CI/CD model monitoring drift detection "
        "Python scikit-learn XGBoost LightGBM feature stores MLflow",

        "machine learning scientist with PhD experience in computer vision object detection YOLO "
        "image segmentation transfer learning ResNet EfficientNet data augmentation "
        "Python PyTorch OpenCV CUDA GPU programming research publications",

        "applied ML engineer deploying NLP solutions text classification named entity recognition "
        "sentiment analysis BERT fine-tuning spaCy NLTK TF-IDF word embeddings GloVe Word2Vec "
        "Python Flask REST API production deployment Docker",
    ],
    "Data Analyst": [
        "data analyst proficient in SQL Excel Tableau Power BI business intelligence "
        "data cleaning data wrangling pivot tables VLOOKUP stakeholder reporting KPI dashboards "
        "Google Analytics Adobe Analytics web analytics A/B testing",

        "business intelligence analyst creating dashboards reports SQL Server MySQL PostgreSQL "
        "Power BI Tableau data warehouse ETL data modeling star schema "
        "financial reporting budget analysis forecasting Excel VBA Python",

        "marketing data analyst analyzing campaign performance Google Analytics Google Ads "
        "Facebook Ads SQL Python cohort analysis customer segmentation "
        "Tableau Looker reporting A/B testing conversion rate optimization",

        "operations analyst data-driven process improvement SQL Excel Python "
        "supply chain logistics inventory analysis SQL Server "
        "Power BI Tableau KPI monitoring process documentation lean six sigma",

        "healthcare data analyst SQL Python R clinical data EHR HIPAA compliance "
        "statistical analysis outcomes research quality improvement "
        "Tableau Excel reporting ICD codes CPT codes population health",
    ],
    "Software Engineer": [
        "software engineer Java Spring Boot microservices REST API Docker Kubernetes "
        "CI/CD Jenkins Git Agile Scrum design patterns SOLID principles "
        "SQL PostgreSQL MySQL Redis message queues RabbitMQ Kafka",

        "full-stack software engineer React Node.js TypeScript Python Django "
        "PostgreSQL MongoDB REST GraphQL Docker Git CI/CD unit testing "
        "TDD agile software development distributed systems cloud AWS",

        "backend software engineer Go Python microservices gRPC protobuf "
        "PostgreSQL Redis Kafka distributed systems horizontal scaling "
        "Docker Kubernetes Terraform AWS infrastructure monitoring Prometheus",

        "C++ software engineer game development Unreal Engine rendering pipeline "
        "graphics programming OpenGL Vulkan performance optimization multithreading "
        "memory management systems programming embedded Linux real-time",

        "embedded software engineer C C++ RTOS firmware development "
        "hardware communication protocols I2C SPI UART CAN "
        "ARM Cortex microcontrollers debugging JTAG unit testing automotive",
    ],
    "Web Developer": [
        "frontend web developer React TypeScript Redux HTML5 CSS3 SASS "
        "responsive design mobile-first accessibility WCAG webpack Vite "
        "REST API integration Jest Cypress Git Figma UI/UX collaboration",

        "full stack web developer Vue.js Node.js Express MongoDB "
        "GraphQL REST API authentication JWT OAuth PostgreSQL "
        "Docker deployment Heroku AWS Amplify responsive web design",

        "React developer building single-page applications hooks context API "
        "Redux state management styled-components Material UI TypeScript "
        "RESTful APIs Git GitHub CI/CD Netlify Vercel frontend performance",

        "web developer specializing in Next.js server-side rendering SSR SSG "
        "TypeScript Tailwind CSS headless CMS Sanity Contentful "
        "SEO optimization Core Web Vitals performance PostgreSQL Prisma",

        "WordPress PHP web developer custom themes plugins "
        "JavaScript jQuery MySQL responsive design CSS HTML "
        "WooCommerce e-commerce SEO Google PageSpeed deployment cPanel",
    ],
    "DevOps Engineer": [
        "DevOps engineer Kubernetes Docker AWS Terraform CI/CD pipelines "
        "Jenkins GitHub Actions GitLab CI infrastructure as code "
        "monitoring Prometheus Grafana ELK stack log aggregation alerting",

        "site reliability engineer SRE Google Cloud Platform GCP Kubernetes "
        "Terraform Ansible configuration management incident response "
        "on-call SLO SLI error budget monitoring observability distributed systems",

        "cloud DevOps engineer AWS Azure Kubernetes Helm charts Terraform "
        "Docker containerization CI/CD GitOps ArgoCD Flux "
        "Prometheus Grafana alerting security scanning SAST DAST",

        "Linux systems administrator DevOps automation Bash Python scripting "
        "Ansible configuration management Nagios Zabbix monitoring "
        "Apache nginx database administration MySQL PostgreSQL backup recovery",

        "platform engineer internal developer platform Kubernetes operators "
        "Terraform modules Crossplane CI/CD GitHub Actions "
        "service mesh Istio Envoy observability OpenTelemetry Jaeger",
    ],
}


def build_synthetic_dataset(samples_per_class: int = 40) -> pd.DataFrame:
    """
    Generate synthetic training dataset — clearly labelled as synthetic.
    Used ONLY when the real resume-atlas CSV is absent.
    """
    random.seed(42)
    records = []

    for category, templates in SYNTHETIC_TEMPLATES.items():
        for i in range(samples_per_class):
            selected = random.sample(templates, min(2, len(templates)))
            text = " ".join(selected)
            words = text.split()
            if len(words) > 20:
                drop_count = random.randint(0, 5)
                for _ in range(drop_count):
                    idx = random.randint(0, len(words) - 1)
                    words.pop(idx)
            records.append({"text": " ".join(words), "label": category, "source": "synthetic"})

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def load_atlas_dataset(path: Path) -> pd.DataFrame | None:
    """
    Load the pre-processed resume-atlas CSV.
    Expected columns: text, label, source
    """
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except Exception as e:
        print(f"[WARNING] Failed to read Atlas CSV: {e}")
        return None

    if "text" not in df.columns or "label" not in df.columns:
        print(f"[WARNING] Unexpected Atlas CSV columns: {list(df.columns)}")
        return None

    df = df[["text", "label", "source"]].dropna()
    df = df[df["text"].str.strip().str.len() > 50]
    # Keep only known canonical labels
    df = df[df["label"].isin(JOB_CATEGORIES)]
    return df.reset_index(drop=True)


def load_kaggle_dataset(path: Path) -> pd.DataFrame | None:
    """
    Load legacy Kaggle UpdatedResumeDataSet.csv if present.
    """
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")

    if "Category" not in df.columns or "Resume" not in df.columns:
        return None

    df = df.rename(columns={"Category": "raw_label", "Resume": "text"})
    df["label"] = df["raw_label"].map(KAGGLE_LABEL_MAP)
    df = df.dropna(subset=["label"])
    df["source"] = "kaggle"
    df = df[["text", "label", "source"]].dropna()
    df = df[df["text"].str.strip().str.len() > 50]
    # Keep only canonical labels that exist in JOB_CATEGORIES
    df = df[df["label"].isin(JOB_CATEGORIES)]
    return df.reset_index(drop=True)


def build_dataset(force_synthetic: bool = False) -> pd.DataFrame:
    """
    Main entry point.
    Priority:
      1. Atlas real dataset (updated_resume_dataset.csv)
      2. Kaggle legacy CSV (UpdatedResumeDataSet.csv)
      3. Synthetic fallback
    Saves to data/processed/dataset.csv.
    """
    print("Building dataset...")

    real_df = None

    if not force_synthetic:
        # Try Atlas first
        real_df = load_atlas_dataset(ATLAS_CSV)
        if real_df is not None:
            print(f"  Loaded Atlas dataset: {len(real_df)} samples, "
                  f"{real_df['label'].nunique()} categories")
            data_note = "resume_atlas_real"
        else:
            # Try legacy Kaggle
            real_df = load_kaggle_dataset(KAGGLE_CSV)
            if real_df is not None:
                print(f"  Loaded Kaggle dataset: {len(real_df)} samples, "
                      f"{real_df['label'].nunique()} categories")
                data_note = "kaggle_real"

    if real_df is None:
        synth_df = build_synthetic_dataset(samples_per_class=40)
        combined = synth_df
        data_note = "synthetic"
        print("  [NOTE] No real dataset found — using synthetic-only.")
        print("         Run: python _build_real_dataset.py  to fetch the Atlas dataset.")
    else:
        # Cap to 500/class to prevent severe imbalance
        MAX_PER_CLASS = 500
        frames = [
            grp.sample(min(len(grp), MAX_PER_CLASS), random_state=42)
            for _, grp in real_df.groupby("label")
        ]
        combined = pd.concat(frames, ignore_index=True)

    combined = combined.drop_duplicates(subset=["text"]).reset_index(drop=True)
    combined["data_note"] = data_note

    # Distribution report
    print("\n  Label distribution:")
    for lbl, cnt in combined["label"].value_counts().items():
        print(f"    {lbl:<35} {cnt:>5} samples")

    combined.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  Dataset saved -> {OUTPUT_CSV}")
    print(f"  Total samples : {len(combined)}")
    return combined


if __name__ == "__main__":
    df = build_dataset()
