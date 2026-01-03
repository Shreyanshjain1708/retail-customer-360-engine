# Retail Customer 360 Engine

An end-to-end, production-oriented **Customer 360 & Personalization Engine** for retail organizations, designed to unify customer data, generate actionable insights, and enable data-driven marketing and decision-making at scale.

This project demonstrates **real-world Data Science, Machine Learning, and MLOps capabilities** across customer analytics, orchestration, model deployment, and automation.

---

## 🚀 Business Problem

Retail businesses struggle with:
- Fragmented customer data across channels
- Low campaign conversion due to poor personalization
- Lack of explainable customer insights
- Manual, non-scalable ML workflows

**Objective:**  
Build a unified Customer 360 system that enables segmentation, churn prediction, recommendations, and actionable insights through automated, production-ready pipelines.

---

## 🎯 Key Capabilities

- Unified Customer 360 data model
- Automated data ingestion and transformation
- Customer segmentation (RFM, clustering)
- Churn prediction and customer lifetime modeling
- Personalized product recommendations
- End-to-end ML pipeline orchestration with Airflow
- REST APIs for real-time inference
- Explainability-ready architecture
- Production-grade project structure

---

## 🧱 Architecture Overview


Data Sources
│
├── Raw Customer Transactions
├── Behavioral Events
└── Product Metadata
│
▼
Data Processing & Feature Engineering
│
▼
ML Models
├── Customer Segmentation
├── Churn Prediction
└── Recommendation Engine
│
▼
Orchestration (Apache Airflow)
│
▼
Model Serving (FastAPI)
│
▼
Downstream Consumption
├── Marketing Campaigns
├── Dashboards
└── Business Applications

---

## 🧠 Machine Learning Components

### 1. Customer Segmentation
- RFM-based feature engineering
- Clustering (KMeans / hierarchical)
- Behavioral and transactional enrichment
- Segment profiling for marketing teams

### 2. Churn Prediction
- Supervised learning models (Logistic Regression, XGBoost)
- Feature importance analysis
- Business-aligned evaluation metrics

### 3. Recommendation System
- Collaborative filtering
- Association rule mining
- Personalized product bundle recommendations

---

## ⚙️ MLOps & Automation

- Apache Airflow DAGs for:
  - Data ingestion
  - Feature engineering
  - Model training
  - Batch prediction
- Modular and reproducible pipelines
- Environment-based configuration
- Scalable and production-ready folder structure

---

## 🛠️ Tech Stack

| Category | Tools |
|-------|------|
| Language | Python |
| ML | Scikit-learn, XGBoost |
| Orchestration | Apache Airflow |
| APIs | FastAPI |
| Data Processing | Pandas, NumPy |
| Experimentation | Jupyter |
| Version Control | Git |
| MLOps Concepts | Pipelines, automation, modular design |

---

## 📂 Project Structure

retail-customer-360-engine/
│
├── data/
│ ├── raw/
│ └── processed/
│
├── notebooks/
│ ├── EDA
│ └── experiments
│
├── src/
│ ├── ingestion/
│ ├── feature_engineering/
│ ├── models/
│ ├── evaluation/
│ └── utils/
│
├── airflow/
│ └── dags/
│
├── api/
│ └── fastapi_app/
│
├── requirements.txt
├── README.md
└── LICENSE

---

## ▶️ How to Run (Local)

```bash
# Clone repository
git clone https://github.com/Shreyanshjain1708/retail-customer-360-engine.git
cd retail-customer-360-engine

# Install dependencies
pip install -r requirements.txt

# Run Airflow (example)
airflow standalone

# Start API
uvicorn api.fastapi_app.main:app --reload
📊 Evaluation & Results

Improved customer targeting through segmentation

Churn risk identification for proactive retention

Personalized recommendations aligned with purchase behavior

Modular pipelines ready for scaling and deployment

📌 Future Enhancements

Deep learning-based recommender models

SHAP-based explainability

Real-time streaming ingestion

Cloud deployment (AWS/GCP)

CI/CD with GitHub Actions

Feature store integration

👤 Author

Shreyansh Jain
Data Scientist | Machine Learning Engineer
🔗 LinkedIn: linkedin.com/in/shreyansh-jain-574857172
📧 Email: 17shreyanshj@gmail.com

📄 License

This project is licensed under the MIT License.

---

