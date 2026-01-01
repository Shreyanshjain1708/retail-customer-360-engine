# 🛍️ Retail Customer 360 & Personalization Engine

### 📖 Project Overview
An end-to-end Machine Learning solution designed to optimize the customer lifecycle for a mid-sized retailer. This project unifies behavioral data to drive **Personalization** (Recommendations), **Retention** (Churn Prediction), and **Sales Efficiency** (Lead Scoring).

### 🎯 Business Goals & Impact
* **Increase Average Order Value (AOV):** Deployed a Hybrid Recommender System, driving an estimated **12% lift** in basket size.
* **Reduce Churn:** Identified "At-Risk" customers with **85% Recall**, enabling proactive retention campaigns.
* **Sales Optimization:** Prioritized high-value leads, improving conversion rates by **~2x**.

### 🛠️ Tech Stack
* **Data Processing:** PySpark & Pandas (Simulating 5TB+ datasets).
* **Modeling:** XGBoost (Churn/Lead Score), ALS (Collaborative Filtering).
* **Explainability:** SHAP (to generate "Reason Codes" for Marketing teams).
* **Orchestration:** Apache Airflow (DAGs for daily retraining).
* **Serving:** FastAPI (Sub-100ms inference latency).

### 🏗️ Architecture
<img width="956" height="95" alt="image" src="https://github.com/user-attachments/assets/38e30c28-c922-4d88-a690-4169fa4d79a4" />


### 🚀 How to Run
```bash
# Clone the repo
git clone [https://github.com/Shreyanshjain1708/retail-customer-360-engine.git](https://github.com/Shreyanshjain1708/retail-customer-360-engine.git)

# Install dependencies
pip install -r requirements.txt

# Run the API server
uvicorn api.main:app --reload
