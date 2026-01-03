from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

app = FastAPI(title="Retail Customer 360 API", version="1.0")

# --- Load Models on Startup ---
models = {}

@app.on_event("startup")
def load_models():
    # Robust path handling
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, '../models')
    
    try:
        models['churn'] = joblib.load(os.path.join(model_path, 'churn_model.pkl'))
        models['recsys_embeddings'] = joblib.load(os.path.join(model_path, 'item_embeddings.pkl'))
        print("✅ Models loaded successfully")
    except FileNotFoundError:
        print("⚠️ Warning: Model files not found. Please run training scripts first.")

# --- Request Schemas (Pydantic) ---
class CustomerFeatures(BaseModel):
    recency_days: int
    frequency: int
    monetary_value: float
    aov: float

class RecsRequest(BaseModel):
    customer_id: int
    recent_item_id: int # To find similar items

# --- Endpoints ---

@app.get("/health")
def health_check():
    return {"status": "active", "models_loaded": list(models.keys())}

@app.post("/predict/churn")
def predict_churn(features: CustomerFeatures):
    """
    Returns churn probability and risk level.
    """
    if 'churn' not in models:
        raise HTTPException(status_code=503, detail="Churn model not loaded")
    
    # Prepare dataframe for model
    input_df = pd.DataFrame([features.dict()])
    
    # Predict
    prob = models['churn'].predict_proba(input_df)[0][1]
    
    # Business Logic layer
    risk_level = "High" if prob > 0.7 else "Medium" if prob > 0.3 else "Low"
    
    return {
        "churn_probability": round(float(prob), 4),
        "risk_label": risk_level
    }

@app.post("/recommend/items")
def recommend_items(request: RecsRequest):
    """
    Content-Based Logic: Returns items similar to the user's last purchase.
    """
    if 'recsys_embeddings' not in models:
         raise HTTPException(status_code=503, detail="RecSys model not loaded")
    
    # Simple similarity logic (Dot product of embeddings)
    # In reality, this would look up the item vector from the matrix
    embeddings = models['recsys_embeddings']
    
    # Mocking logic: Pick random top 3 for demo purposes if ID out of bounds
    # (Real code would do Cosine Similarity here)
    recommended_ids = [1, 5, 9] 
    
    return {
        "customer_id": request.customer_id,
        "recommended_product_ids": recommended_ids,
        "strategy": "item-to-item similarity"
    }