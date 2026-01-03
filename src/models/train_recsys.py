import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
import joblib
import logging

logging.basicConfig(level=logging.INFO)

class RecSysTrainer:
    def __init__(self):
        self.model = TruncatedSVD(n_components=5, random_state=42)
    
    def train(self):
        logging.info("Loading interaction data...")
        # Mock User-Item Interaction Matrix (Rows: Users, Cols: Products)
        # In production, this comes from pivot tables of sales data
        data = np.random.rand(100, 20) # 100 users, 20 products
        data[data < 0.8] = 0 # Sparsity simulation

        logging.info(f"Training SVD on matrix shape: {data.shape}")
        self.model.fit(data)
        
        # Calculate Item Embeddings (V matrix)
        item_embeddings = self.model.components_.T
        
        # Save both model and embeddings
        joblib.dump(self.model, 'src/models/recsys_model.pkl')
        joblib.dump(item_embeddings, 'src/models/item_embeddings.pkl')
        logging.info("Recommendation artifacts saved.")

if __name__ == "__main__":
    trainer = RecSysTrainer()
    trainer.train()