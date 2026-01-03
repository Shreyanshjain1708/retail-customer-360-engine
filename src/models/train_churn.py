import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import logging
import sys
import os

# Add src to path to import FeatureEngineer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.features.build_features import FeatureEngineer

logging.basicConfig(level=logging.INFO)

class ChurnTrainer:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            use_label_encoder=False,
            n_estimators=100,
            learning_rate=0.05
        )
        self.feature_engineer = FeatureEngineer()

    def load_mock_data(self):
        # In a real scenario, this loads from SQL/S3
        # Creating synthetic data for demonstration
        logging.info("Loading training data...")
        df = pd.DataFrame({
            'customer_id': range(1000),
            'transaction_date': [pd.Timestamp('2023-01-01') + pd.Timedelta(days=i%30) for i in range(1000)],
            'amount': [np.random.randint(10, 500) for _ in range(1000)],
            'product_id': [np.random.choice(['A','B','C']) for _ in range(1000)]
        })
        # Synthetic Target (1 = Churned) based on logic for the model to learn
        # If amount < 50, high chance of churn
        labels = [1 if x < 50 else 0 for x in df['amount']] 
        return df, labels

    def train(self):
        raw_df, y = self.load_mock_data()
        
        # 1. Feature Engineering
        X = self.feature_engineer.generate_rfm_features(raw_df)
        
        # Align labels with features (since grouping changes shape)
        # For simplicity in mock, we assume 1-to-1 mapping for now
        y = pd.Series(y[:len(X)]) 

        # 2. Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 3. Train
        logging.info("Training XGBoost Model...")
        self.model.fit(X_train, y_train)

        # 4. Evaluate
        preds = self.model.predict(X_test)
        auc = roc_auc_score(y_test, self.model.predict_proba(X_test)[:, 1])
        logging.info(f"Model Training Complete. AUC: {auc:.4f}")
        logging.info(f"\n{classification_report(y_test, preds)}")

        # 5. Save Artifacts
        joblib.dump(self.model, 'src/models/churn_model.pkl')
        logging.info("Model saved to src/models/churn_model.pkl")

if __name__ == "__main__":
    trainer = ChurnTrainer()
    trainer.train()