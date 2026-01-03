import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FeatureEngineer:
    """
    Transforms raw retail transaction logs into customer-level RFM features.
    """
    
    def __init__(self):
        pass

    def generate_rfm_features(self, transactions_df: pd.DataFrame, snapshot_date: datetime = None) -> pd.DataFrame:
        """
        Calculates Recency, Frequency, and Monetary value for each customer.
        
        Args:
            transactions_df: DataFrame with columns ['customer_id', 'transaction_date', 'amount', 'product_id']
            snapshot_date: The cutoff date for calculation (simulating 'today').
            
        Returns:
            DataFrame with customer_id as index and RFM features.
        """
        if snapshot_date is None:
            snapshot_date = transactions_df['transaction_date'].max() + timedelta(days=1)
            
        logging.info(f"Generating features based on snapshot date: {snapshot_date}")

        # Ensure date format
        transactions_df['transaction_date'] = pd.to_datetime(transactions_df['transaction_date'])

        # Aggregate data
        rfm = transactions_df.groupby('customer_id').agg({
            'transaction_date': lambda x: (snapshot_date - x.max()).days,  # Recency
            'product_id': 'count',                                         # Frequency (Items bought)
            'amount': 'sum'                                                # Monetary
        }).rename(columns={
            'transaction_date': 'recency_days',
            'product_id': 'frequency',
            'amount': 'monetary_value'
        })
        
        # Feature Engineering: Average Order Value (AOV)
        rfm['aov'] = rfm['monetary_value'] / rfm['frequency']
        
        # Handle outliers/NaNs
        rfm.fillna(0, inplace=True)
        
        logging.info(f"Features generated for {len(rfm)} customers.")
        return rfm

if __name__ == "__main__":
    # Mock data for testing logic directly
    data = {
        'customer_id': [101, 102, 101, 103, 102],
        'transaction_date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-05', '2023-01-10', '2023-01-10']),
        'amount': [50, 20, 100, 200, 40],
        'product_id': ['A', 'B', 'A', 'C', 'B']
    }
    df = pd.DataFrame(data)
    fe = FeatureEngineer()
    print(fe.generate_rfm_features(df))