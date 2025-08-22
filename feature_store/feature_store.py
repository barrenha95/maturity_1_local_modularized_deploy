"""
====================================
 Feature Store Module
====================================
Author: João
Date: 2025-08-20

Description:
------------
This module maintains the feature store of the project.
A feature store is a centralized system to store, manage, and serve features for ML models.

Why to use it? 
A) Keep features in a reliable, reusable way.
b) Provide those features quickly:
- Offline store: for training (big batches, historical data).
- Online store: for real-time inference (low-latency lookups).

Usage:
------
- Import as a module:
    from feature_store.feature_store import save_offline, load_offline

- Run as a script for quick automated tests:
    python feature_store.py

Functions:
----------
- save_offline(filepath: str) -> None
    Save the features into the offline feature store (a parquet file).

- load_offline(filepath: str) -> None
    Load the features from the offline feature store (a parquet file).

"""

import pandas as pd
import os
from pathlib import Path
import numpy  as np

# =========================
# Core Functions
# =========================
class FeatureStore:

    # Constructor
    def __init__(self, feature_storage_path='data/features/'):
        self.feature_storage_path = feature_storage_path
        os.makedirs(self.feature_storage_path, exist_ok=True)
    
    # Offline feature store
    def save_offline(self, df: pd.DataFrame, name: str):
        """Save features to offline store, adding new rows (parquet)."""
        file_path = os.path.join(self.feature_storage_path, f"{name}.parquet")

        if os.path.exists(file_path):
            # load the old offline feature store
            old_df = pd.read_parquet(file_path)

            # bind new rows
            df = pd.concat([old_df, df], ignore_index=True)

            # remove duplicates
            df = df.drop_duplicates()
        
        # save with the new data
        df.to_parquet(file_path, index=False)

    def load_offline(self, name):
        """Load features from offline store (for training)."""
        file_path = os.path.join(self.feature_storage_path, f"{name}.parquet")

        if os.path.exists(file_path):
            return pd.read_parquet(file_path)
        else:
            raise FileNotFoundError(f"No dataset found for {name}")


# =========================
# Standalone Script (Testing)
# =========================

"""
Functions:
----------
- Standard record, with good case of fillement in all columns
- A record filled only with integer number for all columns
- A record filled with None for all columns
"""

def test_save_and_load():
    path = "data/features/test_store.parquet"
    if os.path.exists(path):
        os.remove(path)

    store = FeatureStore()
    df1 = pd.DataFrame({
        "Transaction_ID"       : [1, 2, 3],
        "Timestamp"            : ["2023-01-06 11:20:00	", 1, None],
        "Vehicle_Type"         : ["Bus", 1, None],
        "FastagID"             : ["FTG-001-ABC-121	", 1, None],
        "TollBoothID"          : ["A-101	", 1, None],
        "Lane_Type"            : ["Express	", 1, None],
        "Vehicle_Dimensions"   : ["Large	", 1, None],
        "Transaction_Amount"   : ["350", 1, None],
        "Amount_paid"          : ["120	", 1, None],
        "Geographical_Location": ["13.059816123454882, 77.77068662374292	", 1, None],
        "Vehicle_Speed"        : ["65	", 1, None],
        "Vehicle_Plate_Number" : ["KA11AB1234	", 1, None],
        "Fraud_indicator"      : ["Fraud", 1, None]
    })

    try:
        store.save_offline(df1, name = 'test_store')
    except ValueError as e:
        print("❌ test_save_offline failed")
        print(f"Error caught: {e}")

    try:
        loaded = store.load_offline(name = 'test_store')
    except ValueError as e:
        print("❌ test_load_offline failed")
        print(f"Error caught: {e}")

    assert len(loaded) == 3
    assert "Transaction_ID" in loaded.columns
    print("✅ test_save_and_load passed")


def test_append_data():
    path = "data/features/test_store.parquet"
    if os.path.exists(path):
        os.remove(path)

    store = FeatureStore(path)
    df1 = pd.DataFrame({
        "Transaction_ID"       : [1, 2],
        "Timestamp"            : ["2023-01-06 11:20:00	", 1],
        "Vehicle_Type"         : ["Bus", 1],
        "FastagID"             : ["FTG-001-ABC-121	", 1],
        "TollBoothID"          : ["A-101	", 1],
        "Lane_Type"            : ["Express	", 1],
        "Vehicle_Dimensions"   : ["Large	", 1],
        "Transaction_Amount"   : ["350", 1],
        "Amount_paid"          : ["120	", 1],
        "Geographical_Location": ["13.059816123454882, 77.77068662374292	", 1],
        "Vehicle_Speed"        : ["65	", 1],
        "Vehicle_Plate_Number" : ["KA11AB1234	", 1],
        "Fraud_indicator"      : ["Fraud", 1]
    })
    store.save_offline(df1)

    df2 = pd.DataFrame({
        "Transaction_ID"       : [3],
        "Timestamp"            : [None],
        "Vehicle_Type"         : [None],
        "FastagID"             : [None],
        "TollBoothID"          : [None],
        "Lane_Type"            : [None],
        "Vehicle_Dimensions"   : [None],
        "Transaction_Amount"   : [None],
        "Amount_paid"          : [None],
        "Geographical_Location": [None],
        "Vehicle_Speed"        : [None],
        "Vehicle_Plate_Number" : [None],
        "Fraud_indicator"      : [None]
    })
    store.save_offline(df2)

    loaded = store.load_offline()
    assert len(loaded) == 3
    assert loaded["Transaction_ID"].tolist() == [1, 2, 3]
    print("✅ test_append_data passed")


def test_load_nonexistent():
    path = "data/features/nonexistent.parquet"
    store = FeatureStore(path)
    try:
        store.load_offline()
    except FileNotFoundError:
        print("✅ test_load_nonexistent passed")
        return
    raise AssertionError("❌ test_load_nonexistent failed")


if __name__ == "__main__":
    print("\nTesting: Save and Load offline feature store:")
    test_save_and_load()

    print("\nTesting: append data offline feature store:")
    test_append_data()
    
    print("\nTesting: Load nonexistent offline featurestore:")
    test_load_nonexistent()
    
    print("\n✅ All tests passed for feature_store.py")