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
import shutil
from pathlib import Path
from datetime import datetime
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
            bind_df = pd.concat([old_df, df], ignore_index=True)

            # remove duplicates
            bind_df = bind_df.drop_duplicates()
            
            # remove old files
            shutil.rmtree(file_path)

            # save with the new data
            bind_df.to_parquet(file_path, partition_cols=['month','year'])

        else:
            df.to_parquet(file_path, partition_cols=['month','year'])

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
    path = "data/features/"
    
    # Check if the directory exists before attempting to remove it (optional, but good practice)
    if os.path.exists(path) and os.path.isdir(path):
        try:
            shutil.rmtree(path)
            print(f"Directory '{path}' and its contents removed successfully.")
        except OSError as e:
            print(f"Error: {path} : {e.strerror}")


    store = FeatureStore()
    df1 = pd.DataFrame({
        "transaction_id"        : [1],             # drop
        "vehicle_type"          : [2.0],           
        "fastagid"              : [1],             
        "tollboothid"           : [1.0],           # drop
        "lane_type"             : [0.0],           # drop
        "vehicle_dimensions"    : [3.0],           
        "fraud_indicator"       : ['fraud'],       # drop
        "date"                  : ['2025-08-25'],  # drop
        "day"                   : ['25'],          # drop
        "hour"                  : ['20'],          # drop
        "transaction_amount_cat": [2],
        "discount"              : [1],
        "no_change"             : [0],
        "penalty"               : [0],
        "vehicle_speed_cat"     : [1],             # drop
        "month"                 : [1],             # drop
        "year"                  : [2023]           # drop 
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

    assert len(loaded) == 1
    assert "transaction_id" in loaded.columns
    print("✅ test_save_and_load passed")


def test_append_data():
    path = "data/features/"
    
    # Check if the directory exists before attempting to remove it (optional, but good practice)
    if os.path.exists(path) and os.path.isdir(path):
        try:
            shutil.rmtree(path)
            print(f"Directory '{path}' and its contents removed successfully.")
        except OSError as e:
            print(f"Error: {path} : {e.strerror}")
    
    store = FeatureStore(path)
    
    df1 = pd.DataFrame({
        "transaction_id"        : [1],
        "vehicle_type"          : [2.0],
        "fastagid"              : [1],
        "tollboothid"           : [1.0],
        "lane_type"             : [0.0],
        "vehicle_dimensions"    : [3.0],
        "fraud_indicator"       : ['fraud'],
        "date"                  : ['2025-08-25'],
        "day"                   : ['25'],
        "hour"                  : ['20'],
        "transaction_amount_cat": [2],
        "discount"              : [1],
        "no_change"             : [0],
        "penalty"               : [0],
        "vehicle_speed_cat"     : [1],
        "month"                 : [1],
        "year"                  : [2023]
    })

    store.save_offline(df1, name = 'test_store')

    df2 = pd.DataFrame({
        "transaction_id"        : [2, 3],
        "vehicle_type"          : [3.0, 9.0],
        "fastagid"              : [4, 5],
        "tollboothid"           : [5.0, 6.0],
        "lane_type"             : [6.0, 10.0],
        "vehicle_dimensions"    : [7.0, 14.0],
        "fraud_indicator"       : ['fraud', 'fraud'],
        "date"                  : ['2025-08-25', '2025-08-25'],
        "day"                   : ['25', '26'],
        "hour"                  : ['20', '19'],
        "transaction_amount_cat": [8, 1],
        "discount"              : [9, 1],
        "no_change"             : [0, 1],
        "penalty"               : [0, 1],
        "vehicle_speed_cat"     : [1, 1],
        "month"                 : [1, 1],
        "year"                  : [2023, 2023]
    })
    store.save_offline(df2, name = 'test_store')

    loaded = store.load_offline(name = 'test_store')
     
    assert len(loaded) == 3
    assert loaded["transaction_id"].tolist() == [1, 2, 3]
    print("✅ test_append_data passed")


def test_load_nonexistent():
    path = "data/features/"
    store = FeatureStore(path)
    try:
        store.load_offline(name = 'nonexistent')
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

    path = "data/features/"
    
    # Check if the directory exists before attempting to remove it (optional, but good practice)
    if os.path.exists(path) and os.path.isdir(path):
        try:
            shutil.rmtree(path)
            print(f"Directory '{path}' and its contents removed successfully.")
        except OSError as e:
            print(f"Error: {path} : {e.strerror}")