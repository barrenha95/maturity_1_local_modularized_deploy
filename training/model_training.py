"""
====================================
 Model Training Module
====================================
Author: João
Date: 2025-08-20

Description:
------------
This module handles the process of training the model . 

Usage:
------
- Import as a module:
    from load_data import load_data, clean_data

- Run as a script for quick automated tests:
    python load_data.py

Functions:
----------
- load_feature_store(filepath: str) -> pd.DataFrame
    Loads data from a CSV file.

- train_model(df: pd.DataFrame) -> pd.DataFrame
    Cleans missing values, duplicates, and basic formatting.

"""

import pandas as pd
import numpy  as np
import joblib
from feature_store.feature_engineering import load_data, clean_data, engineering
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier



# =========================
# Core Functions
# =========================
def load_feature_store_data(filepath: str) -> pd.DataFrame:
    """Load data from a CSV file."""


def train_model(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw data (remove duplicates, handle missing values)."""
    
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

if __name__ == "__main__":
    print("Running quick self-test for load_data.py...")

    # Create a dummy DataFrame for testing
    test_df = pd.DataFrame({
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

    print("\nOriginal test dataframe:")
    print(test_df)

    cleaned = clean_data(test_df)

    print("\nCleaned test dataframe:")
    print(cleaned)

    # Check basic conditions
    assert cleaned.isna().sum().sum() == 0, "❌ Missing values were not handled!"
    assert cleaned.duplicated().sum() == 0, "❌ Duplicates were not removed!"
    print("\n✅ All tests passed for load_data.py!")