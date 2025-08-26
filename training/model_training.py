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
import os
from feature_store.feature_engineering  import (load_data, clean_data, engineering)
from feature_store.feature_store  import (save_offline, load_offline)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier



# =========================
# Core Functions
# =========================
def load_feature_store_data(filepath: str) -> pd.DataFrame:
    """Load data from a parquet file."""

    df = load_data(filepath + 'train.csv')
    df = clean_data(df)
    df = engineering(df)

    print(df)
    


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

    df = load_feature_store_data('data/')

    print(df)