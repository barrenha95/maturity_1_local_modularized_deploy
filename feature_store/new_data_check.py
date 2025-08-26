"""
====================================
 New data check Module
====================================
Author: João
Date: 2025-08-26

Description:
------------
This module will check if exists new data on the repository, preparing the feature store. 

Usage:
------
- Import as a module:
    from feature_store.new_data_check import check_data, apply_transformations, save_features

- Run as a script for quick automated tests:
    python new_data_check.py

Functions:
----------
- check_data(filepath: str) -> 
    Check if exists new_data on the repository.

- apply_transformations(df: pd.DataFrame) -> pd.DataFrame
    Apply all the transformations of feature_engineering.py and feature_store.py.

- save_features(df: pd.DataFrame) -> 
    Save what will be called as train dataframe

"""

import pandas as pd
import numpy  as np
import os
from feature_store.feature_engineering  import (load_data, clean_data, engineering)
from feature_store.feature_store        import (save_offline, load_offline)


# =========================
# Core Functions
# =========================
def check_data(filepath = 'data/'):
    """Check if exists new data."""

    fs_path = filepath + "features/"
    
    # Check if the directory exists before attempting to remove it (optional, but good practice)
    if os.path.exists(fs_path) and os.path.isdir(fs_path):
        pass

    else:
        store = FeatureStore(fs_path)

    if os.path.exists(filepath):
        try:
            df = load_data(filepath)
            print(f"Directory '{filepath}' exists.")
        except OSError as e:
            print(f"Error: {filepath} : {e.strerror}")
    else:
            print(f"Directory '{filepath}' not exists.")

    




def apply_transformations(df: pd.DataFrame) -> pd.DataFrame:

    #df = load_data(file_path + 'train.csv')
    #df = clean_data(df)
    #df = engineering(df)

    print(df)
    

def save_features(df: pd.DataFrame):
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

    check_data()