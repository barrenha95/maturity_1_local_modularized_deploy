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
- check_and_save(filepath: str) -> 
    Check if exists new_data on the repository.

- apply_transformations(df: pd.DataFrame) -> pd.DataFrame
    Apply all the transformations of feature_engineering.py and feature_store.py.

"""

import pandas as pd
import numpy  as np
import shutil

import os
from feature_store.feature_store import FeatureStore
from feature_store.feature_engineering  import (load_data, clean_data, engineering)


# =========================
# Core Functions
# =========================

def apply_transformations(filepath: str) -> pd.DataFrame:
    """Apply feature engineering transformations."""

    df = load_data(filepath)
    df = clean_data(df)
    df = engineering(df)

    # creating the partition cols
    df['month']  = pd.to_datetime(df['date']).dt.month
    df['year']   = pd.to_datetime(df['date']).dt.year

    return df
    
def check_and_save(filepath = 'data/'):
    """Check if exists new data."""

    # Checking if exists feature store
    fs_path = filepath + "features/"
    train_path = filepath + "train.csv"
    new_path = filepath + 'new.csv'
    
    if os.path.exists(fs_path):
        pass
    else:
        store = FeatureStore(fs_path)
        
        df = apply_transformations(train_path)

        store.save_offline(df, name = 'train_store')

    # Checking if exists new data

    if os.path.exists(new_path):
        print("New file exists")

        store = FeatureStore(fs_path)

        new_df = apply_transformations(new_path)

        store.save_offline(new_df, name = 'train_store')
        
        os.remove(new_path) # remove new data

    else:
        print("There are no new files")
    
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

    check_and_save()