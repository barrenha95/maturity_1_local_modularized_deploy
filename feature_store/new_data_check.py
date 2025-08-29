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
from datetime import datetime

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
    if os.path.exists(filepath + "features/"):
        print("Feature Store already exists.")
        pass
    else:
        if os.path.exists(filepath + "train.csv"):
            store = FeatureStore(filepath + 'features/')
            df = apply_transformations(filepath + "train.csv")
            store.save_offline(df, name = 'train_store')
            print("Feature Store created.")
        else:
            print("There are no train file.")

    # Checking if exists new data

    if os.path.exists(filepath + 'new.csv'):
        print("New file exists")
        store = FeatureStore(filepath + 'features/')
        new_df = apply_transformations(filepath + 'new.csv')
        store.save_offline(new_df, name = 'train_store')
        os.remove(filepath + 'new.csv') # remove new data
    else:
        print("There are no new files")
    
# =========================
# Standalone Script (Testing)
# =========================

"""
Functions:
----------
- Run when feature store doesn't exist
- Run when feature store exists
- Ruen when there are no files
"""
def test_fs_non_existance():
    filepath_test = 'data/auto_test/'
    check_and_save(filepath=filepath_test)

    if not os.path.exists(filepath_test):
        print("✅ non existance test passed")

def test_only_train():
    filepath_test = 'data/auto_test/'

    df = pd.DataFrame({
        "Transaction_ID"       : [1],
        "Timestamp"            : [datetime.now()],
        "Vehicle_Type"         : ["Bus"],
        "FastagID"             : ["FTG-001-ABC-121"],
        "TollBoothID"          : ["A-101"],
        "Lane_Type"            : ["Express"],
        "Vehicle_Dimensions"   : ["Large"],
        "Transaction_Amount"   : ["350"],
        "Amount_paid"          : ["120"],
        "Geographical_Location": ["13.059816123454882, 77.77068662374292"],
        "Vehicle_Speed"        : ["65"],
        "Vehicle_Plate_Number" : ["KA11AB1234"],
        "Fraud_indicator"      : ["Fraud"]
    })

    if os.path.exists(filepath_test):
        shutil.rmtree(filepath_test) # removing file

    os.mkdir(filepath_test)
    df.to_csv(filepath_test + 'train.csv')

    try:
        check_and_save(filepath_test)
    except ValueError as e:
        print("❌ test_only_train failed")
        print(f"Error caught, in check_and_save: {e}")

    if os.path.exists(filepath_test + 'train.csv'):
        
        store = FeatureStore(filepath_test + "features/")
        train_df = store.load_offline(name = 'train_store')

        shutil.rmtree(filepath_test) # removing file
        assert len(train_df) == 1
        assert train_df['transaction_id'].tolist() == [1]
        print("✅ test_only_train passed")
    
    else:
        print("❌ test_only_train failed")
        print("There are no files on train feature store.")

def test_train_and_new():
    filepath_test = 'data/auto_test/'

    df = pd.DataFrame({
        "Transaction_ID"       : [1],
        "Timestamp"            : [datetime.now()],
        "Vehicle_Type"         : ["Bus"],
        "FastagID"             : ["FTG-001-ABC-121"],
        "TollBoothID"          : ["A-101"],
        "Lane_Type"            : ["Express"],
        "Vehicle_Dimensions"   : ["Large"],
        "Transaction_Amount"   : ["350"],
        "Amount_paid"          : ["120"],
        "Geographical_Location": ["13.059816123454882, 77.77068662374292"],
        "Vehicle_Speed"        : ["65"],
        "Vehicle_Plate_Number" : ["KA11AB1234"],
        "Fraud_indicator"      : ["Fraud"]
    })

    df2 = pd.DataFrame({
        "Transaction_ID"       : [2],
        "Timestamp"            : [datetime.now()],
        "Vehicle_Type"         : ["Bus"],
        "FastagID"             : ["FTG-001-ABC-121"],
        "TollBoothID"          : ["A-101"],
        "Lane_Type"            : ["Express"],
        "Vehicle_Dimensions"   : ["Large"],
        "Transaction_Amount"   : ["350"],
        "Amount_paid"          : ["120"],
        "Geographical_Location": ["13.059816123454882, 77.77068662374292"],
        "Vehicle_Speed"        : ["65"],
        "Vehicle_Plate_Number" : ["KA11AB1234"],
        "Fraud_indicator"      : ["Fraud"]
    })


    if os.path.exists(filepath_test):
        shutil.rmtree(filepath_test) # removing file

    os.mkdir(filepath_test)
    df.to_csv(filepath_test + 'train.csv')
    df2.to_csv(filepath_test + 'new.csv')

    try:
        check_and_save(filepath_test)
    except ValueError as e:
        print("❌ test_train_and_new failed")
        print(f"Error caught, in check_and_save: {e}")

    if os.path.exists(filepath_test + 'train.csv'):
        
        store = FeatureStore(filepath_test + "features/")
        train_df = store.load_offline(name = 'train_store')

        shutil.rmtree(filepath_test) # removing file
        assert len(train_df) == 2
        assert train_df['transaction_id'].tolist() == [1,2]
        print("✅ test_train_and_new passed")
    
    else:
        print("❌ test_train_and_new failed")
        print("There are no files feature store.")

if __name__ == "__main__":
    test_fs_non_existance()

    test_only_train()
    test_train_and_new()