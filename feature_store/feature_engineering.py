"""
====================================
 Data Loading & Cleaning Module
====================================
Author: João
Date: 2025-08-20

Description:
------------
This module handles the process of loading and cleaning raw data. 

Usage:
------
- Import as a module:
    from feature_store.feature_engineering import load_data, clean_data, engineering

- Run as a script for quick automated tests:
    python feature_engineering.py

Functions:
----------
- load_data(filepath: str) -> pd.DataFrame
    Loads data from a CSV file.

- clean_data(df: pd.DataFrame) -> pd.DataFrame
    Cleans missing values, duplicates, and basic formatting.

- engineering(df: pd.DataFrame) -> pd.DataFrame
    Apply the data wrangling operations selected for this project.

"""

import pandas as pd
import numpy  as np
from project_encodings  import (VEHICLE_ENCODING, TOLLBOOTH_ENCODING, LANE_ENCODING, DIMENSIONS_ENCODING)

# =========================
# Core Functions
# =========================
def load_data(filepath: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(filepath, parse_dates=['Timestamp'])
        return df
    except Exception as e:
        raise IOError(f"Error loading file {filepath}: {e}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw data (remove duplicates, handle missing values)."""
    df = df.drop_duplicates()
    df = df.dropna()  # Simplest strategy: drop missing
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def engineering(df: pd.DataFrame) -> pd.DataFrame:
    """"Apply the data wrangling operations"""

    # encoding textual columns
    if "Vehicle_Type" in df.columns:
        df["Vehicle_Type"] = df["Vehicle_Type"].map(VEHICLE_ENCODING)
    
    if "TollBoothID" in df.columns:
        df["TollBoothID"] = df["TollBoothID"].map(TOLLBOOTH_ENCODING)

    if "Lane_Type" in df.columns:
        df["Lane_Type"] = df["Lane_Type"].map(LANE_ENCODING)

    if "Vehicle_Dimensions" in df.columns:
        df["Vehicle_Dimensions"] = df["Vehicle_Dimensions"].map(DIMENSIONS_ENCODING) 

    # splitting timestamp components, it can be used as feature or to split as timeseries
    df['Date'] = df['Timestamp'].dt.date
    df['Day']  = df['Timestamp'].dt.day
    df['Hour'] = df['Timestamp'].dt.hour           

    # 1 = has fastag 0 = don't have
    df['FastagID'] = np.where(df['FastagID'].isna(), 0, 1)

    # split transaction amount into cuts that makes sense to the problem
    transaction_amount_bins = [-999999, 115, 230, 350, 99999]
    transaction_amount_labels = ['0', '1', '2', '3']

    df['Transaction_Amount_cat'] = pd.cut(df['Transaction_Amount']
                                          , bins   = transaction_amount_bins
                                          , labels = transaction_amount_labels
                                          , right  = True)
    
    # creating new features using the interaction of some columns
    df['Diff_paid'] = df['Transaction_Amount'] - df['Amount_paid']
    df['Discount']  = np.where(df['Diff_paid']> 0, 1, 0)
    df['No_change'] = np.where(df['Diff_paid'] == 0, 1, 0)
    df['Penalty']   = np.where(df['Diff_paid'] < 0, 1, 0)

    # split speed into cuts that makes sense to the problem
    speed_bins = [-999999, 50, 80, 120, 99999]
    speed_labels = ['0', '1', '2', '3']
    df['Vehicle_Speed_cat'] = pd.cut(df['Vehicle_Speed']
                                     , bins   = speed_bins
                                     , labels = speed_labels
                                     , right  = True)

    df = df.drop(['Vehicle_Plate_Number'
                , 'Geographical_Location'
                , 'Timestamp'
                , 'Diff_paid'
                , 'Transaction_Amount'
                , 'Amount_paid'
                , 'Vehicle_Speed'], axis=1)
    
    return df

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
    print("Running quick self-test for feature_engineering.py...")

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

    engineered = engineering(cleaned)
    print("\nEngineered test dataframe:")
    print(engineered)

    # Check basic conditions
    assert cleaned.isna().sum().sum() == 0, "❌ Missing values were not handled!"
    assert cleaned.duplicated().sum() == 0, "❌ Duplicates were not removed!"
    print("\n✅ All tests passed for load_data.py!")