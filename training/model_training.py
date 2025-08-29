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
import mlflow
from itertools import product
from feature_store.feature_engineering  import (load_data, clean_data, engineering)
from feature_store.feature_store  import (save_offline, load_offline)
from feature_store.new_data_check import check_and_save
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import GridSearchCV

# =========================
# ML Flow
# =========================
# Used to store artifcts of the model training
# It contains :
# - mlflow server application (that runs as an server, and store the metadata)
# - mlflow library used to interact with the mlflow server
# To run this script you must open an new terminal and run: mlflow server
# The best way to run the mlflow server is on another machine like an vm or even docker
mlflow.set_tracking_uri('http://localhost:5000') # Set which mlflow server will use
mlflow.set_experiment(experiment_id=620958864307953100) # Set Experiment


# =========================
# Core Functions
# =========================
def load_feature_store_data(filepath: str) -> pd.DataFrame:
    """Load data from a parquet file."""

    df = load_data(filepath + 'train.csv')
    df = clean_data(df)
    df = engineering(df)

    print(df)
    

def train_model(X_train: pd.DataFrame
              , X_test: pd.DataFrame
              , y_train: pd.DataFrame
              , y_test: pd.DataFrame) -> None:
    """Train a DecisionTreeClassifier with different hyperparameters,
    logging each run to MLflow."""

    
    # Define parameter grid
    param_grid = {
        "min_samples_leaf": [10, 20, 30, 50, 100],
        "min_samples_split": [2, 5, 10]
    }

    for params in product(param_grid["min_samples_leaf"], param_grid["min_samples_split"]):
        min_samples_leaf, min_samples_split = params

        with mlflow.start_run: # Used to save runs in the experiment

            mlflow.sklearn.autolog() # Tell mlflow to generate some logs of the run

            # Log parameters manually
            mlflow.log_params({
                "min_samples_leaf": min_samples_leaf,
                "min_samples_split": min_samples_split
            })

            # Define and fit model
            clf = tree.DecisionTreeClassifier(
                min_samples_leaf=min_samples_leaf,
                min_samples_split=min_samples_split,
                random_state=42
            )
            
            clf.fit = (X_train, y_train)
            y_train_predict = clf.predict(X_train)
            y_test_predict  = clf.predict(X_test)


            acc_train = metrics.accuracy_score(y_train, y_train_predict)
            acc_test  = metrics.accuracy_score(y_test , y_test_predict)

            mlflow.log_metrics({
                "acc_train": acc_train,
                "acc_test": acc_test
            })

        
# =================
# Standalone Script
# =================

"""
Functions:
----------
- Call other libs to read, clean, and create the feature store 
- Train the model using Ml Flow library to store model artifacts
- Train an decision tree, testing multiple parameters
"""

if __name__ == "__main__":

    df = load_feature_store_data('data/')

    print(df)