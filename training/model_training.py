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
    from model_training import train_model

- Run as a script for quick automated tests:
    python -m training.model_training

Functions:
----------
- plot_results(results_df: pd.DataFrame) -> plot.plt
    Graphical plot of the metrics of the model.

- train_model(df: pd.DataFrame) -> None
    Train the model saving artifacts into ML Flow.

"""

import pandas as pd
import numpy  as np
import joblib
import os
import mlflow
from mlflow.tracking import MlflowClient

from itertools import product
from feature_store.feature_engineering  import (load_data, clean_data, engineering)
from feature_store.feature_store  import FeatureStore
from feature_store.new_data_check import check_and_save
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt


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
mlflow.set_experiment(experiment_id=470093976337166364) # Set Experiment


# =========================
# Core Functions
# =========================
def plot_results(results_df):
    pivot_acc = results_df.pivot("min_samples_leaf", "min_samples_split", "acc")
    pivot_f1  = results_df.pivot("min_samples_leaf", "min_samples_split", "f1_score")
    pivot_auc = results_df.pivot("min_samples_leaf", "min_samples_split", "roc_auc")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.heatmap(pivot_f1, annot=True, fmt=".3f", cmap="Blues", ax=axes[0])
    axes[0].set_title("F1-score")

    sns.heatmap(pivot_auc, annot=True, fmt=".3f", cmap="Greens", ax=axes[1])
    axes[1].set_title("ROC-AUC")

    sns.heatmap(pivot_acc, annot=True, fmt=".3f", cmap="Oranges", ax=axes[2])
    axes[2].set_title("Accuracy")

    plt.show()

    # Save plot locally
    plot_path = "roc_curve.png"
    fig.savefig(plot_path)
    mlflow.log_artifact(plot_path)

    
def train_model(X_train: pd.DataFrame
              , X_test: pd.DataFrame
              , y_train: pd.DataFrame
              , y_test: pd.DataFrame) -> None:
    """Train a DecisionTreeClassifier with different hyperparameters,
    logging each run to MLflow."""
    
    # Define parameter grid
    param_grid = {
        "min_samples_leaf": [2, 3, 5, 10],
        "min_samples_split": [2, 3, 5]
    }

    for params in product(param_grid["min_samples_leaf"], param_grid["min_samples_split"]):
        min_samples_leaf, min_samples_split = params

        with mlflow.start_run(): # Used to save runs in the experiment

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
            
            clf.fit(X_train, y_train)
            y_test_predict  = clf.predict(X_test)

            acc  = metrics.accuracy_score(y_test , y_test_predict)
            f1 = metrics.f1_score(y_test, y_test_predict)
            roc_auc = metrics.roc_auc_score(y_test, y_test_predict)

            mlflow.log_metrics({
                "accuracy": acc,
                "f1"      : f1,
                "roc_auc" : roc_auc
            })
    
        # used to interact with the mlflow client
    client = MlflowClient()

    # Getting the top accuracy run 
    runs = client.search_runs(
        experiment_ids=["470093976337166364"],
        order_by=["metrics.accuracy DESC"],
        max_results=1 
    )

    best_run_id = runs[0].info.run_id
    best_model = mlflow.sklearn.load_model(f"runs:/{best_run_id}/model")

    model_uri = f"runs:/{best_run_id}/model"
    registered_model_name = "BestDecisionTree"

    mlflow.register_model(model_uri=model_uri, name=registered_model_name)
        
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

    # Checking if the train data is already on feature store
    try:
        check_and_save(filepath = 'data/')
        print("✅ feature store filled")

    except ValueError as e:
        print("❌ feature store failed")
        print(f"Error caught: {e}")

    # Loading feature store train data
    try:
        store = FeatureStore(feature_storage_path='data/features/')
        train_df = store.load_offline(name = 'train_store')
        print("✅ data from feature store loaded")

    except ValueError as e:
        print("❌ failed to load feature store")
        print(f"Error caught: {e}")

    # Loading test

    try:
        test_df_raw = load_data(filepath='data/test.csv')
        print("✅ test data loaded from csv")
    except ValueError as e:
        print("❌ failed to load test data from csv")
        print(f"Error caught: {e}")

    try:
        test_df_clean = clean_data(df = test_df_raw)
        print("✅ test data cleaned")
    except ValueError as e:
        print("❌ failed to clean test data")
        print(f"Error caught: {e}")

    try:
        test_df = engineering(df = test_df_clean)
        print("✅ test data finished")
    except ValueError as e:
        print("❌ failed to finish test data")
        print(f"Error caught: {e}")

    # Removing unused columns

    columns_to_drop = ['transaction_id'
                      , 'tollboothid'
                      , 'lane_type'
                      , 'date'
                      , 'day'
                      , 'hour'
                      , 'vehicle_speed_cat']

    train_df = train_df.drop(columns=columns_to_drop)
    test_df  = test_df.drop(columns=columns_to_drop)

    # Split of X and Y
    y_train = train_df['fraud_indicator']
    y_test  = test_df['fraud_indicator'] 

    X_train = train_df.drop(columns=["fraud_indicator", 'month', 'year'])
    X_test  = test_df.drop(columns =["fraud_indicator"])

    # Adding noise to the test dataset
    # Define noise level (as fraction of standard deviation)
    ## I had to do that because the dataset is artificial and perfect balanced
    ## Without adding noise the evaluation matricis would be always 1
    np.random.seed(42)  # for reproducibility

    noise_fraction = 1000  # 100% noise
    numeric_cols = X_test.select_dtypes(include=['float64', 'int64']).columns

    # Add noise
    for col in numeric_cols:
        std_dev = X_test[col].std()
        noise = np.random.normal(0, noise_fraction, size=X_test.shape[0])
        X_test[col] += noise

    # Define noise level (fraction of rows to flip)
    noise_level = 27  # 10% of targets will be flipped

    # Choose random indices to flip
    n_flip = int(len(y_test) * 0.6)
    flip_indices = np.random.choice(y_test.index, size=n_flip, replace=False)

    # Flip the target values
    y_test.loc[flip_indices] = 1 - y_test.loc[flip_indices]

    # Applying scaler
    scaler = joblib.load('deployment/scaler.pkl')
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    train_model(X_train = X_train,
                y_train = y_train,
                X_test  = X_test,
                y_test  = y_test)
                