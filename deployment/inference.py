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
from sklearn import tree

mlflow.set_tracking_uri('http://localhost:5000') # Set which mlflow server will use

# Loading the best model
def model_load():

    loaded_model = mlflow.sklearn.load_model(f"models:/BestDecisionTree/Production")
    return loaded_model

# Make the prediction
def model_prediction(model, features):

    # Applying scaler
    scaler = joblib.load('deployment/scaler.pkl')
    scaled_features = scaler.transform(features)

    predictions = model.predict(scaled_features)
    return predictions

     
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

    try:
        best_model = model_load()
        print("✅ best model loaded")
    except ValueError as e:
        print("❌ failed load model")
        print(f"Error caught: {e}")

    test_features =  pd.DataFrame({
        "vehicle_type"           : [7],
        "fastagid"               : [1],
        "vehicle_dimensions"     : [1],
        "transaction_amount_cat" : [0],
        "discount"               : [0],
        "no_change"              : [1],
        "penalty"                : [0]    
        })

    try:
        test_predictions = model_prediction(model = best_model, features= test_features)
        print("✅ predictions made")
    except ValueError as e:
        print("❌ failed make predictions")
        print(f"Error caught: {e}")

    print()
    print("The testd preidction is: " + str(test_predictions))

