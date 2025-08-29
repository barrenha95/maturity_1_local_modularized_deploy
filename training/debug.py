# %%
full_path = "/home/isabarrenha/Documents/Portfolio/maturity_1_local_modularized_deploy/"

import sys
sys.path.append(full_path)
from feature_store.feature_engineering  import (load_data, clean_data, engineering)
from feature_store.feature_store  import FeatureStore
from feature_store.new_data_check import check_and_save

import pandas as pd
import numpy  as np
import joblib
import os
import mlflow
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt



# %%        
# Loading feature store train data
try:
    store = FeatureStore(feature_storage_path=full_path + 'data/features/')
    train_df = store.load_offline(name = 'train_store')
    print("✅ data from feature store loaded")

except ValueError as e:
    print("❌ failed to load feature store")
    print(f"Error caught: {e}")

print(train_df)

# %%
# Loading test
try:
    test_df_raw = load_data(filepath=full_path + 'data/test.csv')
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

print(test_df)


# %%
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

print(train_df)
print(test_df)


# %%
# Split of X and Y
y_train = train_df['fraud_indicator']
y_test  = test_df['fraud_indicator'] 

X_train = train_df.drop(columns=["fraud_indicator", 'month', 'year'])
X_test  = test_df.drop(columns =["fraud_indicator"])

#print(train_df)
#print(test_df)

# %%
# Adding noise to the test dataset
# Define noise level (as fraction of standard deviation)
noise_fraction = 1  # 5% noise
numeric_cols = X_test.select_dtypes(include=['float64', 'int64']).columns

# Add noise
for col in numeric_cols:
    std_dev = X_test[col].std()
    noise = np.random.normal(0, std_dev * noise_fraction, size=X_test.shape[0])
    X_test[col] += noise

# %%
# Define and fit model
clf = tree.DecisionTreeClassifier(
    min_samples_leaf=10,
    min_samples_split=10,
    random_state=42)

clf.fit(X_train, y_train)
y_test_predict  = clf.predict(X_test)

acc  = metrics.accuracy_score(y_test , y_test_predict)
f1 = metrics.f1_score(y_test, y_test_predict)
roc_auc = metrics.roc_auc_score(y_test, y_test_predict)

print("acc: " + str(acc))
print("f1: " + str(f1))
print("roc_auc: " + str(roc_auc))

# %%
print(y_test.value_counts())
print(pd.Series(y_test_predict).value_counts())
print((y_test == y_test_predict).sum(), len(y_test))
# %%
y_train.value_counts(normalize=True)
y_test.value_counts(normalize=True)
# %%
