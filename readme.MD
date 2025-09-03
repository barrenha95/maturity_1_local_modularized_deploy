This repository was created to demonstrate skills in machine learning deployment, specifically showing how a deployment is made with the level 1 of maturity, or in other words, using modularization structure:

## What's new here?
- Modularized structure (aka it's not anymore a single code that do everything)
- Use of ML Flow to registry and manage the life cycle of the ML model
- Use of a simple Feature Store, in order to make it easier to manage and serve features


## Objetcives
- Show different ways to deploy a machine learning model
- **The objective is not to make the best model.** The focus of this project is to show how a deployment of a machine learning model is made. In other words, the statistical part of the model is there only as a fast example.


## Content 
*Direct from the Kaggle page*

Data obtained from:
https://www.kaggle.com/datasets/thegoanpanda/fastag-fraud-detection-datesets-fictitious

- Transaction_ID: A unique identifier assigned to each toll transaction, used for tracking and referencing purposes. (Like a receipt number for a toll payment).
- Timestamp: The exact date and time when the toll transaction occurred, crucial for chronological analysis and potential dispute resolution.
- Vehicle_Type: The type of vehicle involved in the transaction, such as car, truck, or motorcycle, often influencing toll rates and regulations.
- FastagID: A unique identifier linked to a prepaid toll payment account, enabling electronic toll collection without manual cash payments.
- TollBoothID: A unique identifier assigned to each tollbooth, used for pinpointing locations and analyzing traffic patterns.
- Lane_Type: The specific lane used for the transaction.
- Vehicle_Dimensions: The length, width, and height of the vehicle, potentially influencing toll calculations and safety measures.
- Transaction_Amount: The total cost.
- Amount_paid: The actual amount paid during the transaction, reflecting any discounts, penalties, or outstanding balances.
- Geographical_Location: The precise latitude and longitude of the tollbooth, enabling mapping and spatial analysis.
- Vehicle_Speed: The speed of the vehicle as it passed through the tollbooth, potentially used for safety monitoring and traffic management.
- Vehicle_Plate_Number: The unique license plate number of the vehicle, used for identification and potential law enforcement purposes.
- Fraud_indicator: A flag indicating potential fraudulent activity associated with the transaction, prompting further investigation or preventive measures.


## Steps
To make the project more similar to real-life problems, I split it into small tasks.
As the project progresses, I will add the time spent on each task. 

1. Problem Selection & Setup: Choose a simple dataset, define objective, set up local environment. **1 hour and 23 minutes**
2. Data Module: Load, clean and split datasets. **10 minutes**
3. Feature Engineering Module: Transform raw data into model-ready features. **2 hours and 14 minutes**
4. Feature Store Module: Store processed features making it easier to manage and serve models. **2 hours and 2 minutes**
5. Unit testing Module: Test each feature from the feature store. **1 hour and 45 minutes**
6. Model Training Module: Train and save the model. **61 + 66 + 17 + 137 minutes**
7. Evaluation Module: Model evaluation metrics. **23 + 12 minutes**
8. Manual Inference Module: Load model and predict new data. **88 minutes**
9. Experiment Tracking Module: Use ML Flow to artifacts registry. **16 + 13 minutes**
10. Deployment Modules: API calling the inference script. **75 + 41 minutes**
11. Documentation: Write a README explaining how to run each script manually. Mention which files to edit or execute.
12. Reflection & Publication Prep: Write a summary (for LinkedIn or blog): what this maturity level is, what you did, what’s missing, and the next steps.