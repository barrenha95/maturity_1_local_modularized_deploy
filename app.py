import pandas as pd
from flask import Flask, request, jsonify
from deployment.inference import model_load, model_prediction
from feature_store.feature_engineering import clean_data, engineering

# App creation
app = Flask(__name__)

# Load the last and best model
try:
    best_model = model_load()
except FileNotFoundError:
    best_model = None
    print("Error: Model not found")

@app.route('/predict', methods = ['POST'])
def predict():
    try:
        # Collect data from request
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        #df = pd.DataFrame(data)

        # Cleanning data
        #try:
        #    cleaned = clean_data(df)
        #except:
        #    return jsonify({'error': 'Couldnt clean data'}), 400
        
        # Applying feature engineering
        #try:
        #    engineered = engineering(cleaned)
        #except:
        #    return jsonify({'error': 'Couldnt engineer the features'}), 400

        # Drop non used columns
        #cols_to_drop  = ['transaction_id'
        #              , 'tollboothid'
        #              , 'lane_type'
        #              , 'date'
        #              , 'day'
        #              , 'hour'
        #              , 'vehicle_speed_cat'
        #              , 'fraud_indicator'
        #              , 'month'
        #              , 'year'
        #              ]
        
        #existing_cols = [col for col in cols_to_drop if col in engineered.columns]
    
        #if existing_cols:
        #    engineered = engineered.drop(columns=existing_cols)

        # Prediction
        #if best_model:
        #    predictions = model_prediction(model = best_model, features= engineered)
        #    return jsonify({'predictions':predictions.tolist()})
        #else:
        #    return jsonify({'error':'Model not loaded'}), 500
                
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=False, port=8000)




