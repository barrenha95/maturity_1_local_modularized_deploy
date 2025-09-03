import requests
from datetime import datetime

new_data = [
    {"Transaction_ID": 1
    , "Timestamp": str(datetime.now())
    , "Vehicle_Type": "Bus"
    , "FastagID": "FTG-001-ABC-121"
    , "TollBoothID": "A-101"
    , "Lane_Type": "Express"
    , "Vehicle_Dimensions": "Large"
    , "Transaction_Amount": "350"
    , "Amount_paid": "120"
    , "Geographical_Location": "13.059816123454882, 77.77068662374292"
    , "Vehicle_Speed": "65"
    , "Vehicle_Plate_Number": "KA11AB1234"
    },
]

url = 'http://localhost:8000/predict'

response = requests.post(url, json = new_data)
print(f"\nStatus Code: {response.status_code}")

if response.status_code == 200:
    try:
        print(response.json())
    except requests.exceptions.JSONDecodeError:
        print("Error in the response.")
else:
    print("Error in request.")