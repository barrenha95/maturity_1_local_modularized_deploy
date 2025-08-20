"""
encodings.py

This file stores dictionaries for categorical encoding used across the project.
Keeping them in one place improves maintainability and avoids duplication.
"""

VEHICLE_ENCODING = {
    "suv"       : 1 ,
    "bus"       : 2 ,
    "van"       : 3 ,
    "car"       : 4 ,
    "truck"     : 5 ,
    "sedan"     : 6 ,
    "motorcycle": 7 
}

TOLLBOOTH_ENCODING = {
    "a-101" : 1 ,
    "b-102" : 2 ,
    "c-103" : 3 ,
    "d-104" : 4 ,
    "d-105" : 5 ,
    "d-106" : 6 
}

LANE_ENCODING = {
    "regular": 1,
    "express": 0
}

DIMENSIONS_ENCODING = {
    "small" : 1,
    "medium": 2,
    "large" : 3
}

