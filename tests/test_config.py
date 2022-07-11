import pytest
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from prediction_service.predictor import get_prediction
import prediction_service

input_data = {
    "incorrect_range": 
    {"avg_rss12": -100, 
    "var_rss12": 100, 
    "avg_rss13": 100, 
    "var_rss13": 100, 
    "avg_rss23": -100, 
    "var_rss23": -100, 
    },

    "correct_range":
    {"avg_rss12": 3, 
    "var_rss12": 8, 
    "avg_rss13": 12, 
    "var_rss13": 5, 
    "avg_rss23": 20, 
    "var_rss23": 7, 
    },

    "incorrect_col":
    {"avgrss12": 3, 
    "varrss12": 8, 
    "avgrss13": 12, 
    "varrss13": 5, 
    "avgrss23": 20, 
    "varrss23": 7, 
    }
}

TARGET_range = ['bending1', 'bending2', 'cycling', 'lying', 'sitting', 'standing', 'walking']

def test_correct_range(data=input_data["correct_range"]):
    res = get_prediction(data)
    assert  res in TARGET_range

def test_incorrect_range(data=input_data["incorrect_range"]):
    with pytest.raises(prediction_service.predictor.NotInRange):
        res = get_prediction(data)

def test_incorrect_col(data=input_data["incorrect_col"]):
    with pytest.raises(prediction_service.predictor.NotInColumns):
        res = get_prediction(data)