import pickle
from src.functions import read_params
import json

config_path = "params.yaml"
schema_path = "tests/features_schema.json"

class NotInRange(Exception):
    def __init__(self, message="Entered values not in expected range"):
        self.message = message
        super().__init__(self.message)

class NotInColumns(Exception):
    def __init__(self, message="Not in columns"):
        self.message = message
        super().__init__(self.message)

def get_schema(schema_path=schema_path):
    with open(schema_path) as json_file:
        schema = json.load(json_file)
    return schema


def validate_input(dict_request):
    def _validate_cols(col):
        schema = get_schema()
        actual_cols = schema.keys()
        if col not in actual_cols:
            raise NotInColumns

    def _validate_values(col):
        schema = get_schema()
        if not (schema[col]["min"] <= float(dict_request[col]) <= schema[col]["max"]) :
            raise NotInRange

    for col, val in dict_request.items():
        _validate_cols(col)
        _validate_values(col)
    
    return True    


def predict(data):
    config = read_params(config_path)
    model_path = config["webapp_model"]
    scaler_path = config["scaler"]
    label_encoder_path = config["label_encoder"]  

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        std_scaler = pickle.load(f)
    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    trf_input = std_scaler.transform(data)
    raw_pred = model.predict(trf_input)
    trf_pred = label_encoder.inverse_transform(raw_pred)[0]

    return trf_pred



def get_prediction(dict_request):
    if validate_input(dict_request):
        data = dict_request.values()
        data = [list(map(float, data))]
        response = predict(data)
        return response


