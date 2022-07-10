import pickle
from src.functions import read_params

config_path = "params.yaml"

def get_prediction(avgrss12, varrss12, avgrss13, varrss13, avgrss23, varrss23):

    config = read_params(config_path)
    model_path = config["webapp_model"]
    scaler_path = config["scaler"]
    label_encoder_path = config["label_encoder"]

    with open(model_path, "rb") as f:
        rfc_model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        std_scaler = pickle.load(f)
    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    raw_input = [avgrss12, varrss12, avgrss13, varrss13, avgrss23, varrss23]
    trf_input = std_scaler.transform([raw_input])
    raw_pred = rfc_model.predict(trf_input)
    trf_pred = label_encoder.inverse_transform(raw_pred)[0]

    return trf_pred # sends the activity name as a string