from flask import Flask, jsonify, request
import pickle
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

@app.route("/", methods=["POST", "GET"])
def prediction():
    if (request.method == "POST"):

        with open("C:\\Workspace\\actitivity_recognition_repo\\saved_models\\rfc_model.pkl", "rb") as f:
            rfc_model = pickle.load(f)
        with open("C:\\Workspace\\actitivity_recognition_repo\\saved_artifacts\\std_scaler.pkl", "rb") as f:
            std_scaler = pickle.load(f)
        with open("C:\\Workspace\\actitivity_recognition_repo\\saved_artifacts\\label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)

        avgrss12 = request.json["avgrss12"]
        varrss12 = request.json["varrss12"]
        avgrss13 = request.json["avgrss13"]
        varrss13 = request.json["varrss13"]
        avgrss23 = request.json["avgrss23"]
        varrss23 = request.json["varrss23"]

        raw_input = [avgrss12, varrss12, avgrss13, varrss13, avgrss23, varrss23]
        trf_input = std_scaler.transform([raw_input])
        raw_pred = rfc_model.predict(trf_input)
        trf_pred = label_encoder.inverse_transform(raw_pred)[0]
        return_statement = f"The prediction is {trf_pred}."
        
        return jsonify(return_statement)


if __name__ == '__main__':
    app.run()