import os
from flask import Flask, jsonify, render_template, request
from prediction_service.predictor import get_prediction
from logger.myLogger import getmylogger
import warnings
warnings.filterwarnings("ignore")

webapp_root = "webapp"

static_dir = os.path.join(webapp_root, "static")
template_dir = os.path.join(webapp_root, "templates")

logger = getmylogger(__name__)

app = Flask(__name__, static_folder=static_dir, template_folder=template_dir)

@app.route("/", methods=["POST", "GET"])

def prediction():
    if request.method == "POST":
        dict_req = dict(request.form)
        try:
            pred = get_prediction(dict_req) # returns activity name as string
            logger.info("Prediction Fetched!")
            response = f"The activity is {pred}."
            return render_template("index.html", response=response)

        except Exception as e:
            logger.critical(f"Error occurred while fetching prediction.\n{e}")
            error = {"error": e}
            return render_template("404.html", error=error)

    else:
        return render_template("index.html")
        

if __name__ == '__main__':
    app.run()