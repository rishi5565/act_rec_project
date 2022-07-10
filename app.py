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

        avgrss12 = request.form["avgrss12"]
        varrss12 = request.form["varrss12"]
        avgrss13 = request.form["avgrss13"]
        varrss13 = request.form["varrss13"]
        avgrss23 = request.form["avgrss23"]
        varrss23 = request.form["varrss23"]

        try:
            pred = get_prediction(avgrss12, varrss12, avgrss13, varrss13, avgrss23, varrss23) # returns activity name as string
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