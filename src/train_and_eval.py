import os
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from functions import read_params
import argparse
import pickle
from logger.myLogger import getmylogger
import json

logger = getmylogger(__name__)


def eval_metrics(actual, pred):
    # roc_score = roc_auc_score(actual, pred)
    acc_score = accuracy_score(actual, pred)
    return acc_score


def train_and_evaluate(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    random_state = config["base"]["random_state"]
    algorithm_name = config["algorithm_name"]
    model_dir = config["model_dir"]

    n_estimators = config["estimators"]["RandomForestClassifier"]["params"]["n_estimators"]
    criterion = config["estimators"]["RandomForestClassifier"]["params"]["criterion"]

    target = config["base"]["target_col"]

    train = pd.read_csv(train_data_path)
    test = pd.read_csv(test_data_path)

    train_y = train[target]
    test_y = test[target]

    train_x = train.drop(target, axis=1)
    test_x = test.drop(target, axis=1)

    rfc_model = RandomForestClassifier(
                n_estimators=n_estimators, 
                criterion=criterion,
                random_state=random_state)
    logger.info(f"Fitting the {algorithm_name} model......")
    rfc_model.fit(train_x, train_y)
    logger.info("Model fit successful!")

    prediction = rfc_model.predict(test_x)

    acc_score = eval_metrics(test_y, prediction)
    # logger.info(f"ROC AUC: {round(roc_score*100, 2)}%")
    logger.info(f"Accuracy: {round(acc_score*100, 2)}%")


    scores_file = config["reports"]["scores"]
    params_file = config["reports"]["params"]

    with open(scores_file, "w") as f:
        scores = {
            # "ROC AUC": roc_score,
            "Accuracy": acc_score}
        json.dump(scores, f, indent=4)

    with open(params_file, "w") as f:
        params = {
            "n_estimators": n_estimators,
            "criterion": criterion}
        json.dump(params, f, indent=4)
    logger.info("Saved Scores and Parameters in respective json files.")

    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_path = os.path.join(model_dir, "rfc_model.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(rfc_model, f)



if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config) 