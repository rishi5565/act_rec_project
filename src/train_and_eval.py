import os
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from functions import read_params
import argparse
import pickle
from logger.myLogger import getmylogger
import mlflow
from urllib.parse import urlparse

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
    model_dir = config["production_model_dir"]

    n_estimators = config["estimators"]["RandomForestClassifier"]["params"]["n_estimators"]
    criterion = config["estimators"]["RandomForestClassifier"]["params"]["criterion"]

    target = config["base"]["target_col"]

    train = pd.read_csv(train_data_path)
    test = pd.read_csv(test_data_path)

    train_y = train[target]
    test_y = test[target]

    train_x = train.drop(target, axis=1)
    test_x = test.drop(target, axis=1)

    ############# MLFLOW ##############

    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]
    experiment_name = mlflow_config["experiment_name"]
    run_name = mlflow_config["run_name"]

    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
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

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("criterion", criterion)

        mlflow.log_metric("Accuracy", acc_score)

        registered_model_name = mlflow_config["registered_model_name"]
        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(rfc_model, 
                                    "RFC_model", 
                                    registered_model_name=registered_model_name)
        else:
            mlflow.sklearn.load_model(rfc_model, "RFC_model")

        # cmd run:
        # --backend-store-uri sqlite://mlflow.db
        # --default-artifact-root ./mlflow_artifacts
        # --host 0.0.0.0 -p 1234

        # if not os.path.exists(model_dir):
        #     os.mkdir(model_dir)
        # model_path = os.path.join(model_dir, "rfc_model.pkl")

        # with open(model_path, "wb") as f:
        #     pickle.dump(rfc_model, f)



if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config) 