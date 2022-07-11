from distutils.command.config import config
import os
from preprocess_functions import preprocess_data
from functions import read_params
from logger.myLogger import getmylogger
import argparse
from sklearn.model_selection import train_test_split
import json
import pandas as pd


logger = getmylogger(__name__)

def preprocess_dataset(config_path):
    config = read_params(config_path)
    raw_data_path = config["data_paths"]["raw_dataset_csv"]
    cols_to_remove = config["preprocess"]["cols_to_remove"]
    target_col = config["base"]["target_col"]
    artifact_dir = config["artifact_dir"]
    features_schema = config["features_schema_path"]
    data_to_prp = preprocess_data(raw_data_path)
    data_to_prp.remove_cols(cols_to_remove)
    min_max_dict = data_to_prp.df.describe().loc[["min", "max"]].to_dict() #feature schema in dict
    data_to_prp.label_encode(target_col, artifact_dir)
    data_to_prp.standard_scale(target_col, artifact_dir)
    logger.info("Pre-processing of raw data successful!!")
    if not os.path.exists("../tests"):
        os.mkdir("../tests")
    with open(features_schema, "w") as f:
        json.dump(min_max_dict, f)
    logger.info("Features schema saved!")

    return data_to_prp.df



def split_and_save_data(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    split_ratio = config["split_data"]["test_size"]
    random_state = config["base"]["random_state"]


    df = preprocess_dataset(config_path)
    train, test = train_test_split(df, test_size=split_ratio, random_state=random_state)

    train.to_csv(train_data_path, index=False)
    test.to_csv(test_data_path, index=False)
    logger.info("Saved training and testing sets!")


if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    split_and_save_data(config_path=parsed_args.config)