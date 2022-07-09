
from data_preprocess.preprocess import preprocess_data
from functions import read_params
from logger.myLogger import getmylogger
import argparse

logger = getmylogger(__name__)

def preprocess_dataset(config_path):
    config = read_params(config_path)
    raw_data_path = config["data_paths"]["raw_dataset_csv"]
    cols_to_remove = config["preprocess"]["cols_to_remove"]
    target_col = config["base"]["target_col"]
    artifact_dir = config["artifact_dir"]
    data_to_prp = preprocess_data(raw_data_path)
    data_to_prp.remove_useless_cols(cols_to_remove)
    data_to_prp.label_encode(target_col, artifact_dir)
    data_to_prp.standard_scale(target_col, artifact_dir)
    print(data_to_prp.df.head())



if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    preprocess_data(config_path=parsed_args.config)