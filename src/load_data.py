from distutils.command.config import config
import pandas as pd
import argparse
from logger.myLogger import getmylogger
from data_validation.validate_data import validate_data
from functions import log_bad_data, read_params

logger = getmylogger(__name__)

def get_data(config_path):
    config = read_params(config_path)
    data_dir = config["data_source"]["source_dir"]
    data_val = validate_data(data_dir)
    data_val.run_all_validations()
    log_bad_data(data_val.bad_files)
    logger.info(f"{len(data_val.good_files)} files were successfully validated!")
    logger.info(f"{len(data_val.bad_files)} files failed validation tests and will not proceed further!")
    return data_val.good_files


def load_and_save_raw(config_path):
    df = pd.DataFrame()
    files = get_data(config_path)
    config = read_params(config_path)
    raw_data_path = config["data_paths"]["raw_dataset_csv"]
    for file in files:
        subdf = pd.read_csv(file, header=4)
        with open(file, "r") as f:
            label = str(f.readline().split(" ")[-1]).replace("\n", "")
        f.close()
        subdf["label"] = label
        df = pd.concat([df, subdf], axis=0, ignore_index=True)
    df.to_csv(raw_data_path, index=False)
    logger.info("Created the master raw dataset!")




if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    load_and_save_raw(config_path=parsed_args.config)
