import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os
from logger.myLogger import getmylogger


logger = getmylogger(__name__)


class preprocess_data:
    def __init__(self, raw_path):
        self.raw_path = raw_path
        self.df = pd.read_csv(self.raw_path)


    def remove_cols(self, cols_to_remove):
        self.df.drop(cols_to_remove, axis=1, inplace=True)
        logger.debug("Removed irrelevant columns!")


    def label_encode(self, target_col, artifact_dir):
        self.label_encoder = LabelEncoder()
        self.df[target_col] = self.label_encoder.fit_transform(self.df[target_col])

        if not os.path.exists(artifact_dir):
            os.mkdir(artifact_dir)
        with open(os.path.join(artifact_dir, "label_encoder.pkl"), "wb") as f:
            pickle.dump(self.label_encoder, f)
        logger.debug("Dumped label encoder pickle file!")


    def standard_scale(self, target_col, artifact_dir):
        self.std_scaler = StandardScaler()
        dfx = self.df.drop(target_col, axis=1)
        dfx = pd.DataFrame(self.std_scaler.fit_transform(dfx), columns=dfx.columns)
        self.df = pd.concat([dfx, self.df[target_col]], axis=1)

        if not os.path.exists(artifact_dir):
            os.mkdir(artifact_dir)
        with open(os.path.join(artifact_dir, "std_scaler.pkl"), "wb") as f:
            pickle.dump(self.std_scaler, f)
        logger.debug("Dumped standard scaler pickle file!")