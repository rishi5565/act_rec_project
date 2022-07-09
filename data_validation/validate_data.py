import pandas as pd
import os
import re

from requests import head

regex = r'(.*)\.csv' # the regex to validate csv files before importing
col_name_list = ['# Columns: time', 'avg_rss12', 'var_rss12', 'avg_rss13', 'var_rss13', 'avg_rss23', 'var_rss23']
col_dtype_list = ['int64', 'float64', 'float64', 'float64', 'float64', 'float64', 'float64']

class validate_data:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.all_files = []
        self.good_files = []
        self.bad_files = []

        for root, _, files in os.walk(self.data_dir):
            for name in files:
                self.all_files.append(os.path.join(root, name))
    
    def csv_file_val(self):
        for file in self.all_files:
            if re.match(regex, file):
                self.good_files.append(file)
            else:
                self.bad_files.append(file)

    def panda_read_val(self): # we verify if pandas can read the data according to the DSA format
        for file in self.good_files:
            try:
                pd.read_csv(file, header=4)
            except:
                self.good_files.remove(file)
                self.bad_files.append(file)
    

    def col_name_val(self):
        for file in self.good_files:
            df = pd.read_csv(file, header=4)
            if list(df.columns) != col_name_list:
                self.good_files.remove(file)
                self.bad_files.append(file)

    def col_dtype_val(self):
        for file in self.good_files:
            df = pd.read_csv(file, header=4)
            if list(df.dtypes) != col_dtype_list:
                self.good_files.remove(file)
                self.bad_files.append(file)

    def missing_data_val(self):
        for file in self.good_files:
            df = pd.read_csv(file, header=4)
            if not pd.DataFrame.equals(df, df.dropna()):
                self.good_files.remove(file)
                self.bad_files.append(file)               

    def run_all_validations(self):
        self.csv_file_val()
        self.panda_read_val()
        self.col_name_val()
        self.col_dtype_val()
        self.missing_data_val()
