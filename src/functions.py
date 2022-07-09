import os

def log_bad_data(bad_data_list):
    with open("bad_data_files.log", "w") as f:
        for file in bad_data_list:
            f.write(file + "\n")
        f.close()