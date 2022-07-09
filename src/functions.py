import yaml


def log_bad_data(bad_data_list):
    with open("bad_data_files.log", "w") as f:
        for file in bad_data_list:
            f.write(file + "\n")
        f.close()

    
def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config