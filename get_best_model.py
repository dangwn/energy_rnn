'''
Author: Dan Gawne
Date: 2021-01-13
'''

import mlflow
import yaml

with open('file_paths.yml', 'r') as f:
    try:
        file_paths = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)

tracking_uri = file_paths['tracking_uri'][0]
destination_folder = file_paths['model_destination'][0]

seq_len = file_paths['seq_len'][0]
experiment_name = f'German Power - GRU ({seq_len} Day)'

