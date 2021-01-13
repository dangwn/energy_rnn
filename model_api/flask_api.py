'''
Author: Dan Gawne
Date: 2021-01-13
'''

from flask import Flask, request
import torch
import mlflow
import waitress
import yaml

with open('model_api\\api_file_paths.yml','r') as f:
    try:
        file_paths = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(e)

model_loc = r'file:' + file_paths['model_destination'][0]
seq_len = 30
experiment_name = f'German Power - GRU ({seq_len} Day)'

# Get Model
model = ml