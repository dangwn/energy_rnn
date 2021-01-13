'''
Author: Dan Gawne
Date: 2021-01-13
'''

import torch
import mlflow
import yaml
import os

with open('api_file_paths.yml','r') as f:
    try:
        file_paths = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(e)

dir = r'file://' + os.path.join(os.getcwd(), 'model').replace('\\','/')
model = mlflow.pytorch.load_model(dir)