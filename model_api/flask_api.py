'''
Author: Dan Gawne
Date: 2021-01-13
'''

from flask import Flask, request
import torch
import mlflow
import waitress
import yaml
import os

with open('api_file_paths.yml','r') as f:
    try:
        file_paths = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(e)

cwd = file_paths['model_destination'][0]

model_loc = r'file://' + os.path.join(os.getcwd(),'model').replace('\\','/')
print(model_loc,'\n\n')
model = mlflow.pytorch.load_model(model_loc)

