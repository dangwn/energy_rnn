'''
Author: Dan Gawne
Date: 2021-01-13
'''

import mlflow
import yaml
import os
import torch

with open('file_paths.yml', 'r') as f:
    try:
        file_paths = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)

tracking_uri = r'file:' + file_paths['tracking_uri'][0]
destination_folder = file_paths['model_destination'][0]

seq_len = file_paths['seq_len'][0]
experiment_name = f'German Power - GRU ({seq_len} Day)'


client = mlflow.tracking.MlflowClient(tracking_uri = tracking_uri)

experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
runs = client.search_runs(experiment_id, order_by = ['metrics.Test_MSE'])

best_run_id = runs[0].info.run_id
client.download_artifacts(best_run_id, 'model', destination_folder + '/..')

model = mlflow.pytorch.load_model(r'file://' + destination_folder + '/../model')
torch.save(model, destination_folder + '/final_model.pt')

