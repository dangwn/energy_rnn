'''
Author: Dan Gawne
Date  : 2021-01-12
'''
#%%
#--------------------------------------------------------------------------------------------------
# Imports
#--------------------------------------------------------------------------------------------------
from model_code import create_model
from model_code import get_data

import mlflow
import yaml
import numpy as np

import torch

#%%
#--------------------------------------------------------------------------------------------------
# Setup Device
#--------------------------------------------------------------------------------------------------
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('GPU Detected:', torch.cuda.get_device_name())

#%%
#--------------------------------------------------------------------------------------------------
# Read in File Paths
#--------------------------------------------------------------------------------------------------
with open('file_paths.yml','r') as f:
    try:
        file_paths = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(e)

tracking_uri = r'file:' + file_paths['tracking_uri'][0]
experiment_name = 'German Power - GRU'

#%%
#--------------------------------------------------------------------------------------------------
# Get Sequence Data and Make DataLoaders
#--------------------------------------------------------------------------------------------------
X,y = get_data.get_sequences(
    keys = ['Consumption'],
    target_key = 'Consumption',
    seq_len = 30
)

print(len(X))

'''
#%%
#--------------------------------------------------------------------------------------------------
# MLFlow Setup
#--------------------------------------------------------------------------------------------------
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(experiment_name)

#%%
#--------------------------------------------------------------------------------------------------
# Optimizer, Criterion, Metrics
#--------------------------------------------------------------------------------------------------
optimizer = torch.optim.Adam(net.parameters(), lr = 1e-3)
criterion = torch.nn.MSELoss()
val_metrics = {'MSE' : MeanSquaredError()}

'''
#%%
#--------------------------------------------------------------------------------------------------
# 
#--------------------------------------------------------------------------------------------------



#%%
#--------------------------------------------------------------------------------------------------
# 
#--------------------------------------------------------------------------------------------------
