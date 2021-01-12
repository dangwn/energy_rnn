'''
Author: Dan Gawne
Date  : 2021-01-12
'''
#%%
#--------------------------------------------------------------------------------------------------
# Imports
#--------------------------------------------------------------------------------------------------
from model_code.create_model import GRU, train_model
from model_code import get_data

import mlflow
import yaml
import numpy as np

import torch
from ignite.metrics import MeanSquaredError

#%%
#--------------------------------------------------------------------------------------------------
# File Params
#--------------------------------------------------------------------------------------------------
train_perc = 0.9
batch_size = 32

keys = ['Consumption']
target_key = 'Consumption'
seq_len = 30

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
X_raw, y_raw = get_data.get_sequences(
    keys = keys,
    target_key = target_key,
    seq_len = seq_len
)

train_size = round(train_perc*len(X_raw))

X = torch.tensor(X_raw, dtype = torch.float32).to(device)
y = torch.tensor(y_raw, dtype = torch.float32).unsqueeze(1).to(device)

class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, X, y):
        super().__init__()
        
        self.X = X
        self.y = y

    def __getitem__(self, ind):
        return self.X[ind], self.y[ind]

    def __len__(self):
        return len(self.X)

trainset = MyDataSet(X[:train_size],y[:train_size])
testset = MyDataSet(X[train_size:],y[train_size:])

trainloader = torch.utils.data.DataLoader(trainset, shuffle = True, batch_size = batch_size)
testloader = torch.utils.data.DataLoader(testset, shuffle = True, batch_size = batch_size)


#%%
#--------------------------------------------------------------------------------------------------
# MLFlow Setup
#--------------------------------------------------------------------------------------------------
'''mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(experiment_name)
num_experiments = 10'''

#%%
#--------------------------------------------------------------------------------------------------
# Create and Train model
#--------------------------------------------------------------------------------------------------
net = GRU(
    input_size = len(keys),
    gru_size = 64,
    gru_layers = 2,
    hidden_size = 64,
    output_size = 1,
    device = device
).to(device)

optimizer = torch.optim.Adam(net.parameters(), lr = 1e-3)
criterion = torch.nn.MSELoss()
val_metrics = {'MSE' : MeanSquaredError()}
num_epochs = 10

train_model(
    net,
    trainloader = trainloader,
    testloader = testloader,
    optimizer = optimizer,
    criterion = criterion,
    val_metrics = val_metrics,
    num_epochs = num_epochs,
    device = device
)

#%%
#--------------------------------------------------------------------------------------------------
# 
#--------------------------------------------------------------------------------------------------
