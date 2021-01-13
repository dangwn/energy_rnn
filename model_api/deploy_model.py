'''
Author: Dan Gawne
Date: 2021-01-13
'''

import torch
import mlflow
import os

dir = r'file://' + os.path.join(os.getcwd(), 'model').replace('\\','/')
model = mlflow.pytorch.load_model(dir)

def deploy_model(
    X : torch.Tensor,
    continuous_modelling : int = 0
):
    '''
    Returns predictions from the model based on sequenced data
    =============================================
    Inputs:
        - X                    : A tensor containing the modelling data
        - continuous_modelling : The number of extra days to be modelled (default = 2)
            Note: This feature may only be invoked when using one feature
    Returns:
        - predictions : A list containing the model predictions
    '''
        
    y = model(X)
    predictions = [y.item()]

    if X.shape[2] == 1:
        for i in range(continuous_modelling):
            y = y.view(1,1,-1)
            
            X = torch.cat((X,y),1)
            
            y = model(X)
            predictions.append(y.item())

    return predictions    

