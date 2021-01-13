'''
Author: Dan Gawne
Date: 2021-01-13
'''

from flask import Flask, request
import torch
import mlflow
import waitress
import json

from deploy_model import deploy_model

app = Flask(__name__)

@app.route('/')
def index():
    '''
    Index page for API
    '''
    return('Welcome to my API')

@app.route('/invocations',methods = ['POST'])
def query_model():
    '''
    Returns model predictions from posted data
    =============================================
    Posted Data:
        Form -> {"data":data, "num_pred":num_pred}
        - data     : The data that will be used to predict
        - num_pred : The number of extra predicted days (if function conditions are satisfied)
    Returns:
        Either:
            - Model predictions
            - Message to say something went wrong
    '''
    try:
        data = json.loads(request.data)
        X = torch.tensor(data['data'],dtype = torch.float32)
        num_pred = data['num_pred']
        pred = deploy_model(X, num_pred)
        return json.dumps(pred)
    except:
        pass
    
    return 'Something went wrong...'


# Serve the model to port 6000
waitress.serve(app, host='0.0.0.0', port = 6000)