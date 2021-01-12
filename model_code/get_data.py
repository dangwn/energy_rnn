'''
Author: Dan Gawne
Date  : 2021-01-12
'''

import yaml
import pandas as pd
import numpy as np
import random

with open('file_paths.yml','r') as f:
    try:
        file_paths = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(e)

data_dir = file_paths['data'][0]

def pull_data():
    df =  pd.read_csv(data_dir)
    df.set_index('Date', inplace = True)
    return df

def get_sequences(
    keys,
    target_key : str,
    seq_len : int,
    num_future_days = 1,
    shuffle = True
):
    '''
    Returns sequences of features and their respective labels
    =============================================================================================
    Inputs:
        - keys            : The columns in the data you want to use (str/array-like)
        - target_key      : The target column to make the labels (str)
        - seq_len         : The number of days in each sequence (int)
        - num_future_days : The number of days into the future you want to predict (default = 1)
        - shuffle         : Whether the sequences are shuffled
    Outputs:
        - X : Sequences of features
        - y : Labels
    =============================================================================================
    '''
    if type(keys) == str:
        df = pull_data()[[keys]]
    else:
        df = pull_data()[keys]

    df['Label'] = df.copy()[target_key].shift(-num_future_days)
    df = df.dropna()
    sequences = []
    labels = []

    for i in range(len(df) - seq_len):
        temp = df.iloc[i:i+seq_len]
        sequences.append(np.array(temp[keys]))
        labels.append(temp.iloc[-1,-1])

    if shuffle:
        p = list(zip(sequences,labels))
        random.shuffle(p)

        sequences = [i for i,j in p]
        labels    = [j for i,j in p]

    X = np.array(sequences)
    y = np.array(labels)

    if len(X.shape) == 2:
        X = np.expand_dims(X,-1)

   




    return X,y
