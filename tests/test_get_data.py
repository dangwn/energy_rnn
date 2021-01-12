from model_code import get_data
import pandas as pd
import numpy as np

def test_pull_data():
    df = get_data.pull_data()

    # Ensure full data is returned as DataFrame
    assert type(df) == pd.DataFrame
    assert list(df.columns) == ['Consumption','Wind','Solar','Wind+Solar']
    

def test_get_sequences():
    keys = ['Wind','Solar']
    num_future_days = 20

    X,y = get_data.get_sequences(keys,'Wind',num_future_days)

    # Check compatibility of dimensions of outputs
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == num_future_days
    assert X.shape[2] == len(keys)

    # Ensure extra dimension is added to single key case
    X,y = get_data.get_sequences(keys[0],'Wind',num_future_days)
    assert X.shape[2] == 1