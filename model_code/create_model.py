'''
Author: Dan Gawne
Date  : 2021-01-12
'''

#%%
#--------------------------------------------------------------------------------------------------
# Imports
#--------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

#%%
#--------------------------------------------------------------------------------------------------
# Model Class
#--------------------------------------------------------------------------------------------------
class GRU(nn.Module):
    def __init__(
        self,
        input_size,
        gru_size,
        gru_layers,
        hidden_size,
        output_size = 1
    ):
        super().__init__()
        
        self.gru_size = gru_size
        self.gru_layers = gru_layers
        self.hidden_size = hidden_size

        self.gru = nn.GRU(
            input_size, gru_size, gru_layers, batch_first = True
        )
        
        self.fc1 = nn.Linear(gru_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(
            self.gru_layers, x.size(0), self.gru_size
        )

        x , _ = self.gru(x, h0)
        
        out = x[:,-1,:]

        out = F.relu(self.fc1(out))
        return self.fc_out(out)


#%%
#--------------------------------------------------------------------------------------------------
# Model Trainer Function 
#--------------------------------------------------------------------------------------------------


#%%
#--------------------------------------------------------------------------------------------------
# Main Method 
#--------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    l = [[1,1,1],
         [2,2,2]]

    X = torch.tensor(l, dtype = torch.float32)
    X = X.unsqueeze(0)

    net = GRU(
        input_size = 3,
        gru_size = 64,
        gru_layers = 3,
        hidden_size = 32,
        output_size = 1,
    )

    y = net(X)
    print(y.item())