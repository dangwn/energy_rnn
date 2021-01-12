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

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import MeanSquaredError


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
        num_hidden = 0,
        output_size = 1,
        device = 'cpu',
    ):
        super().__init__()
        
        self.gru_size = gru_size
        self.gru_layers = gru_layers
        self.hidden_size = hidden_size
        self.device = device
        self.num_hidden = num_hidden

        self.gru = nn.GRU(
            input_size, gru_size, gru_layers, batch_first = True
        )
        
        self.fc_in = nn.Linear(gru_size, hidden_size)
        self.fc_hidden = nn.Linear(hidden_size,hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(
            self.gru_layers, x.size(0), self.gru_size
        ).to(self.device)

        x , _ = self.gru(x, h0)
        
        out = x[:,-1,:]
        for i in range(self.num_hidden):
            out = F.relu(self.fc_hidden(out))
        out = F.relu(self.fc_in(out))
        
        return self.fc_out(out)


#%%
#--------------------------------------------------------------------------------------------------
# Model Trainer Function 
#--------------------------------------------------------------------------------------------------
def train_model(
    model,
    trainloader,
    testloader,
    optimizer,
    criterion,
    val_metrics,
    num_epochs,
    device,
    verbose = True
):
    trainer = create_supervised_trainer(
        model, optimizer, criterion, device = device
    )
    evaluator = create_supervised_evaluator(
        model, metrics = val_metrics, device = device
    )

    metrics_to_return = 0

    if verbose:
        @trainer.on(Events.EPOCH_STARTED)
        def print_epoch(trainer):
            s = f'Epoch: {trainer.state.epoch}'
            print('================================')
            print(s)
        
        @trainer.on(Events.EPOCH_COMPLETED)
        def print_metrics(trainer):
            evaluator.run(testloader)
            metrics = evaluator.state.metrics
            epoch_no = trainer.state.epoch
            print(f'Metrics: {metrics}')
            
    
    trainer.run(trainloader, max_epochs = num_epochs)


