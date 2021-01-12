from model_code import create_model
import torch
import torch.optim as optim

from ignite.metrics import MeanSquaredError


net = create_model.GRU(
    input_size = 3,
    gru_size = 32,
    gru_layers = 1,
    hidden_size = 16,
    output_size = 1,
)
X = torch.randn(1,2,3)

loader_data = [(torch.randn(2,3),torch.rand(1)) for i in range(5)]
loader = torch.utils.data.DataLoader(loader_data, batch_size = 1) 


def test_model_creation():
    #Mainly ensuring the GRU can be constructed and data fed through
    y = net(X)

    #Ensure output tensor is of correct size
    assert y.size() == torch.Size([1, 1])


def test_model_trainer():
    
    optimizer = torch.optim.SGD(net.parameters(), lr = 1e-3)
    criterion = torch.nn.MSELoss()
    val_metrics = {'MSE' : MeanSquaredError()}

    create_model.train_model(
        model = net,
        trainloader = loader,
        testloader = loader,
        optimizer = optimizer,
        criterion = criterion,
        val_metrics = val_metrics,
        num_epochs = 1,
        device = 'cpu',
        verbose = False
    )
    
    #Mainly ensuring the GRU can be constructed and data fed through
    y = net(X)

    #Ensure output tensor is of correct size
    assert y.size() == torch.Size([1, 1])