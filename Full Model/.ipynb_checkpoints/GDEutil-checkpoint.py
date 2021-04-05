# save & load checkpoint
import shutil
from torchsummary import summary
from tqdm import tqdm
from GDE import *
from GDEsolvers import *
import numpy as np

def save_ckp(state, f_path):
    torch.save(state, f_path)
    print("model saved")


def load_ckp(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    return checkpoint

def load_model(checkpoint, device_name ='cpu' ):
    if device_name == 'cuda':
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print("device set to cuda") if device == torch.device('cuda') else print("cuda is not available")
    elif device_name == 'cpu':
        device = torch.device('cpu')
        print("device set to cpu")
    else:
        device = torch.device('cpu')
        print("unknown device")
    
    parameters = checkpoint['parameters']
    model = ODENet(parameters['solver'], parameters['body'], parameters['hidden_layers'], A,
                   parameters['solver_params']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=parameters['learning_rate'],
                                 weight_decay=parameters['weight_decay'])
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer

# takes in numpy arrays
def train_GDE(A, X_train, X_test, checkpoint, device_name = 'cpu',load=False, print_summary=False):
    # preprocess inputs
    A = torch.from_numpy(A).float()
    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    if device_name == 'cuda':
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print("device set to cuda") if device == torch.device('cuda') else print("cuda is not available")
    elif device_name == 'cpu':
        device = torch.device('cpu')
        print("device set to cpu")
    else:
        device = torch.device('cpu')
        print("unknown device")
    parameters = checkpoint['parameters']
    cumepoch = 0
    A = A.to(device)
    input_features = X_train.to(device)
    valid_features = X_test.to(device)

    # (solver, body_channels, hidden_layers, A, solver_params)
    # initialize model
    model = ODENet(parameters['solver'], parameters['body'], parameters['hidden_layers'], A,
                   parameters['solver_params']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=parameters['learning_rate'],
                                 weight_decay=parameters['weight_decay'])
    criterion = sim_loss()

    # load past checkpoint if any
    if load:
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        cumepoch = checkpoint['cumepoch']

    # print some model info
    if print_summary:
        print(model)
        summary(model, [(31, 640)], device = device_name)

    n_epochs = parameters['num_epochs']
    batch_size = parameters['batch_size']

    model.train()
    for epoch in range(1, n_epochs + 1):

        permutation = torch.randperm(input_features.shape[0])
        permutation_v = torch.randperm(valid_features.shape[0])

        for i in tqdm(range(0, input_features.shape[0], batch_size)):
            model.train()
            optimizer.zero_grad()

            indices = permutation[i:i + batch_size] if i + batch_size < input_features.shape[0] else permutation[i:]
            indices_v = permutation_v[i:i + batch_size] if i + batch_size < valid_features.shape[0] else permutation_v[i:]
            batch_x = input_features[indices, :, :]
            batch_v = valid_features[indices_v, :, :]

            sim_matrix = model(batch_x).to(device)
            train_loss = criterion(sim_matrix, A)
            train_loss.backward()
            optimizer.step()

            model.eval()
            sim_matrix_valid = model(batch_v).to(device)
            valid_loss = criterion(sim_matrix_valid, A)

            print("Epoch:", epoch + cumepoch, "  Batch:", int((i / batch_size) + 1), "of size", batch_size,
                  "  Train loss:", float(train_loss), "  Valid loss:", float(valid_loss), end="\r")
        print()

    checkpoint = {
        'parameters': parameters,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'cumepoch': n_epochs + cumepoch
    }

    return model, checkpoint