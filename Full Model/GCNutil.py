import numpy as np
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as F
from GCN import *
from DataManager import dataManager
from torchsummary import summary
from tqdm import tqdm
from DataManagerUtil import create_DAD as create_DAD

def save_ckp(state, f_path):
    torch.save(state, f_path)
    print("model saved")

def load_ckp(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    return checkpoint

# takes in numpy arrays
def train_GCN(A, X_train, X_test, checkpoint, device_name='cpu', load=False, print_summary=True):
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

    # initialize model
    model = Net(parameters['body'], parameters['n_layers'], F.relu, bias=True).to(device)
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
        summary(model, [(31, 640), (31, 31)], device = device_name)

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

            sim_matrix = model(batch_x, A).to(device)
            train_loss = criterion(sim_matrix, A)
            train_loss.backward()
            optimizer.step()

            model.eval()
            sim_matrix_valid = model(batch_v, A).to(device)
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

# get sim matrix from unsupervised model
def get_sim_matrix_from_model(dm, model_dir):
    checkpoint = load_ckp(model_dir)

    # preprocess inputs
    A = torch.from_numpy(create_DAD(dm.A_train)).float()
    X_train = torch.from_numpy(dm.X_train).float()
    X_test = torch.from_numpy(dm.X_test).float()
    device = torch.device('cpu')
    A = A.to(device)
    parameters = checkpoint['parameters']
    input_train = X_train.to(device)
    input_test = X_test.to(device)

    # initialize model
    model = Net(parameters['body'], parameters['n_layers'], F.relu, bias=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=parameters['learning_rate'],
                                 weight_decay=parameters['weight_decay'])

    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    model.eval()
    sim_train = []
    sim_test = []
    permutation = range(input_train.shape[0])
    batch_size = parameters['batch_size']

    sim1 = model(input_train, A).to(device)
    sim2 = model(input_test, A).to(device)
    return sim1.cpu().detach().numpy(), sim2.cpu().detach().numpy()

# get pretrained sim matrix
def load_flattened_sim_matrix(DIR, dm):
    sim_train = np.load(DIR)
    sim_test = np.load(DIR)

    sim_train = np.delete(sim_train, np.s_[dm.train_indices], axis=0)
    sim_test = np.delete(sim_test, np.s_[dm.test_indices], axis=0)

    sim_train_list = []
    tri_indices = torch.triu_indices(31, 31)
    for j in range(sim_train.shape[0]):
        sim = np.zeros((1, 31, 31))
        for i in range(496):
            sim[0, int(tri_indices[0, i]), int(tri_indices[1, i])] = sim_train[j, i]
        sim_train_list.append(sim)

    sim_test_list = []
    for j in range(sim_test.shape[0]):
        sim = np.zeros((1, 31, 31))
        for i in range(496):
            sim[0, int(tri_indices[0, i]), int(tri_indices[1, i])] = sim_test[j, i]
        sim_test_list.append(sim)

    return np.vstack(sim_train_list), np.vstack(sim_test_list)


