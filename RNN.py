import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import copy
from crossbar import crossbar, ticket
from tqdm import tqdm
'''
RNN models for brain state identification
'''
class Simple_block(nn.Module):
    '''
    simple RNN block
    '''
    def __init__(self, in_dim, out_dim, hidden_dim, activation_h = nn.Tanh(), activation_o = nn.Sigmoid()):
        super(Simple_block, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.activation_h = activation_h
        self.activation_o = activation_o

        self.Win = nn.Parameter(torch.Tensor(in_dim, hidden_dim))
        self.bh = nn.Parameter(torch.Tensor(1, hidden_dim))
        self.Wh = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.Wout = nn.Parameter(torch.Tensor(hidden_dim, out_dim))
        self.bo = nn.Parameter(torch.Tensor(1, out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / self.Win.size(1) ** 1 / 2
        self.Win.data.uniform_(-stdv, stdv)

        stdv = 1. / self.bh.size(1) ** 1 / 2
        self.bh.data.uniform_(-stdv, stdv)

        stdv = 1. / self.Wh.size(1) ** 1 / 2
        self.Wh.data.uniform_(-stdv, stdv)

        stdv = 1. / self.Wout.size(1) ** 1 / 2
        self.Wout.data.uniform_(-stdv, stdv)

        stdv = 1. / self.bo.size(1) ** 1 / 2
        self.bo.data.uniform_(-stdv, stdv)

    def forward(self, X, h):
        h_1 = self.activation_h(torch.matmul(X, self.Win) + torch.matmul(h, self.Wh) + self.bh)
        out = self.activation_o(torch.matmul(h_1, self.Wout) + self.bo)
        return out, h_1


class RNN(nn.Module):
    '''
    full RNN model
    '''
    def __init__(self, in_dim, out_dim, hidden_dim, activation_h = nn.Tanh(), activation_o = nn.Sigmoid(), device = torch.device('cpu')):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        in_dim = torch.triu_indices(in_dim, in_dim).shape[1]
        self.block = Simple_block(in_dim, out_dim, hidden_dim, activation_h, activation_o)
        self.device = device

    def forward(self, X):
        X = self.flatten(X)
        T = X.shape[0]
        # X is a tensor of chronological samples with T x ... dimension where T is time
        h = torch.zeros([1, self.hidden_dim]).to(self.device)
        predictions = torch.zeros([T,1]).to(self.device)
        for t in range(T):
            predictions[t, :], h = self.block(X[t, :], h)
        #print(predictions)
        return predictions

    def flatten(self, sim_matrices):
        tri_indices = torch.triu_indices(sim_matrices.shape[1], sim_matrices.shape[2])
        return sim_matrices[:, tri_indices[0, :], tri_indices[1, :]]


# acc functions

from sklearn.metrics import f1_score as F1_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn import metrics
'''
Accuracy functions
'''
def accuracy(X, Y, threshold=0.5):
    '''
    plain accuracy (not good for imbalanced dataset like EU patients)
    '''
    X = (X >= threshold)
    num = torch.sum(X == Y)
    return float(num / Y.shape[0])

def F1(y_pred, y_true, threshold=0.5):
    '''
    F1 score
    '''
    y_pred = y_pred > threshold
    f1 = F1_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())
    return f1

def precision(y_pred, y_true, threshold=0.5):
    '''
    precision
    '''
    y_pred = y_pred > threshold
    return precision_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())

def recall(y_pred, y_true, threshold=0.5):
    '''
    recall
    '''
    y_pred = y_pred > threshold
    return recall_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())

def auc(y_pred, y_true, threshold = None):
    '''
    AUC score
    '''
    fpr, tpr, thresholds = metrics.roc_curve(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), pos_label=1)
    return metrics.auc(fpr, tpr)

import matplotlib.pyplot as plt


def plot_AUC(y_pred, y_true):
    '''
    plotting the ROC curve
    '''
    fpr, tpr, threshold = metrics.roc_curve(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), pos_label=1)
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 10))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='AUC = {:.3f}'.format(metrics.auc(fpr, tpr)))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend()

def train_RNN(dm, sim_train, sim_test, parameters, acc_fn= F1, autostop_decay=0.995, print_summary=True, verbose=True):
    '''
    RNN training code
    '''
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    RNNmodel = RNN(sim_train.shape[1], 1, parameters['hidden_dim'], device = device).to(device)
    X_train = torch.from_numpy(sim_train).float().to(device)
    X_test = torch.from_numpy(sim_test).float().to(device)
    Y_train = torch.from_numpy(dm.Y_train).float().to(device)
    Y_test = torch.from_numpy(dm.Y_test).float().to(device)

    optimizer = torch.optim.Adam(RNNmodel.parameters(), lr=parameters['learning_rate'], betas=parameters['betas'],
                                 eps=parameters['eps'], weight_decay=parameters['weight_decay'], amsgrad=False)
    criterion = nn.BCELoss()
    # criterion = F1_Loss()

    max_v_a = 0
    bestmodel = None

    if print_summary:
        print(RNNmodel)
        summary(RNNmodel, (sim_train.shape[1], sim_train.shape[1]))

    n_epochs = parameters['num_epochs']
    batch_size = parameters['batch_size']

    # early stopping
    beta = autostop_decay
    epoch = 0
    V = 0
    while (True):
        # X is a torch Variable
        v_l = 0
        v_a = 0
        val_acc = 0
        val_loss = 0
        epoch += 1

        optimizer.zero_grad()
        RNNmodel.train()
        train_pred = RNNmodel(X_train)
        train_loss = criterion(train_pred, Y_train)
        train_loss.backward()
        optimizer.step()
        train_acc = acc_fn(train_pred, Y_train, threshold=0.5)

        # get val accuracy
        RNNmodel.eval()
        for i in range(5, 100, 5):
            t = i / 100
            val_pred = RNNmodel(X_test)
            tval_loss = criterion(val_pred, Y_test)
            tva = acc_fn(val_pred, Y_test, threshold=t)
            if tva > val_acc:
                val_acc = tva
                val_loss = tval_loss

        v_l += float(val_loss)
        v_a += float(val_acc)

        epoch_val_loss = v_l
        if epoch == 1:
            v = epoch_val_loss
        else:
            v = beta * v + (1 - beta) * epoch_val_loss

        if verbose:
            print("Epoch:", epoch, "  Train loss:", round(float(train_loss) , 4), "  Train accuracy:", round(float(train_acc) , 3),
                  "  Val loss:", round(v_l, 4), "  Val accuracy:", round(v_a, 3), "   weighted Val loss:",
                  round(v, 4))
        if v_a > max_v_a:
            max_v_a = v_a
            # bestmodel = MLP(sim_train.shape[1], parameters['n_layers'], parameters['layer_size_factor'], parameters['dropout']).to(device)
            # bestmodel.load_state_dict(copy.deepcopy(MLPmodel.state_dict()))
            bestmodel = copy.deepcopy(RNNmodel)
            checkpoint = {
                'parameters': parameters,
                'state_dict': bestmodel.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            print(round(max_v_a, 3), "----------saved-----------")
        if epoch > n_epochs:
            break

    return bestmodel, max_v_a, epoch, checkpoint


def eval_RNN(model, sim_test, dm, device_name='cpu', threshold=0.5, verbose=True):
    '''
    RNN evaluation script
    '''
    if device_name == 'cuda':
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print("device set to cuda") if device == torch.device('cuda') else print("cuda is not available")
    elif device_name == 'cpu':
        device = torch.device('cpu')
        print("device set to cpu")
    else:
        device = torch.device('cpu')
        print("unknown device")
    X_test = torch.from_numpy(sim_test).float().to(device)
    Y_test = torch.from_numpy(dm.Y_test).float().to(device)
    criterion = nn.BCELoss()
    model.eval()
    val_pred = model(X_test)
    val_loss = criterion(val_pred, Y_test)
    F1_acc = F1(val_pred, Y_test, threshold=threshold)
    p_acc = precision(val_pred, Y_test, threshold=threshold)
    r_acc = recall(val_pred, Y_test, threshold=threshold)
    # auc_acc = auc(val_pred, Y_test, threshold=threshold)
    auc_acc = auc(val_pred, Y_test)
    # auc_acc = auc_nf(val_pred, Y_test)
    if verbose:
        print("threshold:", threshold, " validation loss:", round(float(val_loss), 4), "F1 accuracy",
              round(float(F1_acc), 3), "Precision accuracy", round(float(p_acc), 3), "Recall accuracy",
              round(float(r_acc), 3), "AUC accuracy:", round(float(auc_acc), 3))
    return F1_acc


def eval_plot_RNN(model, sim_test, dm, device_name='cpu', verbose=True):
    '''
    plots ROC for RNN model
    '''
    if device_name == 'cuda':
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print("device set to cuda") if device == torch.device('cuda') else print("cuda is not available")
    elif device_name == 'cpu':
        device = torch.device('cpu')
        print("device set to cpu")
    else:
        device = torch.device('cpu')
        print("unknown device")
    X_test = torch.from_numpy(sim_test).float().to(device)
    Y_test = torch.from_numpy(dm.Y_test).float().to(device)
    criterion = nn.BCELoss()
    model.eval()
    val_pred = model(X_test)
    val_loss = criterion(val_pred, Y_test)
    plot_AUC(val_pred, Y_test)


























