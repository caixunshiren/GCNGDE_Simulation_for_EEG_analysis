import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import copy


class MLP(nn.Module):
    def __init__(self, matrix_dim, n_layers=2, layer_size_factor=[1, 5], dropout=[-1, 0.5]):
        super(MLP, self).__init__()
        feature_len = torch.triu_indices(matrix_dim, matrix_dim).shape[1]
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if dropout[i] > 0:
                self.layers.append(nn.Dropout(dropout[i]))
            if i < n_layers - 1:
                self.layers.append(
                    nn.Linear(int(feature_len // layer_size_factor[i]), int(feature_len // layer_size_factor[i + 1])))
                self.layers.append(nn.ReLU())
            else:
                self.layers.append(nn.Linear(int(feature_len // layer_size_factor[i]), 1))
                self.layers.append(nn.Sigmoid())

    def flatten(self, sim_matrices):
        tri_indices = torch.triu_indices(sim_matrices.shape[1], sim_matrices.shape[2])
        return sim_matrices[:, tri_indices[0, :], tri_indices[1, :]]

    def forward(self, sim_matrices):
        x = self.flatten(sim_matrices)
        for layer in self.layers:
            x = layer(x)
        return x

# --- CrossBar Implementation --- #
class Batched_VMM(torch.autograd.Function):
    #Modified from Louis: Custom pytorch autograd function for crossbar VMM operation
    @staticmethod
    def forward(ctx, ticket, x, W, b):
        ctx.save_for_backward(x, W, b)
        return ticket.vmm(x) + b
        
    @staticmethod
    def backward(ctx, dx):
        x, W, b = ctx.saved_tensors
        grad_input = W.t().mm(dx)
        grad_weight = dx.mm(x.t())
        grad_bias = dx
        return (None, grad_input, grad_weight, grad_bias)

class Linear_block(nn.Module):
    def __init__(self, in_size, out_size, cb, w = None, b = None):
        if w is not None and b is not None:
            self.w = w
            self.b = b
        else:
            self.w = nn.Parameter(torch.rand(out_size, in_size))
            self.b = nn.Parameter(torch.rand(out_size, 1))
        self.cb = cb
        self.f = Batched_VMM

class MLPwCB(nn.Module):
    def __init__(self, matrix_dim, cb, weights=None, n_layers=2, layer_size_factor=[1, 5], dropout=[-1, 0.5]):
        super(MLP, self).__init__()
        feature_len = torch.triu_indices(matrix_dim, matrix_dim).shape[1]
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if dropout[i] > 0:
                self.layers.append(nn.Dropout(dropout[i]))
            if i < n_layers - 1:
                self.layers.append(
                    nn.Linear(int(feature_len // layer_size_factor[i]), int(feature_len // layer_size_factor[i + 1])))
                self.layers.append(nn.ReLU())
            else:
                self.layers.append(nn.Linear(int(feature_len // layer_size_factor[i]), 1))
                self.layers.append(nn.Sigmoid())

    def flatten(self, sim_matrices):
        tri_indices = torch.triu_indices(sim_matrices.shape[1], sim_matrices.shape[2])
        return sim_matrices[:, tri_indices[0, :], tri_indices[1, :]]

    def forward(self, sim_matrices):
        x = self.flatten(sim_matrices)
        for layer in self.layers:
            x = layer(x)
        return x
# --- --- --- --- --- --- --- --- --- --- #

class F1_Loss(nn.Module):
    '''Calculate F1 score. Can work with gpu tensors

    The original implmentation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. epsilon <= val <= 1

    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    - http://www.ryanzhang.info/python/writing-your-own-loss-function-module-for-pytorch/
    '''

    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        # assert y_pred.ndim == 2
        # assert y_true.ndim == 1
        # y_true = y_true.to(torch.float32)
        # y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1.mean()


from sklearn.metrics import f1_score as F1_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn import metrics

def accuracy(X, Y, threshold=0.5):
    X = (X >= threshold)
    num = torch.sum(X == Y)
    return float(num / Y.shape[0])


def F1(y_pred, y_true, threshold=0.5):
    y_pred = y_pred > threshold
    f1 = F1_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())

    return f1


def precision(y_pred, y_true, threshold=0.5):
    y_pred = y_pred > threshold
    return precision_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())


def recall(y_pred, y_true, threshold=0.5):
    y_pred = y_pred > threshold
    return recall_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())

def auc(y_pred, y_true, threshold = 0.5):
    y_pred = y_pred > threshold
    return roc_auc_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())

def auc2(y_pred, y_true):
    fpr, tpr, thresholds = metrics.roc_curve(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy(), pos_label=1)
    return metrics.auc(fpr, tpr)


def train_MLP(dm, sim_train, sim_test, parameters, acc_fn=F1, autostop_decay=0.995, print_summary=True, verbose=True):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    MLPmodel = MLP(sim_train.shape[1], parameters['n_layers'], parameters['layer_size_factor'],
                   parameters['dropout']).to(device)
    X_train = torch.from_numpy(sim_train).float().to(device)
    X_test = torch.from_numpy(sim_test).float().to(device)
    Y_train = torch.from_numpy(dm.Y_train).float().to(device)
    Y_test = torch.from_numpy(dm.Y_test).float().to(device)

    optimizer = torch.optim.Adam(MLPmodel.parameters(), lr=parameters['learning_rate'], betas=parameters['betas'],
                                 eps=parameters['eps'], weight_decay=parameters['weight_decay'], amsgrad=False)
    criterion = nn.BCELoss()
    # criterion = F1_Loss()

    max_v_a = 0
    bestmodel = None

    if print_summary:
        print(MLPmodel)
        summary(MLPmodel, (sim_train.shape[1], sim_train.shape[1]))

    n_epochs = parameters['num_epochs']
    batch_size = parameters['batch_size']

    # early stopping
    beta = autostop_decay
    epoch = 0
    V = 0
    while (True):
        # X is a torch Variable
        permutation = torch.randperm(X_train.shape[0])
        t_l = 0
        v_l = 0
        t_a = 0
        v_a = 0
        n_b = 0
        val_acc = 0
        epoch += 1

        for i in range(0, X_train.shape[0], batch_size):
            optimizer.zero_grad()

            indices = permutation[i:i + batch_size] if i + batch_size < X_train.shape[0] else permutation[i:]
            batch_x_train = X_train[indices, :, :]
            batch_y_train = Y_train[indices, :]

            MLPmodel.train()
            train_pred = MLPmodel(batch_x_train)
            train_loss = criterion(train_pred, batch_y_train)
            train_loss.backward()
            optimizer.step()
            train_acc = acc_fn(train_pred, batch_y_train, threshold=0.5)

            t_l += float(train_loss)
            t_a += float(train_acc)
            n_b += 1

        # get val accuracy
        MLPmodel.eval()
        for i in range(5, 100, 5):
            t = i / 100;
            val_pred = MLPmodel(X_test)
            val_loss = criterion(val_pred, Y_test)
            tva = acc_fn(val_pred, Y_test, threshold=t)
            if tva > val_acc:
                val_acc = tva

        v_l += float(val_loss)
        v_a += float(val_acc)

        epoch_val_loss = v_l
        if epoch == 1:
            v = epoch_val_loss
        else:
            v = beta * v + (1 - beta) * epoch_val_loss

        if verbose:
            print("Epoch:", epoch, "  Train loss:", round(t_l / n_b, 4), "  Train accuracy:", round(t_a / n_b, 3),
                  "  Val loss:", round(v_l, 4), "  Val accuracy:", round(v_a, 3), "   weighted Val loss:",
                  round(v, 4))
        if v_a > max_v_a:
            max_v_a = v_a
            # bestmodel = MLP(sim_train.shape[1], parameters['n_layers'], parameters['layer_size_factor'], parameters['dropout']).to(device)
            # bestmodel.load_state_dict(copy.deepcopy(MLPmodel.state_dict()))
            bestmodel = copy.deepcopy(MLPmodel)
            checkpoint = {
                'parameters': parameters,
                'state_dict': bestmodel.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            print(round(max_v_a, 3), "----------saved-----------")
        if epoch_val_loss > v and epoch > 60:
            break


    
    return bestmodel, max_v_a, epoch, checkpoint




def eval_mlp(model, sim_test, dm, device_name ='cpu', threshold = 0.5, verbose = True):
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
    #auc_acc = auc(val_pred, Y_test, threshold=threshold)
    auc_acc = auc2(val_pred, Y_test)
    if verbose:
        print("threshold:", threshold," validation loss:",round(float(val_loss), 4),"F1 accuracy", round(float(F1_acc), 3), "Precision accuracy", round(float(p_acc), 3), "Recall accuracy", round(float(r_acc), 3), "AUC accuracy:", round(float(auc_acc), 3))
    return F1_acc

def save_ckp(state, f_path):
    torch.save(state, f_path)
    print("model saved")

def load_ckp(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    return checkpoint

def load_model(checkpoint, A, device_name ='cpu' ):
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
    model = MLP(A.shape[1], parameters['n_layers'], parameters['layer_size_factor'],
                   parameters['dropout']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=parameters['learning_rate'], betas=parameters['betas'],
                                 eps=parameters['eps'], weight_decay=parameters['weight_decay'], amsgrad=False)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer

def pca():
    pass