import json
import numpy as np
import tensorflow as tf
from collections import namedtuple
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from graphsage.Graph_Learning_utils import Hybrid_Rep_Feat
from supervised_tasks import FFT, data_load
import gc



TaskCore = namedtuple('TaskCore', ['data_dir', 'target', 'classifier', 'normalize', 'cv_ratio'])

GraphLCore = namedtuple('GraphLCore', ['model_size', 'num_nodes', 'dim', 'fixed_params','aggregator_type',\
                                       'concat','num_layers', 'coordinate_gradient', 'projected_gradient',\
                                       'conv_sizes'])
ClassifCore = namedtuple('ClassifCore', ['num_samples','classifier','epochs','learning_rate', 'n_hidden_1','n_hidden_2', \
                                         'num_classes','max_total_steps','print_every','batch_size','loss_type', 'A_proj_th'])
LoadCore = namedtuple('ClassifCore', ['howmany_to_load', 'num_windows', 'sampling_freq', 'data_conversions',\
                                       'num_classes', 'train_test', 'start_num', 'concat', 'down_sampl_ratio',
                                       'freq_bands', 'initial_freq_band', 'only_seizures', 'welchs_win_len','welchs_stride'])

with open('SETTINGS.json') as f:
    settings = json.load(f)

data_dir = str(settings['data-dir'])
target = 'chb01'
cv_ratio = 0.5

def should_normalize(classifier):
    clazzes = [LogisticRegression]
    return np.any(np.array([isinstance(classifier, clazz) for clazz in clazzes]) == True)

classifier = RandomForestClassifier(n_estimators=3000, min_samples_split=1, bootstrap=False, n_jobs=4, random_state=0)
task_core = TaskCore(data_dir=data_dir, target=target, classifier=classifier, 
                     normalize=should_normalize(classifier), cv_ratio=cv_ratio)



train_num_sample = 8
test_num_sample = 4
weight_losses = 1
num_classes = 2
epochs = 10
learning_rate = 0.01
batch_size = 100

num_windows = None
down_sampl_ratio = None

welchs_win_len = 1
welchs_stride = 0.75

data_conversions = [FFT()] 
initial_freq_band = 1
freq_bands = [4,8,12,30,400] 
only_seizures=True
loss_type = 'softmax'

load_Core = LoadCore(howmany_to_load=train_num_sample, num_windows=num_windows, sampling_freq=256,\
                      data_conversions=data_conversions, down_sampl_ratio=down_sampl_ratio,\
                      num_classes=num_classes, start_num=0, train_test=False, concat=True, 
                      freq_bands=freq_bands, initial_freq_band=initial_freq_band, only_seizures=only_seizures,
                      welchs_win_len=welchs_win_len, welchs_stride=welchs_stride) 

X, y, num_nodes, dim, conv_sizes = data_load(task_core).run(load_Core)
graphL_core = GraphLCore(model_size='small', num_nodes=num_nodes, dim=dim, fixed_params=False, aggregator_type='mean',\
                        concat=False, num_layers=2, coordinate_gradient=False, projected_gradient=True, conv_sizes=conv_sizes)
classification_core = ClassifCore(num_samples=X.shape[0], classifier=task_core.classifier, \
                                  epochs=epochs, learning_rate=learning_rate,\
                                  n_hidden_1=10, n_hidden_2=10, num_classes=num_classes, max_total_steps=1e10, \
                                  print_every=10, batch_size=batch_size, loss_type=loss_type, A_proj_th=5)
feature_extraction = Hybrid_Rep_Feat(graphL_core, classification_core, weight_losses)
feature_extraction.train(X, y)
del X, y
gc.collect()


start_num = 0
load_Core = LoadCore(howmany_to_load=train_num_sample, num_windows=num_windows, sampling_freq=256,\
                      data_conversions=data_conversions, down_sampl_ratio=down_sampl_ratio,\
                      num_classes=num_classes, start_num=start_num, train_test=True, concat=False, 
                      freq_bands=freq_bands, initial_freq_band=initial_freq_band, only_seizures=only_seizures,
                      welchs_win_len=welchs_win_len, welchs_stride=welchs_stride) 
X, y, num_nodes, dim, conv_sizes = data_load(task_core).run(load_Core)
feature_extraction.test(X, y, show_plots=True, bias_name=start_num, training_samples='training')
del X, y
gc.collect()


start_num = train_num_sample
load_Core = LoadCore(howmany_to_load=test_num_sample, num_windows=num_windows, sampling_freq=256,\
                      data_conversions=data_conversions, down_sampl_ratio=down_sampl_ratio,\
                      num_classes=num_classes, start_num=start_num, train_test=True, concat=False, 
                      freq_bands=freq_bands, initial_freq_band=initial_freq_band, only_seizures=only_seizures,
                      welchs_win_len=welchs_win_len, welchs_stride=welchs_stride) 
X, y, num_nodes, dim, conv_sizes = data_load(task_core).run(load_Core)
feature_extraction.test(X, y, show_plots=True, bias_name=start_num, training_samples='testing')
del X, y
gc.collect()
    
    
    
    
    
    