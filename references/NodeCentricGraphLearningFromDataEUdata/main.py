import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

""" 
Main Function: includes the whole process, i.e. runs training and testing separately on 
1. graph neural network, and similarity matrix
2. classification of samples of similarity matrix for seizure detection
"""

                                    
"""
all_targets:   shows the number associated with the data of each patient 

all_feature_modes:  
            shows all types of features that can be selected as the input-output of the graph-learning process to be fed to the classifier
"""      
all_targets = [253, 264, 273, 375, 384, 548, 565, 583, 590, 620, 862, 916, 922, 958, 970, \
                    1084, 1096, 1146, 115, 139, 442, 635, 818, 1073, 1077, 1125, 1150, 732, 13089, 13245]
all_feature_modes = [ 'X_raw', 'W_raw_correlation', 'W_raw_coherence', 'W_graphL_correlation', 'W_graphL_coherence']  


class TaskCore(object):
    """
     A class to save the hyper-parameters of the GNN, graph learning, and classification tasks:
     data_dir : directory where data is read (i.e. input features to GNN)
     sidinfo_dir : directory where side information to input features are read
     settings_dir :  directory where input features' settings are read
     target :  refers to the patient that our algorithms is applied to
     cv_ratio:       the type of feature as the input-output of the graph-learning process to be fed to the classifier
     valid_split_ratio: ratio to split the training data to training and validation sets
     adj_calc_mode: 
     feature_mode: 
     TrainTest_mode:
     supervised: default is False for disjointed training of [GNN+similarity matrix generation] and the [classifier] 
    """
    def __init__(self, data_dir, sidinfo_dir, settings_dir, target, adj_calc_mode, feature_mode, TrainTest_mode, supervised=False):
        self.data_dir = data_dir
        self.sidinfo_dir = sidinfo_dir
        self.settings_dir = settings_dir
        self.target = target
        self.cv_ratio = 0.5
        self.adj_calc_mode = adj_calc_mode
        self.feature_mode = feature_mode
        self.TrainTest_mode = TrainTest_mode
        self.supervised = supervised


        # Training and Validation
        self.sensitivity_score_target = [1, 0.92, 0.85]
        self.valid_split_ratio = 5
        self.weight_losses = 1
        
        
        # Hyperparameters
        self.epochs_arr = [1]
        self.A_density_ratio_arr = [0.7]
        self.learning_rate_arr = [0.1] #, 0.001]
        self.num_trees_arr = [1000] #, 100, 5000]
        self.eval_thresh_func_arr = [np.mean]
        self.batch_size_arr = [200]
        self.aggregator_type_arr = ['mean'] # 'mean', 'maxpool'
        self.A_val_string_arr = ['fixedA_invCov'] # 'optimizedA', 'fixedA_invCov'
        self.mvAvg_winlen = 3 # sec2win(2)
        self.multiClass_ratio_arr = [-1]
        self.feature_normalize_arr = [ False]
        self.varType_arr = [['scalar', 'full']] # similarity, graphL
        self.band_nums = None # [1, 3]         # 'diag_repeated', 'full_repeated', 'full', 'scalar'
        self.A_regularizer_arr = [None] # None or weight of regularization
        self.extra_feat_functions = None #   self.freq_bands = [[0.1, 4], [4, 8], [8, 13], [13, 30], [30, 50], [70, 100]]
        self.freq_bands = [[14, 25]]     # 253: 3 1125: 5


"""
SETTINGS.json:  contains the directories for data, its settings, and the side information

targets:        refers to patients that our algorithms is applied to

feature_mode:   what features to be fed to the classifier,  
                            'X_raw'-> raw input signal
                            'W_raw_correlation', 'W_raw_coherence' -> direct correlation and coherence matrices calculated from raw signal
                            'W_graphL_correlation' -> similarity matrix that the graph learning procedure outputs, in time-domain
                            'W_graphL_coherence' -> similarity matrix that the graph learning procedure outputs, in frequency-domain
adj_calc_mode:  the method to calculate the initial adjacency matrix that constructs the foundation of GNN
TrainTest_mode: the mode to train and test, default is '50%-50%-coarse'
                    Training and testing of '50%-50%-coarse' selects all the seizure samples, 
                                              and randomly select 10X from non-seizure samples. 
                                              all samples are shuffled.
                                              all training samples are chosen chronologically prior to all the testing samples
                    
                    ----------
                    Training and testing of '24h-1h-coarse' is in real-time mode, i.e. samples correspond to consecutive windows. 
                    Using a 1-hour sliding window, the model is trained on each 24 hours and tested on the next 1 hour. 
"""

targets= [620]
feature_mode = 'W_graphL_correlation'
adj_calc_mode=['invcov']
TrainTest_mode = '50%-50%-coarse'

with open('SETTINGS.json') as f:
    settings = json.load(f)
    
for target in targets:
    print('**************Patient ', target)
    task_core = TaskCore(
         data_dir=str(settings['data-dir-FFT']) if 'FFT' in feature_mode or ('coherence' in feature_mode ) 
                                                else str(settings['data-dir-Raw']),\
         sidinfo_dir=str(settings['sidinfo-dir']), \
         settings_dir=str(settings['settings-dir-FFT'])  if 'FFT' in feature_mode or ('coherence' in feature_mode ) 
                                                        else str(settings['settings-dir-Raw']),\
         target=target, adj_calc_mode=adj_calc_mode, feature_mode=feature_mode, \
         TrainTest_mode=TrainTest_mode)   
    train_test(task_core).run()


