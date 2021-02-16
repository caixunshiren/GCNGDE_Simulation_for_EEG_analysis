# In the name of God
import json
import numpy as np
import tensorflow as tf
from collections import namedtuple
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from supervised_tasks5 import train_test
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""


#np.set_printoptions(threshold=np.nan)
                                          
# TaskCore = namedtuple('TaskCore', ['data_dir', 'sidinfo_dir', 'settings_dir', 'target', 'cv_ratio', 'adj_calc_mode','feature_mode', 'TrainTest_mode', 'supervised'])
class TaskCore (object):
    def __init__(self, data_dir, sidinfo_dir, settings_dir, target, cv_ratio, adj_calc_mode, feature_mode, TrainTest_mode, supervised):
        self.data_dir = data_dir
        self.sidinfo_dir = sidinfo_dir
        self.settings_dir = settings_dir
        self.target = target
        self.cv_ratio = cv_ratio
        self.adj_calc_mode = adj_calc_mode
        self.feature_mode = feature_mode
        self.TrainTest_mode = TrainTest_mode
        self.supervised = supervised

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

def seizure_state_estimation(build_target):
    with open('SETTINGS.json') as f:
        settings = json.load(f)

    targets = [253, 264, 273, 375, 384, 548, 565, 583, 590, 620, 862, 916, 922, 958, 970, \
                        1084, 1096, 1146, 115, 139, 442, 635, 818, 1073, 1077, 1125, 1150, 732, 13089, 13245]
    targets = [620] # 
    cv_ratio = 0.5
    feature_mode = 'W_graphL_correlation'   #     'X_raw'    'W_raw_correlation' 'W_raw_coherence'   'W_graphL_correlation'    'W_graphL_coherence'    
    def should_normalize(classifier):
        clazzes = [LogisticRegression]
        return np.any(np.array([isinstance(classifier, clazz) for clazz in clazzes]) == True)
    
    def run_train_test():
        for target in targets:
#             try: 
                print('**************Patient ', target)
                task_core = TaskCore(data_dir=str(settings['data-dir-FFT']) if 'FFT' in feature_mode or ('coherence' in feature_mode ) else str(settings['data-dir-Raw']),\
                                     sidinfo_dir=str(settings['sidinfo-dir']), \
                                     settings_dir=str(settings['settings-dir-FFT'])  if 'FFT' in feature_mode or ('coherence' in feature_mode ) else str(settings['settings-dir-Raw']),\
                                     target=target, \
                                     cv_ratio=cv_ratio, adj_calc_mode=['invcov'], feature_mode=feature_mode, TrainTest_mode='50%-50%-coarse', supervised=False) 
                # '50%-50%-coarse', '24h-1h-coarse'
                # and not 'raw' in feature_mode
                
                train_test(task_core).run()
#             except Exception as e: 
#                 print(e)
#                 continue

    if build_target == 'train_test':
        run_train_test()
        
seizure_state_estimation('train_test')
        
        