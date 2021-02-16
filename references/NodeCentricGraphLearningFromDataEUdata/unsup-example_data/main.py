# In the name of God
import json
import numpy as np
import tensorflow as tf
from collections import namedtuple
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from supervised_tasks import train_test
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""


#np.set_printoptions(threshold=np.nan)
                                          
TaskCore = namedtuple('TaskCore', ['data_dir', 'sidinfo_dir', 'settings_dir', 'target', 'cv_ratio', 'adj_calc_mode','feature_mode', 'TrainTest_mode'])
                                          

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

def seizure_state_estimation(build_target):
    with open('SETTINGS.json') as f:
        settings = json.load(f)

    targets = [253, 264, 273, 375, 384, 548, 565,\
                    583, 590, 620, 862, 916, 922, 958, 970, 1084, 1096, 1146,\
                        115, 139, 442, 635, 818, 1073, 1077, 1125, 1150, 732, 13089, 13245]
    targets = [970] # , 273, 565, 620, 862, 958
    cv_ratio = 0.5
    
    def should_normalize(classifier):
        clazzes = [LogisticRegression]
        return np.any(np.array([isinstance(classifier, clazz) for clazz in clazzes]) == True)
    
    def run_train_test():
        for target in targets:
#             try:
                print('**************Patient ', target)
                task_core = TaskCore(data_dir=str(settings['data-dir']), sidinfo_dir=str(settings['sidinfo-dir']), settings_dir=str(settings['settings-dir']), target=target, \
                                        cv_ratio=cv_ratio, adj_calc_mode=['invcov'], feature_mode='W_graphL', TrainTest_mode='50%-50%') 
                # 'X_raw'    'W_raw'    'W_graphL'
                train_test(task_core).run()
#             except Exception as e: 
#                 print(e)
#                 continue

    if build_target == 'train_test':
        run_train_test()
        
seizure_state_estimation('train_test')
        
        