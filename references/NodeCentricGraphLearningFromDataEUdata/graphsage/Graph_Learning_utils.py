import tensorflow as tf
import common.time as time
import numpy as np
from graphsage.models import SampleAndAggregate
# from __future__ import division
# from __future__ import print_function
import os
import common.time
from graphsage.models import SampleAndAggregate, SAGEInfo, Node2VecModel
from graphsage.minibatch import EdgeMinibatchIterator
from graphsage.neigh_samplers import UniformNeighborSampler
from graphsage.utils import load_data
from graphsage.utils import normal_SDP_solver, Quadratic_SDP_solver, Graph_complement
import matplotlib.pyplot as plt
import networkx as nx
from .inits import glorot, zeros, repeated_variable, repeated2D_variable
from .aggregators import MeanAggregator, MaxPoolingAggregator, MeanPoolingAggregator, SeqAggregator, GCNAggregator
import sklearn
from sklearn.metrics import roc_curve, auc
import scipy
from mpl_toolkits.axes_grid1 import AxesGrid
import random
# from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources
from utils import NN_classifier, RandomForest_classifier, upper_triangle, class_relabel, sec2win, win2sec, inpython_online_wrapper, half1D_to_2D, network_stats_calc
# import matlab.engine
import h5py
# from tensorflow.contrib.tensor_forest.client import eval_metrics
from sklearn.utils.class_weight import compute_sample_weight
import pandas
# from tensorflow.contrib.distributions import fill_triangular
from scipy import stats
import scipy.io as io

# import theano
# Settings
flags = tf.compat.v1.flags # tf.app.flags
FLAGS = flags.FLAGS

def plotting_figure(dir, list_arr, title, x_array=None, show_flag=False):
    fig = plt.figure(num=None, figsize=(60, 40), dpi=120)
    for arr in list_arr:
        if(x_array is None):
            x_array = np.arange(arr.size)
        if not os.path.exists(dir):
                os.makedirs(dir)
        plt.plot(x_array, arr)
    fig.tight_layout()
    plt.savefig(dir + title + '.png')
    if(show_flag):
        plt.show()

def plotting_box(dir, data, x_array=None, title='box_plot', show_flag=False):
    if not os.path.exists(dir):
            os.makedirs(dir)
    if(x_array is None):
        x_array = np.arange(len(data))
    counter = 0
    for arr in data:
        plt.figure()
        plt.boxplot(arr)
        plt.savefig(dir + title + str(x_array[counter]) + '.png')
        counter += 1
    if(show_flag):
        plt.show()        
        
class online_miniBatchIterator():
    def __init__(self, matlab_load_core, placeholders, task_core, load_Core=None, graphL_minibatch=None, batch_size=None, clip_sizes=None):
        self.batch_num = matlab_load_core.settings_TrainNumFiles 
        self.batch_size = batch_size
        self.placeholders = placeholders
        self.graphL_minibatch = graphL_minibatch
        self.matlab_load_core = matlab_load_core 
        self.clip_sizes = clip_sizes 
        self.matlab_engin = matlab.engine.start_matlab()
        self.task_core = task_core
        self.load_Core = load_Core
    
    def current_x(self):
        return self.dataX
    
    def current_y(self):
        return self.dataY
    
    
    def feed_dict_update(self):
        feed_dict = {self.placeholders['X']: self.dataX, self.placeholders['Y']: self.dataY}
        if(self.graphL_minibatch is not None):
            feed_dict.update(self.graphL_minibatch.next_minibatch_feed_dict())
        return feed_dict
        
        
    def next(self):
        self.batch_num += 1
        self.dataX, self.dataY, sel_win_nums, conv_sizes, clip_sizes = \
                    inpython_online_wrapper(self.matlab_engin, self.matlab_load_core, [self.batch_num], 'total', self.task_core.data_dir, self.load_Core)
#         X, Y, conv_sizes, sel_win_nums, clip_sizes = self.matlab_engin.python_online_wrapper(self.matlab_load_core.target, [self.batch_num], 'total', nargout=5)
#         self.dataX = np.asarray(X)
#         self.dataY = class_relabel(np.reshape(np.asarray(Y),[-1]))
#         sel_win_nums = np.reshape(np.asarray(sel_win_nums),[-1])
#         conv_sizes = np.reshape(np.asarray(conv_sizes),[-1])
#         clip_sizes = np.asarray(clip_sizes)
        return self.feed_dict_update()
    
    
    def end(self):
        return self.batch_num >= self.matlab_load_core.settings_TrainNumFiles + self.matlab_load_core.settings_TestNumFiles
        
        
        
class miniBatchIterator():
    def __init__(self, graphL_minibatch, batch_size, placeholders, dataX, dataY, clip_sizes=None):
        self.batch_num = 0
        self.batch_size = batch_size
        self.dataX = dataX
        self.dataY = dataY
        self.placeholders = placeholders
        self.graphL_minibatch = graphL_minibatch
        self.num_samples = self.dataX.shape[0]
        self.clip_sizes = clip_sizes
    
    
    def current_x(self):
        return self.dataX[self.start_idx:self.end_idx,...]
    
    def current_y(self):
        return self.dataY[self.start_idx:self.end_idx]
    
    def current_idx(self):
        return self.start_idx, self.end_idx
    

    def feed_dict_update(self):
        feed_dict = {self.placeholders['X']: self.dataX[self.start_idx:self.end_idx,...],
                     self.placeholders['Y']: self.dataY[self.start_idx:self.end_idx]}
        if(self.graphL_minibatch is not None):
            feed_dict.update(self.graphL_minibatch.next_minibatch_feed_dict())
        return feed_dict
        
        
    def next(self):
        if(self.clip_sizes is None):
            if(self.num_samples>1):
                self.start_idx = self.batch_num * self.batch_size
                self.batch_num += 1
                self.end_idx = self.start_idx + self.batch_size
                if(self.end_idx>self.num_samples):
                    self.start_idx = np.max((0, self.num_samples - self.batch_size))
                    self.end_idx = self.num_samples
            else:
                self.start_idx = 0
                self.end_idx = self.num_samples
        else:
            self.start_idx =  int(self.clip_sizes[self.batch_num, 0]) 
            self.end_idx = int(self.clip_sizes[self.batch_num, 1]) 
            self.batch_num += 1

        return self.feed_dict_update()
    
    
    def shuffle(self):
        if(self.num_samples>1):
            self.dataX, self.dataY = sklearn.utils.shuffle(self.dataX, self.dataY, random_state=0)
        self.batch_num = 0
        
    def end(self):
        if(self.clip_sizes is None):
            return self.batch_num * self.batch_size >= self.num_samples
        else:
            return self.batch_num >= self.clip_sizes.shape[1]
    
 
def flatten(data):
#     Largest_num = 100000
#     data[np.isneginf(data)] = - Largest_num
#     data[np.isnan(data)] = 0
#     data[np.isinf(data)]= Largest_num
#     if data.ndim > 2:
#         return data.reshape((data.shape[0], np.product(data.shape[1:])))
#     else:
#         return data
    return tf.reshape(data, [-1])
       
def construct_placeholders(dimArray=None, num_classes=None):
    placeholders = {
        'X' : tf.placeholder(tf.complex64, shape=(None, None, dimArray[0], dimArray[1]), name='X'),
        'Y' : tf.placeholder(tf.int32, shape=(None,), name='Y'), 
        'weight_losses' : tf.placeholder(tf.float32, shape=(), name='weight_losses'),
#         'num_to_load' : tf.placeholder(tf.int32, shape=(), name='num_to_load'),
    }
    return placeholders




     
class Hybrid_Rep_Feat():
    """
        Class comprised of the functions implementing GNN and similarity matrix generation (feature extraction)
    """
    def __init__(self, graphL_core, classif_core, matlab_load_core, task_core, weight_losses):
        self.task_core = task_core
        self.graphL_core = graphL_core
        self.classif_core = classif_core
        self.matlab_load_core = matlab_load_core
#         self.weight_losses = weight_losses
        self.aggregators = None
        self.graphL_minibatch = None
        if self.graphL_core.aggregator_type == "mean":
            self.aggregator_cls = MeanAggregator
        elif self.graphL_core.aggregator_type == "seq":
            self.aggregator_cls = SeqAggregator
        elif self.graphL_core.aggregator_type == "maxpool":
            self.aggregator_cls = MaxPoolingAggregator
        elif self.graphL_core.aggregator_type == "meanpool":
            self.aggregator_cls = MeanPoolingAggregator
        elif self.graphL_core.aggregator_type == "gcn":
            self.aggregator_cls = GCNAggregator
        self.build()
        
    def build(self):
        self.placeholders = construct_placeholders(dimArray=self.graphL_core.dimArray, num_classes=self.classif_core.num_classes) # num_samples=self.classif_core.num_samples
        self._graphL()
        self.optimizer = tf.train.AdamOptimizer(self.classif_core.learning_rate) #  GradientDescentOptimizer, AdamOptimizer
        self._loss()
        self.min_loss()
        if(self.graphL_core.side_adj_mat is None):
            self.project_GD()
            
        
    
    def aggregate(self, hidden):
#         hidden = hidden if hidden is not None else [input_features for layer in range(self.graphL_core.num_layers+1)] # [tf.nn.embedding_lookup(input_features, node_samples) for node_samples in samples]
        new_agg = self.aggregators is None
        if new_agg:
            self.aggregators = []
        windSize = self.graphL_core.dimArray[-1]
        if(np.sum(self.graphL_core.conv_sizes)==self.graphL_core.dimArray[-1]):
            windSize = self.graphL_core.dimArray[-2]
        for layer in range(self.graphL_core.num_layers):
            
            if(new_agg):
                dim_mult = 2 if self.graphL_core.concat and (layer != 0) else 1
                # aggregator at current layer
                if layer == self.graphL_core.num_layers - 1:
                    aggregator = self.aggregator_cls(dim_mult*self.graphL_core.dim, self.graphL_core.dim, 
                                                     (self.adj_mat+1)/2, self.graphL_core.conv_sizes*windSize, act=lambda x : x,
                                                        concat=self.graphL_core.concat, model_size=self.graphL_core.model_size, 
                                                            variables=self.variables[layer], bias=self.biasVar[layer]) # !!!!!!!!!!change later: 
                else:
                    aggregator = self.aggregator_cls(dim_mult*self.graphL_core.dim, self.graphL_core.dim, 
                                                     (self.adj_mat+1)/2, self.graphL_core.conv_sizes*windSize, act=lambda x : x,
                                                        concat=self.graphL_core.concat, model_size=self.graphL_core.model_size, 
                                                            variables=self.variables[layer], bias=self.biasVar[layer]) # !!!!!!!!!!change later: 
                self.aggregators.append(aggregator)
            else:
                aggregator = self.aggregators[layer]
            # hidden representation at current layer for all support nodes that are various hops away
            next_hidden = []
            # as layer increases, the number of support nodes needed decreases
            for hop in range(self.graphL_core.num_layers - layer):
                dim_mult = 2 if self.graphL_core.concat and (layer != 0) else 1
                self_vecs = hidden[hop]
                neigh_vecs = hidden[hop+1] 
#                 tf.py_func( np.repeat, [input_features[ np.newaxis, :, :], self.graphL_core.num_nodes, 0], tf.complex64) 
                #[tf.where(self.adj_mat[ii,:]!=0, input_features, [])  for ii in range(self.graphL_core.num_nodes)]#  hidden[hop+1] # wrongggggggggggggggggg
                h = aggregator((self_vecs, neigh_vecs))
#                 h = tf.Print(h, [h], "h after aggregation: ")
#                 self.adj_mat = tf.Print(self.adj_mat, [self.adj_mat], "A after aggregation: ")
                next_hidden.append(h)
            hidden = next_hidden
        Z = hidden[0]
        return Z

         
    def Feature2Weight(self, arr, theta_array, Theta_mat=None):
        if(Theta_mat is None):
            Theta_mat = tf.linalg.tensor_diag(theta_array)
        if(not np.any(np.array(self.graphL_core.dimArray)==1)):
            arr = tf.transpose(arr, [1,0,2])
#             poww = tf.sqrt(tf.reduce_sum(tf.abs(arr)**2, -1))
#             poww = tf.linalg.matmul(tf.expand_dims(poww, 2), tf.expand_dims(poww, 1))
#             norm_fac = tf.tile (tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.abs(arr)**2, -1)), 2), [1, 1, self.graphL_core.dimArray[-1]])
            arr_real = tf.real(arr)#/norm_fac
            arr_imag = tf.imag(arr)#/norm_fac
    #         W_complex = tf.complex( tf.matmul(Z_real, tf.matmul(self.Theta, tf.transpose(Z_real))) - \
    #                                         tf.matmul(Z_imag, tf.matmul(self.Theta, tf.transpose(Z_imag))), 
    #                                             tf.matmul(Z_real, tf.matmul(self.Theta, tf.transpose(Z_imag))) + \
    #                                                 tf.matmul(Z_imag, tf.matmul(self.Theta, tf.transpose(Z_real)))  )
    #         W = tf.abs(W_complex)
            W = tf.complex( tf.linalg.matmul(arr_real, tf.transpose(arr_real, [0,2,1])) \
                                - tf.linalg.matmul(arr_imag, tf.transpose(arr_imag, [0,2,1])), 
                                    tf.linalg.matmul(arr_real, tf.transpose(arr_imag, [0,2,1])) \
                                        + tf.linalg.matmul(arr_imag, tf.transpose(arr_real, [0,2,1]))  )
#             W = tf.truediv(tf.abs(W), poww) # tf.abs(W) # , poww
            W = tf.abs(W) # tf.div_no_nan(tf.abs(W), poww)
            params = tf.tile (tf.expand_dims(tf.expand_dims(theta_array, 1), 2) , [1, self.graphL_core.num_nodes, self.graphL_core.num_nodes])
            W = tf.reduce_sum(tf.multiply(params, W),0)
#             W = tf.reduce_sum(W, 0)
        else:
            arr = tf.squeeze(arr)
            arr_real = tf.real(arr)
            arr_imag = tf.imag(arr)
            arr_real = tf.math.l2_normalize(arr_real-tf.expand_dims(tf.reduce_mean(arr_real, 1), 1), axis=1)
            W = tf.matmul(arr_real, tf.matmul(Theta_mat, tf.transpose(arr_real)))
            
        

        if(self.classif_core.feature_normalize):
            W = W/tf.sqrt(tf.reduce_sum(W**2))
        return W
        
    def _graphL(self):
        if(self.graphL_core.side_adj_mat is None):
#             A_initial = tf.random_uniform((int(self.graphL_core.num_nodes*( self.graphL_core.num_nodes+1)/2),))
#             A_flat_var = tf.Variable(A_initial, name='adjacency_matrix')
# #             self.adj_mat = tf_half1D_to_2D(A_flat_var, self.graphL_core.num_nodes)
#             
#             A_initial = tf.where(A_initial>=0.5, 1.0 * tf.ones((self.graphL_core.num_nodes, self.graphL_core.num_nodes)),
#                                       -1.0 * tf.ones((self.graphL_core.num_nodes, self.graphL_core.num_nodes)))
#             A_initial = tf.matrix_set_diag(A_initial, 1.0 * tf.ones((self.graphL_core.num_nodes,))) #[np.diag_indices(self.graphL_core.num_nodes)] = 1
            A_initial = tf.convert_to_tensor((2*self.matlab_load_core.structural_inf.adj_means[0]-1), dtype=tf.float32)
            self.adj_mat = tf.Variable(A_initial, name='adjacency_matrix')
        
        else:
            self.adj_mat = tf.Variable(self.graphL_core.side_adj_mat, dtype=tf.float32, trainable=False) # tf.convert_to_tensor(self.graphL_core.side_adj_mat, dtype=tf.float32)
            print('A inside graphL', self.graphL_core.side_adj_mat)
            
            
        self.variables = []
        self.biasVar = []
        for layer in range(self.graphL_core.num_layers):
            if(not self.graphL_core.fixed_params):
                varName = 'GraphLVars_layer'+str(layer)
                windSize = self.graphL_core.dimArray[-1]
                if(np.sum(self.graphL_core.conv_sizes)==self.graphL_core.dimArray[-1]):
                    windSize = self.graphL_core.dimArray[-2]
                if('diag_repeated' in self.graphL_core.varType[1]):
                    varrr = tf.linalg.tensor_diag(repeated_variable(self.graphL_core.conv_sizes*windSize, name=varName))
                    biasVarrr = repeated_variable(self.graphL_core.conv_sizes*windSize, name='bias_layer'+str(layer))
                elif('full_repeated' in self.graphL_core.varType[1]):
                    varrr = repeated2D_variable(self.graphL_core.conv_sizes*windSize, name=varName)
                    biasVarrr = repeated_variable(self.graphL_core.conv_sizes*windSize, name='bias_layer'+str(layer))
                elif('full' in self.graphL_core.varType[1]):
                    varrr = glorot([self.graphL_core.dim, self.graphL_core.dim], name=varName)
                    biasVarrr = glorot([self.graphL_core.dim, ], name='bias_layer'+str(layer))
                elif('scalar' in self.graphL_core.varType[1]):
                    varrr = glorot([1,], name=varName) * tf.ones((self.graphL_core.dim, self.graphL_core.dim))
                    biasVarrr = glorot([1,], name='bias_layer'+str(layer)) * tf.ones((self.graphL_core.dim, ))
                    
                self.variables.append(varrr) # 
                self.biasVar.append(biasVarrr)
#                 break # !!!!!!!!!!change later
            else:
                self.variables.append(self.graphL_core.fixed_neigh_weights)   
#         self.adj_info = glorot([self.graphL_core.num_nodes, self.graphL_core.num_nodes], name='adj_info')             
#         sampler = UniformNeighborSampler(self.adj_info)
#         self.layer_infos = [SAGEInfo("node", sampler, self.graphL_core.num_nodes, self.graphL_core.dim, self.graphL_core.neigh_num),
#                             SAGEInfo("node", sampler, self.graphL_core.num_nodes, self.graphL_core.dim, self.graphL_core.neigh_num)]
        
        
#         self.vars_Theta_weights = tf.nn.softmax(self.vars_Theta_weights) # /tf.reduce_sum(self.vars_Theta_weights)
#         list_matrices = [np.diag(np.ones((self.graphL_core.conv_sizes[i],))) * self.vars_Theta_weights[i]\
#                          for i in np.arange(self.graphL_core.conv_sizes.size)]
#         block_theta = scipy.linalg.block_diag(*list_matrices)
#         self.Theta = tf.cast(scipy.linalg.block_diag(block_theta, block_theta), tf.float32) #, tf.reshape((np.sum(self.graphL_core.conv_sizes)))
        if('repeated' in self.graphL_core.varType[0]):
            self.theta_array1 = repeated_variable(self.graphL_core.conv_sizes, ffunc=tf.nn.softmax , name='Theta_Vars') # np.concatenate((self.graphL_core.conv_sizes, self.graphL_core.conv_sizes), 0)
            self.theta_array2 = repeated_variable(self.graphL_core.conv_sizes, ffunc=tf.nn.softmax , name='Theta_Vars')
        elif('full' in self.graphL_core.varType[0]):
            self.theta_array1 = tf.nn.softmax(glorot([self.graphL_core.dimArray[0],], name=varName))
            self.theta_array2 = tf.nn.softmax(glorot([self.graphL_core.dimArray[0],], name=varName))
        elif('scalar' in self.graphL_core.varType[0]):
            self.theta_array1 = tf.ones((self.graphL_core.dimArray[0],)) # * glorot([1,])
            self.theta_array2 = glorot([1,]) * tf.ones((self.graphL_core.dimArray[0],))
#         repeated = tf.concat([repeated,repeated], 0)
#         self.Theta1 = tf.linalg.tensor_diag(self.theta_array1) # tf.linalg.tensor_diag(repeated)
#         self.Theta2 = tf.linalg.tensor_diag(self.theta_array2)
#         self.Z = []
#         losses = []
#         self.graphL_W = [] 
#         self.graphL_inner_sums = 0
#         def fn(X):
#             X = tf.squeeze(X)
#             inner_Z = self.aggregate(X) # samples
#             inner_Z = tf.concat((X, inner_Z), axis=-1)
#             self.Z.append(inner_Z)
#             inner_Z_real = tf.real(inner_Z)
#             inner_Z_imag = tf.imag(inner_Z)
#             W_in_complex = tf.complex( tf.matmul(inner_Z_real, tf.matmul(self.Theta, tf.transpose(inner_Z_real))) - \
#                                             tf.matmul(inner_Z_imag, tf.matmul(self.Theta, tf.transpose(inner_Z_imag))), 
#                                                 tf.matmul(inner_Z_real, tf.matmul(self.Theta, tf.transpose(inner_Z_imag))) + \
#                                                     tf.matmul(inner_Z_imag, tf.matmul(self.Theta, tf.transpose(inner_Z_real)))  )
#             W_in = tf.abs(W_in_complex)
#             self.graphL_W.append(tf.reshape(W_in,(self.graphL_core.num_nodes**2,)))
#             multiply = tf.constant([self.graphL_core.num_nodes])
#             vec = tf.reduce_logsumexp(W_in, axis=1)
#             inner_tiled = tf.reshape(tf.tile(vec, multiply), [multiply[0], tf.shape(vec)[0]])
#             inner_sum = W_in-inner_tiled
#             self.graphL_inner_sums += inner_sum
#             inner_mult = tf.multiply(inner_sum, (self.adj_mat+1)/2)
#             inner_loss = tf.reduce_mean(tf.reduce_mean(inner_mult))
#             losses.append(inner_loss)
#             
#         tf.map_fn(fn, self.placeholders['X'])
               
        def body(i_count, Z, graphL_W, losses):
            X = tf.squeeze(self.placeholders['X'][i_count,...])
            X_flattend = tf.reshape(X, [self.graphL_core.num_nodes,-1])
            samples = self.sample(X_flattend)
            pre_Z = self.aggregate(samples) 
            inner_Z = tf.concat([X_flattend, pre_Z], axis=-1)
            Z = tf.concat([Z, inner_Z[None,...]], axis=0)
            W_in = self.Feature2Weight(tf.reshape(pre_Z, np.concatenate((np.array([self.graphL_core.num_nodes]),\
                                        self.graphL_core.dimArray))), theta_array=self.theta_array1, Theta_mat=None) +\
                                            self.Feature2Weight(X, theta_array=self.theta_array2, Theta_mat=None) # inner_Z.reshape(X.shape[:-1] + X.shape[-1]*2 )
            
            graphL_W = tf.concat([graphL_W, tf.reshape(upper_triangle(W_in),(self.classif_core.num_features,))[None,:]],0)        
            multiply = tf.constant([self.graphL_core.num_nodes])
            vec = tf.reduce_logsumexp(W_in, axis=1)
            inner_tiled = tf.reshape(tf.tile(vec, multiply), [multiply[0], tf.shape(vec)[0]])
            inner_sum = W_in-inner_tiled
            inner_mult = tf.multiply(inner_sum, (self.adj_mat+1)/2)
#             self.adj_mat = tf.Print(self.adj_mat, [self.adj_mat], "A after inner multiply in loss: ")
            inner_loss = tf.reduce_mean(tf.reduce_mean(inner_mult))
            losses = tf.concat([losses, [inner_loss]], 0)
            return tf.add(i_count, 1), Z, graphL_W, losses
         
        i_count = tf.constant(0)
        Z = tf.Variable(np.empty([0,self.graphL_core.num_nodes,self.graphL_core.dim*2]), dtype=tf.complex64, trainable=False)
        graphL_W = tf.Variable(np.empty([0, self.classif_core.num_features]), dtype=tf.float32, trainable=False) 
        losses = tf.Variable([], dtype=tf.float32, trainable=False)
         
        while_condition = lambda i_count, Z, graphL_W, losses: tf.less(i_count, tf.shape(self.placeholders['X'])[0])
        _, self.Z, self.graphL_W, losses = tf.while_loop(while_condition, body, \
                                                            [i_count, Z, graphL_W, losses],\
                                                                shape_invariants=[i_count.get_shape(), \
                                                                    tf.TensorShape([None,self.graphL_core.num_nodes,self.graphL_core.dim*2]), \
                                                                        tf.TensorShape([None, self.classif_core.num_features]),\
                                                                            tf.TensorShape([None])])    
        self.loss_graphL = -tf.reduce_mean(losses)
        if(self.graphL_core.A_regularizer is not None):
            self.loss_graphL += self.graphL_core.A_regularizer*tf.norm((self.adj_mat+1)/2)
#         if(self.graphL_core.Theta_regularizing is not None):
#             self.loss_graphL += self.graphL_core.Theta_regularizing*tf.norm(self.Theta)
    
    
    def sample(self, inputs):
        samples = [inputs]
        # size of convolution support at each layer per node
#         support_size = 1
#         support_sizes = [support_size]
        for k in range(self.graphL_core.num_layers):
            t = self.graphL_core.num_layers - k - 1
            
#             support_size *= layer_infos[t].num_samples
#             sampler = layer_infos[t].neigh_sampler
#             node = sampler(( , layer_infos[t].num_samples))
            
            new_samples = tf.tile (tf.expand_dims(samples[k], 0) , [self.graphL_core.num_nodes, 1, 1])
             
            samples.append(new_samples) 
#             support_sizes.append(support_size)
        return samples #, support_sizes
    
    def Extra_Features(self):
        features = []
        X = self.placeholders['X']
        for func in self.classif_core.extra_feat_functions:
            features.append(func.apply(X, self.task_core.target))
        
        return features
        
        
    def _loss(self):
        # tf.reshape(,(self.graphL_core.num_nodes*(self.graphL_core.num_nodes+1)/2,))
        # [self.upper_triangle(self.graphL_W[i]) for i in range(len(self.graphL_W))]
        features = tf.convert_to_tensor(self.graphL_W, dtype=tf.float32) # tf.reshape(self.graphL_W,(self.graphL_core.num_nodes*self.graphL_core.num_nodes,1))
        if(self.classif_core.extra_feat_functions is not None):
            features = tf.concat([features, self.Extra_Feature()], 0)
        labels = self.placeholders['Y']
        classifier = self.classif_core.classifier
        self.loss_class, self.pred_classes, self.pred_probas, self.train_op_class, self.loss_op_class =\
                    classifier(features, labels, self.classif_core)        
#         self.classif_core.first_try = False
        self.loss = self.loss_graphL 
        if(self.task_core.supervised):
            self.loss = self.loss * self.placeholders['weight_losses'] + self.loss_class
        self.loss = self.loss / tf.cast(self.classif_core.batch_size, tf.float32)
    
    
    def min_loss(self):
                
        self.opt_op = self.optimizer.minimize(self.loss)
        
    
    def adj_mat_coordinate_descent(self):       
        self.adj_mat = tf.where(self.graphL_inner_sums>=0, tf.ones((self.graphL_core.num_nodes, self.graphL_core.num_nodes)),
                                  -tf.ones((self.graphL_core.num_nodes, self.graphL_core.num_nodes)))
        self.adj_mat = tf.matrix_set_diag(self.adj_mat, tf.ones((self.graphL_core.num_nodes,)))
        
        
    def project_GD(self):
        self.projection_adj_op = self.adj_mat.assign(tf.matrix_set_diag(tf.where(self.adj_mat + tf.transpose(self.adj_mat)>=0,\
                                                                     tf.ones((self.graphL_core.num_nodes, self.graphL_core.num_nodes)),\
                                                                        -tf.ones((self.graphL_core.num_nodes, self.graphL_core.num_nodes))),\
                                                                             tf.ones((self.graphL_core.num_nodes,))))
        

        
    def array_diff(self, a, b):
        return np.sum(np.abs(a-b)**2)/np.sum(np.abs(b)**2)
        
        
    def train(self, X, Y, target, printing_flag=False, plotting_flag=False):
        """
            training the Graph learning process (GNN+similarity matrix) 
            X: initial features for training samples
            Y: training sample labels (0:non-seizure, 1:seizure)
            target :  refers to the patient that our algorithms is applied to
            plotting_flag: if True, plots are shown as a separate pop-up window
            printing_flag: if True, messages are printed while running the code
        """
        print('training ..')
        start_time = time.get_seconds()
        self.classif_minibatch = miniBatchIterator(self.graphL_minibatch, self.classif_core.batch_size, self.placeholders, X, Y)
        #config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
        #config.gpu_options.allow_growth = True
        #config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_FRACTION
        #config.allow_soft_placement = True
        self.sess = tf.Session() # config = config
        # Initialize the variables (i.e. assign their default value) and forest resources
        init_vars = tf.group(tf.global_variables_initializer(), resources.initialize_resources(resources.shared_resources()))
        self.sess.run(init_vars)
        
        total_steps = 0
        avg_time = 0.0
        self.weight_loss_val = 1.0
        losses = []
        class_losses = []
        graphL_losses = []
        AUC_score = []
        Eval_thresh= []
        graphL_Ws = None
        prob_hat = None
        num_epochs = self.classif_core.epochs 
        outs_before = [0, 0, 0, [0, 0], 0, 0, 0, 0, 0, 0, 0]
        for epoch in range(num_epochs):
            self.classif_minibatch.shuffle()
            iter = 0
            while(not self.classif_minibatch.end()):    
                feed_dict = self.classif_minibatch.next()
                feed_dict.update({self.placeholders['weight_losses']: self.weight_loss_val})
                self.sess.run([self.opt_op], feed_dict=feed_dict)
                if(self.train_op_class is not None and self.task_core.supervised):
                    self.sess.run([self.train_op_class, self.loss_op_class], feed_dict=feed_dict) 
                    
                    
                outs = self.sess.run([self.loss, self.graphL_W, self.adj_mat, self.variables, 
                                        self.loss_class, self.loss_graphL, self.Z, self.theta_array1, 
                                        self.graphL_W, self.pred_probas, self.theta_array2], feed_dict=feed_dict)
                print('self.graphL_W \n', outs[1])
#                 print('self.poww \n', outs[11])
                # vvars = self.sess.run([self.grads_and_variables], feed_dict=feed_dict)
                # ttest = self.sess.run([self.Z, self.neigh_means, self.from_neighs], feed_dict=feed_dict)
#                 self.sess.run([tf.add_check_numerics_ops()])              
#                 print('pred classes:\n ', self.sess.run([self.pred_classes], feed_dict=feed_dict) )
#                 print('pred probabilities:\n ', self.sess.run([self.pred_probas], feed_dict=feed_dict) )
                # self.sess.run([self.adj_mat], feed_dict=feed_dict)
                if(total_steps == 0):
                    self.weight_loss_val = 1 # np.abs(outs[4]/outs[5])
                # outs = self.sess.run([self.loss, self.graphL_W, self.adj_mat, self.variables, self.loss_class, self.loss_graphL, self.Z, self.Theta], feed_dict=feed_dict)
                inn_loss = outs[0]
                if(inn_loss is None):
                    raise Exception('Loss is None!')
                losses.append(inn_loss)
                class_losses.append(outs[4])
                graphL_losses.append(outs[5])
                inn_W = outs[8]
                inn_prob_hat = outs[9][:,1]
                if(self.classif_minibatch.end()):
                    sel_idxx = np.arange(inn_W.shape[0]-(Y.size- graphL_Ws.shape[0]), inn_W.shape[0]) if graphL_Ws is not None else np.arange(inn_W.shape[0])
                    inn_W = inn_W[sel_idxx, ...]
                graphL_Ws = inn_W if graphL_Ws is None else np.concatenate((graphL_Ws, inn_W), axis=0)
                inn_AUC_score, inn_FA, in_eval_thresh, in_sensitivity = fine_eval_performance(self.classif_minibatch.current_y(), inn_prob_hat, \
                                                                self.classif_core.sensitivity_score_target, printing_flag=printing_flag and self.task_core.supervised)
                AUC_score.append(inn_AUC_score)
                if(epoch == num_epochs-1):
                    Eval_thresh.append(in_eval_thresh)
#                 print('')
#                 print('    loss; %f, loss-class: %f, loss-graphL: %f' %(outs[0], outs[4], outs[5]))
#                 print('    A after projection: \n', (outs[2]+1)/2)
                if(self.graphL_core.side_adj_mat is None and self.graphL_core.coordinate_gradient):
                    self.adj_mat_coordinate_descent()
                elif(self.graphL_core.side_adj_mat is None):
#                     self.project_GD()
                    outs_after = self.sess.run([self.projection_adj_op], feed_dict=feed_dict) # outs_after = self.sess.run([self.adj_mat, self.variables, self.Theta], feed_dict=feed_dict)
#                     print('            Inner - A diff: ', np.sum(np.abs(outs_after[0]-outs[2])))
#                     print('            Inner - Theta diff: ', np.sum(np.abs(outs_after[2]-outs[7])))
#                     print('            Inner - GraphL-variables0 diff: ', np.sum(np.abs(outs_after[1][0]-outs[3][0])))
#                     print('            Inner - GraphL-variables1 diff: ', np.sum(np.abs(outs_after[1][1]-outs[3][1])))
#                     print('')
                if(printing_flag):
                    print('        A diff: ', np.sum(np.abs(outs[2]-outs_before[2])))
                    print('        Theta diff1: ', np.sum(np.abs(outs[7]-outs_before[7])))
                    print('        Theta diff2: ', np.sum(np.abs(outs[10]-outs_before[10])))
                    print('        GraphL-variables0 diff: ', np.sum(np.abs(outs[3][0]-outs_before[3][0])))
                    try:
                        print('        GraphL-variables1 diff: ', np.sum(np.abs(outs[3][1]-outs_before[3][1])))
                    except:
                        pass
                    print('        GraphL-Loss:', outs[5])
                    print('')
    #                 A_diff = np.sum(np.abs(outs_after[0]-outs_before[2]))
    #                 print('    A diff: ', A_diff) # (1+outs_after[0])/2
    #                 print('    variables0 diff: ', np.sum(np.abs(outs_after[1][0]-outs_before[3][0])))
    #                 print('    variables1 diff: ', np.sum(np.abs(outs_after[1][1]-outs_before[3][1])))

    #                 if(A_diff<self.classif_core.A_proj_th and A_diff>0 and iter>10):
    #                     self.project_GD()
    #                 outs_before = outs.copy()
#                 print('Sample Z: ', outs[6][3][0])
#                 print('        : ', outs[6][3][3])
#                 print('Sample W: ', np.reshape(outs[1][3], (self.graphL_core.num_nodes, self.graphL_core.num_nodes)))
#                 np.savetxt('Sample_Z.txt', outs[6][3])
#                 np.savetxt('Sample_W.txt', outs[1][3])
                iter += 1
                total_steps += 1
                outs_before = outs.copy()
                if total_steps > self.classif_core.max_total_steps:
                    break
#                 print('      loss class=%f, weighted loss graphL=%f, loss graphL=%f, weight-loss = %f' 
#                             % ( outs[4], self.weight_loss_val*outs[5], outs[5],self. weight_loss_val))
                
            print('    epoch: ', epoch)
#             print('    loss: ', outs[0])
            if total_steps > self.classif_core.max_total_steps:
                    break
                
                
            
                    
        if(self.graphL_core.side_adj_mat is None):
            outs_after = self.sess.run([self.adj_mat, self.variables], feed_dict=feed_dict)
            self.final_A_evaluated = (outs_after[0]+1)/2
#             plotting_figure(np.array(outs_after[0]), 'adjacency_mat')
        else:
            self.final_A_evaluated = (outs[2]+1)/2
        self.final_Theta_evaluated1 = outs[7]
        self.final_Theta_evaluated2 = outs[10]
        
        save_folder = 'EU_plots/'+str(target)+'/'
        plotting_figure(save_folder, [np.log(np.array(losses))], 'training_loss_' + self.graphL_core.A_val_string)
        plotting_figure(save_folder, [np.log(np.array(class_losses))], 'training_Classification_loss_' + self.graphL_core.A_val_string)
        plotting_figure(save_folder, [np.log(np.array(graphL_losses))], 'training_GraphLearning_loss_' + self.graphL_core.A_val_string)
        plotting_figure(save_folder, [np.log(np.array(AUC_score))], 'training_AUC_' + self.graphL_core.A_val_string)
#         plotting_figure(np.log(1+np.exp(-10)-np.array(AUC_score)), 'training_OneMinusAUC_' + self.graphL_core.A_val_string)
#         self.evel_threshold = self.classif_core.eval_thresh_func(np.vstack(Eval_thresh), axis=0)
        print('    time elapsed: ', time.get_seconds()-start_time)  
        
        return self.classif_core.eval_thresh_func(np.vstack(Eval_thresh), axis=0)
        
    def train_disjointed_classifier(self, X, Y, target, printing_flag=False, plotting_flag=False):
        feed_dict = {self.placeholders['X']: X, self.placeholders['Y']: Y}
        feed_dict.update({self.placeholders['weight_losses']: self.weight_loss_val})
        outs = self.sess.run([self.graphL_W], feed_dict=feed_dict)
        graphL_Ws = outs[0]
        self.classif_core.disjointed_classifier = self.classif_core.disjointed_classifier.fit(graphL_Ws, Y)
        prob_hat = self.classif_core.disjointed_classifier.predict_proba(graphL_Ws)[:,1]
        print('train disjointed classifier performance: ---------------')
        fine_eval_performance(Y, prob_hat, self.classif_core.sensitivity_score_target,printing_flag=printing_flag)
        return prob_hat
        
    def test(self, X, Y, target, train_test='testing', out_th=None, printing_flag=False, plotting_flag=False, \
                Delays = [], FAs = [], df=None, total_hours = 0, clip_sizes=None, sel_win_nums=None, task_core=None):
        """
            perform testing for the whole GNN + similarity matrix generation + classifier 
            X: initial features for samples
            Y: sample labels (0:non-seizure, 1:seizure)
            target :  refers to the patient that our algorithms is applied to
            train_test: which samples are being tested: 'training', 'validation', and 'testing'
            out_th: 
            plotting_flag: if True, plots are shown as a separate pop-up window
            printing_flag: if True, messages are printed while running the code
            Delays: the delay to detect a seizure (i.e. the time to first estimation of label 1 for consecutive samples)
            FAs: the rate of False Alarm (i.e. falsely detecting a seizure) per hour 
            df: dataframe containing the performance measures, i.e. 
                                         'FA per hour': the rate of False Alarm (i.e. falsely detecting a seizure) per hour 
                                         'Mean delay': the average of all delays for seizure instances
                                         'Threshold':
                                         'Non-detected szrs': number of the non-detected seizures 
            total_hours: keeps track of the duration of time (in hours) before running this function for a new batch of samples
            clip_sizes: is a 2-d array of shape 2 * #samples, 
                        clip_sizes[0,i] shows the sample index that the i-th clip starts from
                        clip_sizes[1,i] shows the sample index that the i-th clip ends with
                        each clip is a one-hour iEEG that is acquired consecutively and in real time
            sel_win_nums: the real indices of samples (or windows) that are selected as features in our computations, 
                          including windows from all the non-szr, pre-szr, and szr
            task_core: a class instance containing the hyper-parameters of the GNN, graph learning, and classification tasks
        """
        print(train_test +' .. ')
        if(Y is not None):
            print('     sample size = '+ str(Y.size))
        start_time = time.get_seconds()
        if(X is not None):
            if(not self.task_core.supervised):
                feed_dict = {self.placeholders['X']: X, self.placeholders['Y']: Y}
                feed_dict.update({self.placeholders['weight_losses']: self.weight_loss_val})
                outs = self.sess.run([self.graphL_W], feed_dict=feed_dict)
                graphL_Ws = outs[0]
                print('  NCDD graphL: time elapsed to compute %d graphs with %d nodes is %f: ' %( graphL_Ws.shape[0], graphL_Ws.shape[1], time.get_seconds()-start_time))
                prob_hat = self.classif_core.disjointed_classifier.predict_proba(graphL_Ws)[:,1]
                print(train_test + ' disjointed classifier performance: ---------------')
                fine_eval_performance(Y, prob_hat, self.classif_core.sensitivity_score_target,printing_flag=printing_flag)
                return FAs, Delays, df, total_hours, prob_hat, graphL_Ws
            classif_minibatch = miniBatchIterator(self.graphL_minibatch, self.classif_core.batch_size, self.placeholders, X, Y)
        else:
            classif_minibatch = online_miniBatchIterator(self.matlab_load_core, self.placeholders, self.task_core)
        y_hat = None
        prob_hat = None
        W_hat = None
        y_true = None
        total_steps = 0
        
        szr_size = 0
        non_szr_size = 0
        if(df is None and out_th is not None):
            data = {'FA per hour': -1e3*np.ones_like(out_th), 'Mean delay': -1e3*np.ones_like(out_th), 'Threshold': out_th, 'Non-detected szrs': -1e3*np.ones_like(out_th)}
            df = pandas.DataFrame(data=data)
        while(not classif_minibatch.end()):
            feed_dict = classif_minibatch.next()
            feed_dict.update({self.placeholders['weight_losses']: self.weight_loss_val})
            outs = self.sess.run([self.graphL_W, self.loss, self.pred_probas], feed_dict=feed_dict)
            inn_W = outs[0]
            inn_prob = outs[2][:,1]
            inn_y_true = classif_minibatch.current_y()
            szr_size += np.argwhere(inn_y_true!=0).size
            non_szr_size += np.argwhere(inn_y_true==0).size 
            
            if(X is not None):
                if(classif_minibatch.end()):
                    sel_idxx = np.arange(inn_prob.size-(Y.size- prob_hat.size), inn_prob.size) if prob_hat is not None else np.arange(inn_y_true.size)                    
    #                 inn_y = inn_y[sel_idxx]
                    inn_prob = inn_prob[sel_idxx,...]
                    inn_W = [inn_W[i] for i in sel_idxx]
                    inn_y_true = inn_y_true[sel_idxx]
    #             y_hat = inn_y if y_hat is None else np.concatenate((y_hat, inn_y))
                prob_hat = inn_prob if prob_hat is None else np.concatenate((prob_hat, inn_prob), axis=0)
                W_hat = inn_W if W_hat is None else np.concatenate((W_hat, inn_W), axis=0)
                y_true = inn_y_true if y_true is None else np.concatenate((y_true, inn_y_true), axis=0)
                fine_eval_performance(y_true, prob_hat, self.classif_core.sensitivity_score_target, th=out_th, printing_flag=printing_flag, features=W_hat) 
                 
            else:
                if(plotting_flag): #
                    print('******* inn_W size: ', inn_W.shape)
                    print('******* clip_sizes:', clip_sizes)
                    plot_samples(inn_y_true, inn_prob, inn_W, self.matlab_load_core, self.task_core, clip_sizes, \
                                    sel_win_nums, '_'+str(total_steps) + task_core.feature_mode, num_plots=10*15, mode_plotting='one_clip')
                    
                print(train_test + ' %f percent is done, %d samples so far' % (total_steps*100/self.matlab_load_core.settings_TestNumFiles, szr_size+non_szr_size) )
                if(not self.task_core.supervised):
                    inn_prob = self.classif_core.disjointed_classifier.predict_proba(inn_W)[:,1]
                inFA, inDelay = coarse_eval_performance(inn_y_true, inn_prob, self.classif_core.sensitivity_score_target, th=out_th, \
                                                                printing_flag=False, matlab_load_core=self.matlab_load_core, 
                                                                        mvAvg_winlen=self.classif_core.mvAvg_winlen, counter=total_steps)
                FAs.append(inFA)
                Delays.append(inDelay) # [~np.isnan(Delays) and ~np.isinf(Delays)]
                total_hours += win2sec(inn_y_true.size, self.matlab_load_core)/3600
                print('total_hourse spent: ', total_hours)
                print('Number of files passed: ', len(FAs))
                if(printing_flag):
                    Inf2NanDelays = np.copy(Delays)
                    Inf2NanDelays[Inf2NanDelays==np.inf] = np.nan
                    try: 
                        meanDelay = np.nanmean(np.vstack(np.array(Inf2NanDelays)), axis=0)
                        worstDelay = np.nanmax(np.vstack(np.array(Inf2NanDelays)), axis=0)
                        bestDelay = np.nanmin(np.vstack(np.array(Inf2NanDelays)), axis=0)
                    except:
                        meanDelay = None
                        worstDelay = None
                        bestDelay = None
                        
#                         Delays[Delays==None] = np.nan
                    print ('From %d szr samples and  %d nonszr samples' % ( szr_size, non_szr_size))
                    FApHour = np.sum(np.vstack(np.array(FAs)), axis=0)/total_hours
                    df['Non-detected szrs'] = np.sum(np.array(Delays)==np.inf, axis=0)
                    df['FA per hour'] = FApHour
                    df['Mean delay'] = meanDelay
                    if(not np.all(np.isnan(inDelay))):
                        df['Delay'+str(len(FAs))] = pandas.Series(inDelay)
                    with pandas.option_context('display.max_rows', None, 'display.max_columns', None):
                        print(df)
                        df.to_csv(str(target)+'_out.csv', encoding='utf-8', index=False)

#                     print('FA per hour=           %s' %( FApHour))
#                     print('Delays=                %s seconds' %  (Delays))
#                     print('-----Mean delay=       %s seconds' %  (meanDelay))
#                     print('-----Worst delay=      %s seconds' %  (worstDelay))
#                     print('-----Best delay=       %s seconds' %  (bestDelay))
#                     print('Threshold=             %s' % (out_th))
#                     print('Non-detected szrs=     %s' % (np.sum(np.array(Delays)==np.inf, axis=0)))
                     
            total_steps += 1
            
#         print('** FINAL ' +train_test +' loss: ', outs[1])
#         if(X is not None):
#             print(train_test  + ' '+ self.graphL_core.A_val_string + ' for patient '+ str(target))
#             eval_performance(Y, prob_hat, self.classif_core.sensitivity_score_target, th=th)  
        
        """
            Evaluating the classification performance and plotting the figures
        """
        if(X is not None):
            if(plotting_flag):
                plot_samples(y_true, prob_hat, W_hat, self.matlab_load_core, self.task_core, clip_sizes, \
                                    sel_win_nums, 'RepL', num_plots=10*15, mode_plotting='aggregate_clips')
                
                
        return FAs, Delays, df, total_hours, prob_hat, W_hat
        
             
    
    def printing(self, adj_calc_modes):
        print('Optimized Theta: ')
        print(np.unique(self.final_Theta_evaluated1))
        print(np.unique(self.final_Theta_evaluated2))
        print('Optimized A: ')
        print(self.final_A_evaluated)
#         print('final eval-thresholds =%s for sensitivities = %s' %(self.evel_threshold, self.classif_core.sensitivity_score_target))
        for i in range(len(self.matlab_load_core.structural_inf.adj_means)):
            print('------Adj mode: ' + adj_calc_modes[i])
            print('    Side A mean:  ')
            print(self.matlab_load_core.structural_inf.adj_means[i])
            print('    A difference ratio: ', np.sum(np.abs(self.final_A_evaluated-self.matlab_load_core.structural_inf.adj_means[i]))
                                                                /np.size(self.final_A_evaluated))
#             print('    A common ratio: ', np.sum(np.abs(self.final_A_evaluated-matlab_load_core.structural_inf.adj_means[i]))
#                                                                 /np.size(self.final_A_evaluated))
            print('    Side A var: ')
            print(self.matlab_load_core.structural_inf.adj_vars[i])
        
        
        




def tf_tril_indices(N, k=0):
    M1 = tf.tile(tf.expand_dims(tf.range(N), axis=0), [N,1])
    M2 = tf.tile(tf.expand_dims(tf.range(N), axis=1), [1,N])
    mask = (M1-M2) >= -k
    ix1 = tf.boolean_mask(M2, tf.transpose(mask))
    ix2 = tf.boolean_mask(M1, tf.transpose(mask))
    return ix1, ix2



def tf_half1D_to_2D(arr, num_nodes):  
    
#     # Create boolean mask of maximums
#     bmask = tf.equal(inds, max_inds[:, None])
#     
#     # Convert boolean mask to ones and zeros
#     imask = tf.where(bmask, tf.zeros_like(tmat), tf.ones_like(tmat))
#     
#     # Create new tensor that is masked with maximums set to zer0
#     tri = tri * imask
#     ones = tf.ones((num_nodes, num_nodes))
#     mask_a = tf.matrix_band_part(ones, 0, -1) # Upper triangular matrix of 0s and 1s
#     mask = tf.cast(mask_a, dtype=tf.bool) # - mask_b Make a bool mask
#     
# #     tri = tf.where(mask, arr, 0)
#     tri = tf.zeros((num_nodes, num_nodes))
# #     ix1, ix2 = tf_tril_indices(num_nodes)
#     np.tril_indices(num_nodes, 0)
#     tri[ix1, ix2] = arr
    tri = fill_triangular(arr, upper=True)
    tri = (tri+tf.transpose(tri))/2
    return tri

def pure_plot_sample(inn_true_y, inn_prob_hat, inn_sel_win_num, counter, num_plots, matlab_load_core, feats, task_core, W_method):
    ictal_list = np.squeeze(np.argwhere(inn_true_y!=0))
    # determining windows to plot
    try:
        n_pre_szr = int(matlab_load_core.pre_ictal_num_win)
    except:
        n_pre_szr = int(matlab_load_core.n_pre_szr)
    if(ictal_list.size>0):
        inter_ictal_list = np.arange(0, np.min((n_pre_szr*2,ictal_list[0]-n_pre_szr*2-matlab_load_core.window_size_sec*60*10)), 1)
        pre_ictal_list = np.arange(np.max((0, ictal_list[0]-n_pre_szr*2)), ictal_list[0], 1) # np.max((inn_true_y.size,n_pre_szr))
        post_ictal_list = np.arange(ictal_list[-1] + 1, np.min((ictal_list[-1] + n_pre_szr*2, inn_true_y.size)) , 1) # np.min((inn_true_y.size, change_arg[-1] + num_plots-change_arg.size-pre_ictal_list.size))
        change_arg = list(inter_ictal_list) + list(pre_ictal_list) + list(ictal_list) + list(post_ictal_list)
        
    elif(inn_true_y.size<=num_plots):
        change_arg = np.arange(inn_true_y.size)
    else:
        change_arg = np.random.choice(inn_true_y.size, num_plots, replace=False) 
    
    change_arg = np.sort(np.array(change_arg).astype(int))
    print('*** change_arg: ', change_arg)
    print('*** feats size: ', feats.shape)
    print('*** inn_true_y size: ', inn_true_y.shape)
    print('*** inn_prob_hat size: ', inn_prob_hat.shape)
    print('*** inn_sel_win_num size: ', inn_sel_win_num.shape)
    inn_feats = [half1D_to_2D(feats[int(i),...], matlab_load_core.num_nodes) for i in change_arg]
    if(ictal_list.size>0):
        inn_true_y = np.concatenate((np.zeros_like(inter_ictal_list),np.ones_like(pre_ictal_list),\
                                     2*np.ones_like(ictal_list), 3*np.ones_like(post_ictal_list)))
    else:
        inn_true_y = inn_true_y[change_arg]
    
    inn_prob_hat = inn_prob_hat[change_arg]
    inn_sel_win_num = None # inn_sel_win_num[change_arg]
    
    try:
        plotting_weights('EU_plots/'+str(task_core.target)+'/', str(counter) + W_method, inn_feats, true_states=inn_true_y, \
                     estimated_states=inn_prob_hat, soz_ch_ids=matlab_load_core.soz_ch_ids, sel_win_num=inn_sel_win_num)
    except:
        print('sdfsdf')
    
    
    
def plot_samples( true_y, prob_hat, feats, matlab_load_core, task_core, clip_sizes=None, sel_win_nums=None, W_method=' ', num_plots=8*8, mode_plotting='one_clip'):
    """
        Evaluating the classification performance and plotting the figures
    """
    print('plotting ..')
    if(mode_plotting == 'aggregate_clips'):
        num_nonszr_plot = 10
        non_szr_plot_prob = np.min((1, num_nonszr_plot/np.shape(matlab_load_core.y_test)[0]))
            
        for counter in np.arange(clip_sizes.shape[1]):
            
            start_idx = int(clip_sizes[0, counter])
            end_idx   = int(clip_sizes[1, counter])
            
            inn_true_y   = true_y[start_idx:end_idx]
            inn_prob_hat = prob_hat[start_idx:end_idx]
            inn_sel_win_num = sel_win_nums[start_idx:end_idx]
            inn_feats    = feats[start_idx:end_idx, ...]
            
            change_arg = np.squeeze(np.argwhere(inn_true_y!=0))
            
            if(change_arg.size == 0 and \
                np.random.choice([0,1], p=[non_szr_plot_prob, 1-non_szr_plot_prob])==1):
                continue
            
            pure_plot_sample(inn_true_y, inn_prob_hat, inn_sel_win_num, counter, num_plots, matlab_load_core, inn_feats, task_core, W_method)
            
    elif(mode_plotting == 'one_clip'):
        pure_plot_sample(true_y, prob_hat, sel_win_nums, 0, num_plots, matlab_load_core, feats, task_core, W_method)
    
               
    
def Eval_threshold(percent_target, _prob, pre_threshold=1):
    target_prob = np.sort(_prob)
    target_prob = target_prob[target_prob.size-int((target_prob.size*percent_target)):]
    if(target_prob.size == 0):
        return 1.0
    new_threshold = np.min(target_prob)
    new_threshold = np.min((new_threshold, pre_threshold))
    return new_threshold

def Eval_FA(Y, prob_hat, eval_thresh):
    zero_idx = np.argwhere(Y==0)
    return np.sum(prob_hat[zero_idx]>=eval_thresh)

def sensitivity_score(y, yhat):
    return np.sum(yhat[np.argwhere(y==1)])/np.sum(y)


def network_comparison(a, b, printing_flag=True):
    if(printing_flag):
        t, p = stats.ttest_ind(a,b, equal_var = False)
        
        
        
        
def fine_eval_performance(Y, prob_hat, sensitivity_score_target, th=None, printing_flag=False, features=None):
    
#     if(features is not None):
#         network_comparison(features[Y==0,...], features[Y!=0,...], printing_flag=True)
    
    
#         print('    sensitivity (true positive) score: ', sklearn.metrics.sensitivity_score(Y, y_hat))
#         print('    Recall (true negative) score: ', sklearn.metrics.recall_score(Y, y_hat))
    if(np.unique(Y).size == 1):
        return 1, np.zeros_like(sensitivity_score_target), np.ones_like(sensitivity_score_target), np.ones_like(sensitivity_score_target)
    FA_num = []
    eval_threshold = []
    sensitivity = []
#     sample_weight = compute_sample_weight(class_weight='balanced', y=Y)
    
    AUC_score = sklearn.metrics.roc_auc_score(Y, prob_hat) # , sample_weight=sample_weight
    fpr, tpr, thrshld = sklearn.metrics.roc_curve(Y, prob_hat) # , sample_weight=sample_weight
    tpr = tpr[:-1]
    fpr = fpr[:-1]
    
#         prcsn, rcl, thrshld = sklearn.metrics.sensitivity_recall_curve(Y, prob_hat)
#         prcsn = prcsn[:-1]
#         rcl = rcl[:-1]
#         print(prcsn, rcl, thrshld)
    for i in np.arange(len(sensitivity_score_target)):
        if(th is None):
            try:
                in_eval_thresh = min([thrshld[j] for j in range(len(thrshld)) if tpr[j]>=sensitivity_score_target[i]]) 
                # thrshld[np.argmax(tpr - fpr)]
            except:
                in_eval_thresh = Eval_threshold(sensitivity_score_target[i], prob_hat[np.argwhere(Y!=0)], pre_threshold=1)
            
        else:
            in_eval_thresh = th[i]
            
        YHAT = np.zeros_like(prob_hat)
        YHAT[prob_hat>in_eval_thresh] = 1
        in_FA_num = np.sum(YHAT[np.argwhere(Y==0)])
        FA_num.append(in_FA_num)
        eval_threshold.append(in_eval_thresh)
        in_sensitivity = sensitivity_score(Y, YHAT)
        sensitivity.append(in_sensitivity)
        if(printing_flag):
            print ('    FA number=%d from %d samples and  %d zero samples, for sensitivity=%f, threshold=%f, AUC=%f' 
                            % ( in_FA_num, Y.size, np.argwhere(Y==0).size, \
                                 in_sensitivity, in_eval_thresh, AUC_score)) # sklearn.metrics.sensitivity_score
#         plotting_figure(tpr, 'ROC_'+ training_samples, fpr, False)
    return AUC_score, np.array(FA_num), np.array(eval_threshold), np.array(sensitivity)


def moving_average(a, n=1) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return np.concatenate((a[0:n-1], ret[n - 1:] / n)) 


    
def coarse_eval_performance(Y, prob_hat, sensitivity_score_target, YREF=None, mvAvg_winlen=1, th=None, printing_flag=False, matlab_load_core=None, counter=0):
    save_folder = 'EU_plots/'+str(matlab_load_core.target)+'/'
    plotting_figure(save_folder, [Y, prob_hat], 'output_probability_'+ str(counter))
    if (YREF is None):
        YREF = np.copy(Y)
        if(np.any(Y!=0)):
            preszr_onset = int(np.argwhere(Y==1)[0])
            szr_offset = int(np.argwhere(Y==1)[-1])
            if(preszr_onset==0):
                print('Preszr labels started from the beginning of the file.')
            szr_onset = preszr_onset + matlab_load_core.pre_ictal_num_win
            YREF[preszr_onset:szr_onset] = 0
#            Y[szr_offset:np.min((szr_offset + sec2win(3*60, matlab_load_core) , YREF.size))] = 1 #
            Y[np.max((szr_onset - sec2win(10, matlab_load_core), 0)):szr_onset] = 1
#             else:
#                 raise NotImplementedError(
#                     'Preszr labels started from the beginning of the file.')
        
    FA_num = []
    delay = []
    
    if(th is not None):
        for i in np.arange(len(th)):
            in_eval_thresh = th[i]
            
            prob_hat_new = moving_average(prob_hat, n=mvAvg_winlen)
            YHAT = np.zeros_like(prob_hat_new)
            YHAT[prob_hat_new>in_eval_thresh] = 1
            if(np.any(Y!=0)):
                in_FA_num = 0
                if(np.any(YHAT!=0)):
                    szr_onset = preszr_onset + matlab_load_core.pre_ictal_num_win;
                    try:
                        szr_alarm = np.intersect1d(np.argwhere(YHAT==1), np.argwhere(Y==1))[0]
                        in_delay = win2sec(szr_alarm-szr_onset, matlab_load_core)
                    except:
#                         szr_alarm = np.argwhere(YHAT==1)[0]
                        in_delay = np.inf
                else:
                    in_delay = np.inf
            else:
                in_delay = np.nan
                in_FA_num = np.sum(YHAT[np.argwhere(Y==0)])
            
            FA_num.append(in_FA_num)    
            delay.append(in_delay)
            if(printing_flag):
                print ('    FA number=%d from %d real szr samples and  %d real nonszr samples, for delay=%f seconds, threshold=%f' 
                                % ( in_FA_num, np.argwhere(YREF==1).size, np.argwhere(YREF==0).size, in_delay, in_eval_thresh)) 
                
    return np.array(FA_num), np.array(delay)

    
def plotting_weights(save_folder, filename, mat, true_states=None, estimated_states=None, soz_ch_ids=None, sel_win_num=None):
#     title_fontSize = 40
# #     ictal_indices = np.argwhere(true_states!=0).astype(np.int)
# #     ictal_indices = ictal_indices.reshape((ictal_indices.size,))
#     num_wind = len(mat)
#     plot_num_rows = np.min((8,int(np.ceil(num_wind**0.5))))
#     plot_num_cols = int(np.ceil(num_wind/plot_num_rows)) 
#     if not os.path.exists(save_folder):
#         os.makedirs(save_folder)
#     fig = plt.figure(num=None, figsize=(60, 40), dpi=120)
#     grid = AxesGrid(fig, 111,
#                 nrows_ncols=(plot_num_rows, plot_num_cols),
#                 axes_pad=1,
#                 cbar_mode='single',
#                 cbar_location='right',
#                 cbar_pad=0.1)
#     i = 0
#     for ax in grid: 
#         if(i>=len(mat)):
#             break
#         N = int(mat[i].size**0.5)
#         im = ax.imshow(np.reshape(mat[i], (N, N)))
#         if(true_states is not None and true_states[i] != 0 and estimated_states is not None):
#             ax.set_title('--'+ str(int(estimated_states[i]))+'--', fontsize=title_fontSize+5)
#         elif(true_states is not None and true_states[i] == 0 and estimated_states is not None):
#             ax.set_title(str(int(estimated_states[i])), fontsize=title_fontSize)
#         if(sel_win_num is not None):
#             ax.set_xlabel(str(sel_win_num[i]))
#         i += 1
#     cbar = ax.cax.colorbar(im)
#     cbar.ax.tick_params(labelsize=30) 
#     cbar = grid.cbar_axes[0].colorbar(im)   
#     if(soz_ch_ids is not None):
#         fig.suptitle('Seizure Onset Channels: '+str(soz_ch_ids), fontsize=title_fontSize) 
#     fig.tight_layout()
#     plt.savefig(save_folder + 'W_' + filename + '.png')  
    io.savemat(save_folder + 'matW_' + filename, mdict={'mat': np.array(mat),'labels':true_states, 'SOZ': soz_ch_ids})   

# tf_bias = tf.Print(tf_bias, [tf_bias], "Bias: ")