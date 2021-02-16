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
from .inits import glorot, zeros
from .aggregators import MeanAggregator, MaxPoolingAggregator, MeanPoolingAggregator, SeqAggregator, GCNAggregator
import sklearn
from sklearn.metrics import roc_curve, auc
import scipy
from mpl_toolkits.axes_grid1 import AxesGrid
# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

def plotting_figure(arr, title, x_array=None, show_flag=False):
    if(x_array is None):
        x_array = np.arange(arr.size)
    plt.figure(num=None, figsize=(60, 40), dpi=120)
    plt.plot(x_array, arr)
    plt.savefig(title + '.png')
    if(show_flag):
        plt.show()

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
        feed_dict = {self.placeholders['X']: self.dataX[self.start_idx:self.end_idx,:,:],
                     self.placeholders['Y']: self.dataY[self.start_idx:self.end_idx]}
#                      self.placeholders['num_to_load']: self.batch_size
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
                    self.start_idx = self.num_samples-self.batch_size
                    self.end_idx = self.num_samples
            else:
                self.start_idx = 0
                self.end_idx = self.num_samples
        else:
            self.start_idx = self.batch_num * self.batch_size
            self.batch_num += 1
            self.end_idx = self.start_idx + self.batch_size
            if(self.end_idx>self.num_samples):
                self.start_idx = self.num_samples-self.batch_size
                self.end_idx = self.num_samples
            
            
        return self.feed_dict_update()
    
    
    def shuffle(self):
        if(self.num_samples>1):
            self.dataX, self.dataY = sklearn.utils.shuffle(self.dataX, self.dataY, random_state=0)
#             self.dataX = self.dataX # tf.random_shuffle() # np.random.permutation(self.data)
#             self.dataY = self.dataY
        self.batch_num = 0
        
    def end(self):
        if(self.clip_sizes is None):
        return self.batch_num * self.batch_size >= self.num_samples
    
 
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
       
def construct_placeholders(dim=None, num_classes=None):
    placeholders = {
        'X' : tf.placeholder(tf.float32, shape=(None, None, dim), name='X'),
        'Y' : tf.placeholder(tf.int32, shape=(None,), name='Y'), #num_classes
#         'num_to_load' : tf.placeholder(tf.int32, shape=(), name='num_to_load'),
    }
    return placeholders

def neural_net(x, n_hidden_1, n_hidden_2, num_classes):
    layer_1 = tf.layers.dense(x, n_hidden_1)
    layer_2 = tf.layers.dense(layer_1, n_hidden_2)
    out_layer = tf.layers.dense(layer_2, num_classes)
    return out_layer
#     model = keras.Sequential([
#         keras.layers.Flatten(input_shape=(28, 28)),
#         keras.layers.Dense(128, activation=tf.nn.relu),
#         keras.layers.Dense(10, activation=tf.nn.softmax)
#     ])
#     return model


def loss_classif_neural_net(features, labels, n_hidden_1, n_hidden_2, num_classes, loss_type='softmax'):
    logits = neural_net(features, n_hidden_1, n_hidden_2, num_classes)
    pred_classes = tf.argmax(logits, axis=1)
    pred_probas = tf.nn.softmax(logits)
    if(loss_type=='softmax'):
        loss_op = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                    logits=logits, labels=tf.cast(labels, dtype=tf.int32)))
    return loss_op, pred_classes, pred_probas


     
class Hybrid_Rep_Feat():
   
    def __init__(self, graphL_core, classif_core, weight_losses):
        self.graphL_core = graphL_core
        self.classif_core = classif_core
        self.weight_losses = weight_losses
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
        self.placeholders = construct_placeholders(dim=self.graphL_core.dim, num_classes=self.classif_core.num_classes) # num_samples=self.classif_core.num_samples
        self.num_to_load = np.minimum(self.classif_core.batch_size, self.classif_core.num_samples)
        self._graphL()
        self.optimizer = tf.train.AdamOptimizer(self.classif_core.learning_rate) #  GradientDescentOptimizer, AdamOptimizer
        self._loss()
        self.min_loss()
    
    def aggregate(self, input_features, samples=None):
#         hidden = [tf.nn.embedding_lookup(input_features, node_samples) for node_samples in samples]
        new_agg = self.aggregators is None
        if new_agg:
            self.aggregators = []
        for layer in range(self.graphL_core.num_layers):
            if(new_agg):
                dim_mult = 2 if self.graphL_core.concat and (layer != 0) else 1
                # aggregator at current layer
                if layer == self.graphL_core.num_layers - 1:
                    aggregator = self.aggregator_cls(dim_mult*self.graphL_core.dim, self.graphL_core.dim, self.adj_mat, act=lambda x : x,
                            concat=self.graphL_core.concat, model_size=self.graphL_core.model_size, variables=self.variables[layer]) # !!!!!!!!!!change later: 
                else:
                    aggregator = self.aggregator_cls(dim_mult*self.graphL_core.dim, self.graphL_core.dim, self.adj_mat,
                            concat=self.graphL_core.concat, model_size=self.graphL_core.model_size, variables=self.variables[layer]) # !!!!!!!!!!change later: 
                self.aggregators.append(aggregator)
            else:
                aggregator = self.aggregators[layer]
            # hidden representation at current layer for all support nodes that are various hops away
            next_hidden = []
            # as layer increases, the number of support nodes needed decreases
            for hop in range(self.graphL_core.num_layers - layer):
                dim_mult = 2 if self.graphL_core.concat and (layer != 0) else 1
                self_vecs = input_features # hidden[hop]
                neigh_vecs = tf.py_func( np.repeat, [input_features[ np.newaxis, :, :], self.graphL_core.num_nodes, 0], tf.float32) # hidden[hop+1] # wrongggggggggggggggggg
                h = aggregator((self_vecs, neigh_vecs))
                next_hidden.append(h)
            hidden = next_hidden
        
        Z = hidden[0]
        return Z

        
    def sample(self, inputs, layer_infos):
        samples = [inputs]
        # size of convolution support at each layer per node
        support_size = 1
        support_sizes = [support_size]
        for k in range(len(layer_infos)):
            t = len(layer_infos) - k - 1
            support_size *= layer_infos[t].num_samples
            sampler = layer_infos[t].neigh_sampler
            node = sampler((samples[k], layer_infos[t].num_samples))
            samples.append(node) # tf.reshape(node, [support_size * self.graphL_core.num_nodes,])
            support_sizes.append(support_size)
        return samples, support_sizes
 
    
           
    def _graphL(self):
        A_initial = tf.random_uniform((self.graphL_core.num_nodes, self.graphL_core.num_nodes))
        A_initial = tf.where(A_initial>=0.5, tf.ones((self.graphL_core.num_nodes, self.graphL_core.num_nodes)),
                                  -tf.ones((self.graphL_core.num_nodes, self.graphL_core.num_nodes)))
        A_initial = tf.matrix_set_diag(A_initial, tf.ones((self.graphL_core.num_nodes,))) #[np.diag_indices(self.graphL_core.num_nodes)] = 1
        self.adj_mat = tf.Variable(A_initial, name='adjacency_matrix')
        
        self.variables =[]
        for layer in range(self.graphL_core.num_layers):
            if(not self.graphL_core.fixed_params):
                self.variables.append(glorot([self.graphL_core.dim, self.graphL_core.dim], name='neigh_weights'))
#                 break # !!!!!!!!!!change later
            else:
                self.variables.append(self.graphL_core.fixed_neigh_weights)   
#         self.adj_info = glorot([self.graphL_core.num_nodes, self.graphL_core.num_nodes], name='adj_info')             
#         sampler = UniformNeighborSampler(self.adj_info)
#         self.layer_infos = [SAGEInfo("node", sampler, self.graphL_core.num_nodes, self.graphL_core.dim, self.graphL_core.neigh_num),
#                             SAGEInfo("node", sampler, self.graphL_core.num_nodes, self.graphL_core.dim, self.graphL_core.neigh_num)]
        initt = np.ones((len(self.graphL_core.conv_sizes),))
        self.vars_Theta_weights = tf.Variable(initt/np.sum(initt), name='vars_Theta_weights')
        
#         self.vars_Theta_weights = tf.nn.softmax(self.vars_Theta_weights) # /tf.reduce_sum(self.vars_Theta_weights)
#         list_matrices = [np.diag(np.ones((self.graphL_core.conv_sizes[i],))) * self.vars_Theta_weights[i]\
#                          for i in np.arange(self.graphL_core.conv_sizes.size)]
#         block_theta = scipy.linalg.block_diag(*list_matrices)
#         self.Theta = tf.cast(scipy.linalg.block_diag(block_theta, block_theta), tf.float32) #, tf.reshape((np.sum(self.graphL_core.conv_sizes)))
        def tf_repeat(arr, repeats):
            return tf.cast(tf.py_func(np.repeat, [arr, repeats], tf.double), tf.float32)
        
        repeated = tf_repeat(self.vars_Theta_weights, self.graphL_core.conv_sizes) #[item for item, count in zip(self.vars_Theta_weights, self.graphL_core.conv_sizes) for i in range(count)]
        repeated = tf.concat([repeated,repeated], 0)
        repeated = tf.nn.softmax(repeated)
        self.Theta = tf.diag(repeated)
        self.Z = []
#         self.loss_graphL = 0
        losses = []
        self.graphL_W = [] # tf.zeros((self.classif_core.num_samples, self.graphL_core.num_nodes**2))
        self.graphL_inner_sums = 0
#         TensorArr = tf.TensorArray(tf.int32, 1, dynamic_size=True, infer_shape=False)
#         x_unpacked = TensorArr.unstack(self.placeholders['X'])
        for i_count in np.arange(self.num_to_load): # self.placeholders['num_to_load']
#         for X in x_unpacked:
            X = tf.squeeze(self.placeholders['X'][i_count,:,:])
            inner_Z = self.aggregate(X) # samples
            inner_Z = tf.concat((X, inner_Z), axis=-1)
            self.Z.append(inner_Z)
            W_in = tf.matmul(inner_Z, tf.matmul(self.Theta, tf.transpose(tf.conj(inner_Z))))
            self.graphL_W.append(tf.reshape(W_in,(self.graphL_core.num_nodes**2,)))
            multiply = tf.constant([self.graphL_core.num_nodes])
            vec = tf.reduce_logsumexp(W_in, axis=1)
            inner_tiled = tf.reshape(tf.tile(vec, multiply), [multiply[0], tf.shape(vec)[0]])
            inner_sum = W_in-inner_tiled
            self.graphL_inner_sums += inner_sum
            inner_mult = tf.multiply(inner_sum, (self.adj_mat+1)/2)
            inner_loss = tf.reduce_sum(tf.reduce_sum(inner_mult))
            losses.append(inner_loss)
        self.loss_graphL = -tf.reduce_sum(tf.convert_to_tensor(losses, dtype=tf.float32))
#         self.adj_mat_coordinate_descent() 
    
    def _test_graphL(self):
        self.test_Z = []
        losses = []
        self.test_graphL_W = [] 
        self.test_graphL_inner_sums = 0
        for i_count in np.arange(self.test_num_to_load): 
            X = tf.squeeze(self.placeholders['X'][i_count,:,:])
            inner_Z = self.aggregate(X) 
            inner_Z = tf.concat((X, inner_Z), axis=-1)
            self.test_Z.append(inner_Z)
            W_in = tf.matmul(inner_Z, tf.matmul(self.Theta, tf.transpose(tf.conj(inner_Z))))
            self.test_graphL_W.append(tf.reshape(W_in,(self.graphL_core.num_nodes**2,)))
            multiply = tf.constant([self.graphL_core.num_nodes])
            vec = tf.reduce_logsumexp(W_in, axis=1)
            inner_tiled = tf.reshape(tf.tile(vec, multiply), [multiply[0], tf.shape(vec)[0]])
            inner_sum = W_in-inner_tiled
            self.test_graphL_inner_sums += inner_sum
            inner_mult = tf.multiply(inner_sum, (self.adj_mat+1)/2)
            inner_loss = tf.reduce_sum(tf.reduce_sum(inner_mult))
            losses.append(inner_loss)
        self.test_loss_graphL = -tf.reduce_sum(tf.convert_to_tensor(losses, dtype=tf.float32))
     
    
    def _test_loss(self):
        
        features = tf.convert_to_tensor(self.test_graphL_W, dtype=tf.float32)
        labels = self.placeholders['Y']
        self.test_loss_class, self.test_pred_classes, self.test_pred_probas = loss_classif_neural_net(features, labels, self.classif_core.n_hidden_1,\
                                                   self.classif_core.n_hidden_2, self.classif_core.num_classes)
        self.test_loss = self.test_loss_class + self.weight_losses * self.test_loss_graphL       
        self.test_loss = self.test_loss / tf.cast(self.classif_core.batch_size, tf.float32)
        
        
        
    def _loss(self):
        
        features = tf.convert_to_tensor(self.graphL_W, dtype=tf.float32) # tf.reshape(self.graphL_W,(self.graphL_core.num_nodes*self.graphL_core.num_nodes,1))
        labels = self.placeholders['Y']
        self.loss_class, self.pred_classes, self.pred_probas = loss_classif_neural_net(features, labels, self.classif_core.n_hidden_1,\
                                                                           self.classif_core.n_hidden_2, self.classif_core.num_classes, self.classif_core.loss_type)
        self.loss = self.loss_class + self.weight_losses * self.loss_graphL
        self.loss = self.loss / tf.cast(self.classif_core.batch_size, tf.float32)
    
    def min_loss(self):
        
#         clipped_grads_and_variables = [(tf.clip_by_value(grad, -1., 1.), var) if grad is not None else (tf.zeros_like(var), var) for grad, var in self.grads_and_variables]
#         clipped_grads_and_variables = [(tf.clip_by_value(grad, -1.0, 1.0) if grad is not None else None, var) 
#                 for grad, var in self.grads_and_variables]

#         self.grads_and_variables = self.optimizer.compute_gradients(self.loss)
#         clipped_grads_and_variables = [((tf.clip_by_value(grad, -1., 1.), var), var) for grad, var in self.grads_and_variables] #grad if grad is not None else tf.zeros_like(var)
#         self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_variables) # self.grads_and_variables
        if(self.graphL_core.coordinate_gradient):
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES).copy()
            var_list = var_list.remove(self.adj_mat)
            self.opt_op = self.optimizer.minimize(self.loss, var_list=var_list)
        else:
            self.opt_op = self.optimizer.minimize(self.loss)
    
    def adj_mat_coordinate_descent(self):
        
        self.adj_mat = tf.where(self.graphL_inner_sums>=0, tf.ones((self.graphL_core.num_nodes, self.graphL_core.num_nodes)),
                                  -tf.ones((self.graphL_core.num_nodes, self.graphL_core.num_nodes)))
        self.adj_mat = tf.matrix_set_diag(self.adj_mat, tf.ones((self.graphL_core.num_nodes,)))
        
        
    def project_GD(self):
        self.adj_mat = tf.where(self.adj_mat+tf.transpose(self.adj_mat)>=0, tf.ones((self.graphL_core.num_nodes, self.graphL_core.num_nodes)),
                                  -tf.ones((self.graphL_core.num_nodes, self.graphL_core.num_nodes)))
        self.adj_mat = tf.matrix_set_diag(self.adj_mat, tf.ones((self.graphL_core.num_nodes,)))
    
     
#     def apply(self, X, Y, num_clips, train_flag=True, show_plots=True):
#         if(train_flag):
#             print('training ..')
#         else:
#             print('testing ..')
#         self.classif_minibatch = miniBatchIterator(self.graphL_minibatch, self.classif_core.batch_size, self.placeholders, X, Y, num_clips)
#         config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
#         config.gpu_options.allow_growth = True
#         #config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_FRACTION
#         config.allow_soft_placement = True
#         self.sess = tf.Session(config=config)
#         self.sess.run(tf.global_variables_initializer())
#         total_steps = 0
#         avg_time = 0.0
#         num_epochs = self.classif_core.epochs if train_flag else 1
#         for epoch in range(num_epochs):
#             if(not show_plots): 
#                 self.classif_minibatch.shuffle() 
#             iter = 0
#             while(not self.classif_minibatch.end()):            
#                 feed_dict = self.classif_minibatch.next()
# #                 t = time.time()
#                 if(train_flag):
#                     self.sess.run([self.opt_op], feed_dict=feed_dict)
#                 outs = self.sess.run([self.loss, self.graphL_W, self.adj_mat, self.variables, self.grads_and_variables], feed_dict=feed_dict)
#                 if(show_plots):
#                     self.Show_Weights(outs[1])
# #                 avg_time = (avg_time * total_steps + time.time() - t) / (total_steps + 1)
#                 
# #                 if total_steps % self.classif_core.print_every == 0:
#                 print('    loss: ', outs[0])
# #                 print('graphL variables 1: ', outs[3][0])
# #                 print('graphL variables 2: ', outs[4][1])
# #                 print('A_pre: ', outs[2])
#                 print('    A before projection: \n', (outs[2]+1)/2)
#                 self.project_GD()
#                 outs_after = self.sess.run([self.adj_mat, self.variables], feed_dict=feed_dict)
#                 print('    A diff: ', np.sum(np.abs(outs_after[0]-outs[2])))
#                 iter += 1
#                 total_steps += 1
#     
#                 if total_steps > self.classif_core.max_total_steps:
#                     break
#             if(show_plots):
#                 break
#             if total_steps > self.classif_core.max_total_steps:
#                     break
#         print('Final A: \n', (outs_after[0]+1)/2)
        
    def array_diff(self, a, b):
        return np.sum(np.abs(a-b)**2)/np.sum(np.abs(b)**2)
        
        
    def train(self, X, Y):
        print('training ..')
        start_time = time.get_seconds()
        self.classif_minibatch = miniBatchIterator(self.graphL_minibatch, self.classif_core.batch_size, self.placeholders, X, Y)
        #config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
        #config.gpu_options.allow_growth = True
        #config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_FRACTION
        #config.allow_soft_placement = True
        self.sess = tf.Session() #config=config
        self.sess.run(tf.global_variables_initializer())
        total_steps = 0
        avg_time = 0.0
        losses = []
        num_epochs = self.classif_core.epochs 
        outs_before = [0,0,0,[0,0]]
        for epoch in range(num_epochs):
            self.classif_minibatch.shuffle() 
            iter = 0
            while(not self.classif_minibatch.end()):            
                feed_dict = self.classif_minibatch.next()
#                 self.project_GD()
                self.sess.run([self.opt_op], feed_dict=feed_dict)
                outs = self.sess.run([self.loss, self.graphL_W, self.adj_mat, self.variables,
                                       self.loss_class, self.loss_graphL, self.Z, self.Theta],
                                       feed_dict=feed_dict)
                losses.append(outs[0])
#                 print('')
#                 print('    loss; %f, loss-class: %f, loss-graphL: %f' %(outs[0], outs[4], outs[5]))
#                 print('    A after projection: \n', (outs[2]+1)/2)
                if(self.graphL_core.coordinate_gradient):
                    self.adj_mat_coordinate_descent()
                elif(self.graphL_core.projected_gradient):
                    self.project_GD()
                outs_after = self.sess.run([self.adj_mat, self.variables], feed_dict=feed_dict)
#                 print('    A diff-inner: ', np.sum(np.abs(outs_after[0]-outs[2])))
#                 print('    variables0 diff-inner: ', np.sum(np.abs(outs_after[1][0]-outs[3][0])))
#                 print('    variables1 diff-inner: ', np.sum(np.abs(outs_after[1][1]-outs[3][1])))
#                 print('')
#                 A_diff = np.sum(np.abs(outs_after[0]-outs_before[2]))
#                 print('    A diff: ', A_diff) # (1+outs_after[0])/2
#                 print('    variables0 diff: ', np.sum(np.abs(outs_after[1][0]-outs_before[3][0])))
#                 print('    variables1 diff: ', np.sum(np.abs(outs_after[1][1]-outs_before[3][1])))

#                 if(A_diff<self.classif_core.A_proj_th and A_diff>0 and iter>10):
#                     self.project_GD()
                outs_before = outs.copy()
#                 print('Sample Z: ', outs[6][3][0])
#                 print('        : ', outs[6][3][3])
#                 print('Sample W: ', np.reshape(outs[1][3], (self.graphL_core.num_nodes, self.graphL_core.num_nodes)))
#                 np.savetxt('Sample_Z.txt', outs[6][3])
#                 np.savetxt('Sample_W.txt', outs[1][3])
                iter += 1
                total_steps += 1
                if total_steps > self.classif_core.max_total_steps:
                    break
            print('    epoch: ', epoch)
            if total_steps > self.classif_core.max_total_steps:
                    break
        print('    Final A: ', (outs_after[0]+1)/2)
        print('    Final Theta: \n', outs[7])
#         plotting_figure(np.array(losses), 'loss')
        print('    time elapsed: ', time.get_seconds()-start_time)    
#         plotting_weights('', 'adjacency_mat', [outs_after[0]], intervals_seizures=None, estimated_states=None)
        
    def test(self, X, Y, show_plots=True, bias_name=0, training_samples = 'training', soz_ch_ids=None, sel_win_num=None, clip_sizes=None):
        print('testing ..')
        start_time = time.get_seconds()
        bias_name = 0
#         self.test_num_to_load = X.shape[0]* X.shape[1]
#         self._test_graphL()
#         self._test_loss()
#         config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
#         config.gpu_options.allow_growth = True
        #config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_FRACTION
#         config.allow_soft_placement = True
#         self.sess = tf.Session() #config=config
#         self.sess.run(tf.global_variables_initializer())
        
        
#         for counter in np.arange(X.shape[0]):            
#             feed_dict = {self.placeholders['X']: np.squeeze(X[counter,:,:,:]),
#                          self.placeholders['Y']: np.squeeze(Y[counter,:])}
#             outs = self.sess.run([self.test_graphL_W, self.test_loss, self.test_pred_classes], feed_dict=feed_dict)
#             if(show_plots):
#                 plotting_weights('MIT_plots', str(counter+bias_name), outs[0], intervals_seizures=np.squeeze(Y[counter,:]), estimated_states=outs[2])
#             print('    loss: ', outs[1])

#         Y_flat = X # np.reshape(Y,(Y.shape[0] * Y.shape[1],))
#         X_flat = Y # np.reshape(X,(X.shape[0] * X.shape[1], X.shape[2], X.shape[3]))
#         feed_dict = {self.placeholders['X']: X_flat, self.placeholders['Y']: Y_flat}
        classif_minibatch = miniBatchIterator(self.graphL_minibatch, self.classif_core.batch_size, self.placeholders, X, Y, clip_sizes=clip_sizes)
        y_hat = None
        prob_hat = None
        counter = 0
        while(not classif_minibatch.end()):
            feed_dict = classif_minibatch.next()
#             outs = self.sess.run([self.test_graphL_W, self.test_loss, self.test_pred_classes, self.test_pred_probas], feed_dict=feed_dict)
            outs = self.sess.run([self.graphL_W, self.loss, self.pred_classes, self.pred_probas], feed_dict=feed_dict)
            inn_y = outs[2]
            inn_prob = outs[3][:,1]
            start_idx, end_idx = classif_minibatch.current_idx()
            true_y = Y[start_idx:end_idx]
            inn_soz_ch_ids = soz_ch_ids[start_idx:end_idx]
            inn_sel_win_num = sel_win_num[start_idx:end_idx]
            if(classif_minibatch.end()):
                sel_idxx = range(inn_y.size-(Y.size-y_hat.size),inn_y.size)
                inn_y = inn_y[sel_idxx]
                inn_prob = inn_prob[sel_idxx]
                true_y = true_y[sel_idxx]
                inn_soz_ch_ids = inn_soz_ch_ids[sel_idxx]
                inn_sel_win_num = inn_sel_win_num[sel_idxx]
                
            y_hat = inn_y if y_hat is None else np.concatenate((y_hat, inn_y))
            prob_hat = inn_prob if prob_hat is None else np.concatenate((prob_hat, inn_prob), axis=0)
#             print('prob hat shape: ', prob_hat.shape)
#             print('batch num: ', classif_minibatch.batch_num)
            
            change_arg = np.squeeze(np.argwhere(true_y!=0))
            if(show_plots and change_arg.size!=0):
                feats = outs[0]
                change_arg = list(np.arange(np.max((0,change_arg[0]-15)),change_arg[0],1)) + list(change_arg) # + list(np.arange(change_arg[-1], np.min((true_y.size,change_arg[-1]+3)),1))
#                     change_arg = list(np.arange(9))
                print('change_arg: ', change_arg)
                feats = [feats[int(i)] for i in change_arg]
                inn_y = inn_y[change_arg]
                true_y = true_y[change_arg]
                inn_soz_ch_ids = inn_soz_ch_ids[change_arg]
                inn_sel_win_num = inn_sel_win_num[change_arg]
                plotting_weights('EU_plots/', training_samples+str(counter+bias_name), feats, intervals_seizures=true_y, \
                                 estimated_states=inn_y, soz_ch_ids=inn_soz_ch_ids, sel_win_num=inn_sel_win_num)
                counter += 1
            
        print('    loss: ', outs[1])
#         if(self.load_Core.num_classes==2):
        eval_performance(Y, y_hat, prob_hat, training_samples)  
        print('    time elapsed: ', time.get_seconds()-start_time)


def eval_performance(Y, y_hat, prob_hat, training_samples='testing'):
    print('    Precision (true positive) score: ', sklearn.metrics.precision_score(Y, y_hat))
    print('    Recall (true negative) score: ', sklearn.metrics.recall_score(Y, y_hat))
    if(len(set(Y)) == 2):
        print('    AUC score: ', sklearn.metrics.roc_auc_score(Y, prob_hat))
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(Y, prob_hat)
        plotting_figure(tpr, 'ROC_'+ training_samples, fpr, False)
    
    
def plotting_weights(save_folder, filename, mat, true_states=None, estimated_states=None, soz_ch_ids=None, sel_win_num=None):
    title_fontSize = 40
#     ictal_indices = np.argwhere(true_states!=0).astype(np.int)
#     ictal_indices = ictal_indices.reshape((ictal_indices.size,))
    num_wind = len(mat)
    plot_num_rows = int(np.ceil(num_wind**0.5))
    plot_num_cols = int(np.ceil(num_wind/plot_num_rows)) 
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    fig = plt.figure(num=None, figsize=(60, 40), dpi=120)
    grid = AxesGrid(fig, 111,
                nrows_ncols=(plot_num_rows, plot_num_cols),
                axes_pad=1,
                cbar_mode='single',
                cbar_location='right',
                cbar_pad=0.1)
    i = 0
    for ax in grid: 
        if(i>=len(mat)):
            break
        N = int(mat[i].size**0.5)
        im = ax.imshow(np.reshape(mat[i], (N, N)))
        if(true_states is not None and true_states[i] != 0 and estimated_states is not None):
            ax.set_title('--'+ str(estimated_states[i])+'--', fontsize=title_fontSize+5)
        elif(true_states is not None and true_states[i] == 0 and estimated_states is not None):
            ax.set_title(str(estimated_states[i]), fontsize=title_fontSize)
        i += 1
    cbar = ax.cax.colorbar(im)
    cbar.ax.tick_params(labelsize=30) 
    cbar = grid.cbar_axes[0].colorbar(im)   
    plt.savefig(save_folder + 'W_' + filename + '.png')       

