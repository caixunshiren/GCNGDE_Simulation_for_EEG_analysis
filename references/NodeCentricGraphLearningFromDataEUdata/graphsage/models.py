from collections import namedtuple

import tensorflow as tf
import math

import graphsage.layers as layers
import graphsage.metrics as metrics

from .prediction import BipartiteEdgePredLayer
from .aggregators import MeanAggregator, MaxPoolingAggregator, MeanPoolingAggregator, SeqAggregator, GCNAggregator
from .inits import glorot, zeros
from .utils import tensor_repeat
from sklearn.preprocessing import normalize
flags = tf.compat.v1.flags # tf.app.flags
FLAGS = flags.FLAGS

# DISCLAIMER:
# Boilerplate parts of this code file were originally forked from
# https://github.com/tkipf/gcn
# which itself was very inspired by the keras package

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging', 'model_size'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class MLP(Model):
    """ A standard multi-layer perceptron """
    def __init__(self, placeholders, dims, categorical=True, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.dims = dims
        self.input_dim = dims[0]
        self.output_dim = dims[-1]
        self.placeholders = placeholders
        self.categorical = categorical

        self.inputs = placeholders['features']
        self.labels = placeholders['labels']

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        if self.categorical:
            self.loss += metrics.masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                    self.placeholders['labels_mask'])
        # L2
        else:
            diff = self.labels - self.outputs
            self.loss += tf.reduce_sum(tf.sqrt(tf.reduce_sum(diff * diff, axis=1)))

    def _accuracy(self):
        if self.categorical:
            self.accuracy = metrics.masked_accuracy(self.outputs, self.placeholders['labels'],
                    self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(layers.Dense(input_dim=self.input_dim,
                                 output_dim=self.dims[1],
                                 act=tf.nn.relu,
                                 dropout=self.placeholders['dropout'],
                                 sparse_inputs=False,
                                 logging=self.logging))

        self.layers.append(layers.Dense(input_dim=self.dims[1],
                                 output_dim=self.output_dim,
                                 act=lambda x: x,
                                 dropout=self.placeholders['dropout'],
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GeneralizedModel(Model):
    """
    Base class for models that aren't constructed from traditional, sequential layers.
    Subclasses must set self.outputs in _build method

    (Removes the layers idiom from build method of the Model class)
    """

    def __init__(self, **kwargs):
        super(GeneralizedModel, self).__init__(**kwargs)
        

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

# SAGEInfo is a namedtuple that specifies the parameters 
# of the recursive GraphSAGE layers
SAGEInfo = namedtuple("SAGEInfo",
    ['layer_name', # name of the layer (to get feature embedding etc.)
     'neigh_sampler', # callable neigh_sampler constructor
     'num_samples',
     'output_dim', # the output (i.e., hidden) dimension
    ])


def custom_optimizer(learning_rate):
    import numpy as np
    from scipy.optimize import minimize
    def objective(x):
        return x[0]*x[3]*(x[0]+x[1]+x[2])+x[2]
    def constraint1(x):
        return x[0]*x[1]*x[2]*x[3]-25.0
    def constraint2(x):
        sum_eq = 40.0
        for i in range(4):
            sum_eq = sum_eq - x[i]**2
        return sum_eq
    # initial guesses
    n = 4
    x0 = np.zeros(n)
    x0[0] = 1.0
    x0[1] = 5.0
    x0[2] = 5.0
    x0[3] = 1.0
    b = (1.0,5.0)
    bnds = (b, b, b, b)
    con1 = {'type': 'ineq', 'fun': constraint1} 
    con2 = {'type': 'eq', 'fun': constraint2}
    cons = ([con1,con2])
    solution = minimize(objective,x0,method='SLSQP',\
                        bounds=bnds,constraints=cons)
    x = solution.x
    return 1



class SampleAndAggregate(GeneralizedModel):
    """
    Base implementation of unsupervised GraphSAGE
    """

    def __init__(self, placeholders, features, adj, degrees,
            layer_infos, Adj_mat, non_edges, concat=True, aggregator_type="mean", 
            model_size="small", identity_dim=0, loss_function = 'xent', fixed_theta_1 = None, fixed_neigh_weights= None, neg_sample_weights = 1.0, 
            aggbatch_size=None, negaggbatch_size=None,
            **kwargs):
        
        
        super(SampleAndAggregate, self).__init__(**kwargs)
        #print("in __init__")
        self.Adj_mat = Adj_mat
        self.non_edges = non_edges
        self.fixed_theta_1 = fixed_theta_1
        self.neg_sample_weights = neg_sample_weights
        self.all_nodes = placeholders["all_nodes"]
        
        self.agg_batch_Z1 = placeholders["agg_batch_Z1"]
        self.agg_batch_Z2 = placeholders["agg_batch_Z2"]
        self.agg_batch_Z3 = placeholders["agg_batch_Z3"]
        self.agg_batch_Z4 = placeholders["agg_batch_Z4"]
        
        
        self.fixed_neigh_weights = fixed_neigh_weights
        if aggregator_type == "mean":
            self.aggregator_cls = MeanAggregator
        elif aggregator_type == "seq":
            self.aggregator_cls = SeqAggregator
        elif aggregator_type == "maxpool":
            self.aggregator_cls = MaxPoolingAggregator
        elif aggregator_type == "meanpool":
            self.aggregator_cls = MeanPoolingAggregator
        elif aggregator_type == "gcn":
            self.aggregator_cls = GCNAggregator
#         elif aggregator_type == "custom_aggregator":
#             self.aggregator_cls = CustomAggregator
        else:
            raise Exception("Unknown aggregator: ", self.aggregator_cls)
        
        
                
        # get info from placeholders...
        self.inputs1 = placeholders["batch1"]
        self.inputs2 = placeholders["batch2"]
        self.neginputs1 = placeholders["negbatch1"]
        self.neginputs2 = placeholders["negbatch2"]
        
        self.agg_batch_Z1 = placeholders["agg_batch_Z1"]
        self.agg_batch_Z2 = placeholders["agg_batch_Z2"]
        self.agg_batch_Z3 = placeholders["agg_batch_Z3"]
        self.agg_batch_Z4 = placeholders["agg_batch_Z4"]
        
        self.model_size = model_size
        self.adj_info = adj
        if identity_dim > 0:
            self.embeds = tf.get_variable("node_embeddings", [adj.get_shape().as_list()[0], identity_dim])
        else:
            self.embeds = None
        if features is None: 
            if identity_dim == 0:
                raise Exception("Must have a positive value for identity feature dimension if no input features given.")
            self.features = self.embeds
        else:
            self.features = tf.Variable(tf.constant(features, dtype=tf.float32), trainable=False)
            if not self.embeds is None:
                self.features = tf.concat([self.embeds, self.features], axis=1)
        self.degrees = degrees
        
        self.dims = [(0 if features is None else features.shape[1]) + identity_dim]
        self.dims.extend([layer_infos[i].output_dim for i in range(len(layer_infos))])
        self.batch_size = placeholders["batch_size"]
        self.negbatch_size = placeholders["negbatch_size"]
        self.placeholders = placeholders
        self.layer_infos = layer_infos
        self.concat = concat
        self.aggbatch_size = aggbatch_size
        self.negaggbatch_size = negaggbatch_size
        #changed, added from here 
        self.loss_function = loss_function 
        self.current_similarity = tf.zeros(adj.get_shape()) # ???????????
        #with tf.variable_scope(self.name + name + '_vars'): ???????????????????
        #self.vars['threshold'] = glorot([1, ], name='threshold')
#         dim_mult = 2 if self.concat else 1
#         self.vars['theta_1'] = glorot([dim_mult*self.dims[-1],dim_mult*self.dims[-1]], name='theta_1') #[num_nodes,num_nodes]
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate) # tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate) #                 
        self.build()
        
           
        
        
        
    def sample(self, inputs, layer_infos, batch_size=None):
        
        """ Sample neighbors to be the supportive fields for multi-layer convolutions.

        Args:
            inputs: batch inputs
            batch_size: the number of inputs (different for batch inputs and negative samples).
        """
        #print('in sample!')
        if batch_size is None:
            batch_size = self.batch_size
        samples = [inputs]
        # size of convolution support at each layer per node
        support_size = 1
        support_sizes = [support_size]
        for k in range(len(layer_infos)):
            t = len(layer_infos) - k - 1
            support_size *= layer_infos[t].num_samples
            sampler = layer_infos[t].neigh_sampler
            node = sampler((samples[k], layer_infos[t].num_samples))
            samples.append(tf.reshape(node, [support_size * batch_size,])) #
            support_sizes.append(support_size)
        return samples, support_sizes

    def aggregate(self, samples, input_features, dims, num_samples, support_sizes, batch_size=None,
            aggregators=None, name=None, concat=True, model_size="small"):

        """ At each layer, aggregate hidden representations of neighbors to compute the hidden representations 
            at next layer.
        Args:
            samples: a list of samples of variable hops away for convolving at each layer of the
                network. Length is the number of layers + 1. Each is a vector of node indices.
            input_features: the input features for each sample of various hops away.
            dims: a list of dimensions of the hidden representations from the input layer to the
                final layer. Length is the number of layers + 1.
            num_samples: list of number of samples for each layer.
            support_sizes: the number of nodes to gather information from for each layer.
            batch_size: the number of inputs (different for batch inputs and negative samples).
        Returns:
            The hidden representation at the final layer for all nodes in batch
        """
        #print('in aggregate!')
        if batch_size is None:
            batch_size = self.batch_size

        # length: number of layers + 1
        hidden = [tf.nn.embedding_lookup(input_features, node_samples) for node_samples in samples]
        new_agg = aggregators is None
        if new_agg:
            aggregators = []
        for layer in range(len(num_samples)):
            if(self.fixed_neigh_weights is None):
                fixed_U = None
            else:
                fixed_U = self.fixed_neigh_weights[layer]
                
            if new_agg:
                dim_mult = 2 if concat and (layer != 0) else 1
                # aggregator at current layer
                if layer == len(num_samples) - 1:
                    aggregator = self.aggregator_cls(dim_mult*dims[layer], dims[layer+1], act=lambda x : x,
                            dropout=self.placeholders['dropout'], 
                            name=name, concat=concat, model_size=model_size, fixed_neigh_weights=fixed_U, vars=vars)
                else:
                    aggregator = self.aggregator_cls(dim_mult*dims[layer], dims[layer+1], # changed, added act
                            dropout=self.placeholders['dropout'], 
                            name=name, concat=concat, model_size=model_size, fixed_neigh_weights=fixed_U)
                aggregators.append(aggregator)
            else:
                aggregator = aggregators[layer]
            # hidden representation at current layer for all support nodes that are various hops away
            next_hidden = []
            # as layer increases, the number of support nodes needed decreases
            for hop in range(len(num_samples) - layer):
                dim_mult = 2 if concat and (layer != 0) else 1
                neigh_dims = [batch_size * support_sizes[hop], 
                              num_samples[len(num_samples) - hop - 1], 
                              dim_mult*dims[layer]]
                h = aggregator((hidden[hop],
                                tf.reshape(hidden[hop + 1], neigh_dims)))
                next_hidden.append(h)
            hidden = next_hidden
        
        Z = hidden[0]
        D = tf.shape(Z)[1]
        if(concat):
            D1 = tf.cast(D/2, tf.int32)
            Z1 = Z[:, 0:D1]
            Z2 = Z[:, D1:D]
            Z1 = Z1 - tf.reduce_mean(Z1, axis=1)[:,tf.newaxis]
            Z1 = tf.nn.l2_normalize(Z1, 1)
            Z2 = Z2 - tf.reduce_mean(Z2, axis=1)[:,tf.newaxis]
            Z2 = tf.nn.l2_normalize(Z2, 1)
            Z = tf.concat((Z1,Z2), axis=1)

        Z = Z - tf.reduce_mean(Z, axis=1)[:,tf.newaxis]
        Z = tf.nn.l2_normalize(Z, 1)
        return Z, aggregators

    

    def _build(self):
        labels = tf.reshape(
                tf.cast(self.placeholders['batch2'], dtype=tf.int64),
                [self.batch_size, 1])
        
        self.neg_samples, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
            true_classes=labels,
            num_true=1,
            num_sampled=FLAGS.neg_sample_size,
            unique=False,
            range_max=len(self.degrees),
            distortion=0.75,
            unigrams=self.degrees.tolist()))      
        
#         print('negsamples size: ', self.neg_samples.shape)     
        
        
        self.samples1, support_sizes1 = self.sample(self.inputs1, self.layer_infos)
        self.samples2, support_sizes2 = self.sample(self.inputs2, self.layer_infos)
        
        
        num_samples = [layer_info.num_samples for layer_info in self.layer_infos]
        self.outputs1, self.aggregators = self.aggregate(self.samples1, [self.features], self.dims, num_samples,
                support_sizes1, concat=self.concat, model_size=self.model_size)
        self.outputs2, _ = self.aggregate(self.samples2, [self.features], self.dims, num_samples,
                support_sizes2, aggregators=self.aggregators, concat=self.concat,
                model_size=self.model_size)
        
        negnum_samples = [layer_info.negnum_samples for layer_info in self.layer_infos]
        if(self.neginputs1 is not None):
            self.negsamples1, negsupport_sizes1 = self.sample(self.neginputs1, self.layer_infos, self.negbatch_size)
            self.negoutputs1, _ = self.aggregate(self.negsamples1, [self.features], self.dims, negnum_samples,
                    negsupport_sizes1, batch_size=FLAGS.negbatch_size, aggregators=self.aggregators, concat=self.concat,
                    model_size=self.model_size)
            
        if(self.neginputs2 is not None):
            self.negsamples2, negsupport_sizes2 = self.sample(self.neginputs2, self.layer_infos, self.negbatch_size)
            self.negoutputs2, _ = self.aggregate(self.negsamples2, [self.features], self.dims, negnum_samples,
                    negsupport_sizes2, batch_size=FLAGS.negbatch_size, aggregators=self.aggregators, concat=self.concat,
                    model_size=self.model_size)
        
        neg_samples, neg_support_sizes = self.sample(self.neg_samples, self.layer_infos, FLAGS.neg_sample_size)
        self.neg_outputs, _ = self.aggregate(neg_samples, [self.features], self.dims, num_samples,
                neg_support_sizes, batch_size=FLAGS.neg_sample_size, aggregators=self.aggregators,
                concat=self.concat, model_size=self.model_size)

        dim_mult = 2 if self.concat else 1
        self.link_pre_layer = BipartiteEdgePredLayer(dim_mult*self.dims[-1],
                dim_mult*self.dims[-1], self.placeholders, 
                params = self.vars,
                act=tf.nn.sigmoid, 
                bilinear_weights=False,
                loss_fn = FLAGS.loss_function, # changed, added this line to custom loss not 'xent' default loss
                name='edge_predict',  
                fixed_theta_1=self.fixed_theta_1,
                neg_sample_weights=self.neg_sample_weights)

        if(FLAGS.flag_normalized):
            self.outputs1 = self.outputs1 - tf.reduce_mean(self.outputs1, axis=1)[:,tf.newaxis]
            self.outputs2 = self.outputs2 - tf.reduce_mean(self.outputs2, axis=1)[:,tf.newaxis]
            self.neg_outputs = self.neg_outputs - tf.reduce_mean(self.neg_outputs, axis=1)[:,tf.newaxis]
            self.negoutputs1 = self.negoutputs1 - tf.reduce_mean(self.negoutputs1, axis=1)[:,tf.newaxis]
            self.negoutputs2 = self.negoutputs2 - tf.reduce_mean(self.negoutputs2, axis=1)[:,tf.newaxis]
              
      
            self.outputs1 = tf.nn.l2_normalize(self.outputs1, 1)
            self.outputs2 = tf.nn.l2_normalize(self.outputs2, 1)
            self.neg_outputs = tf.nn.l2_normalize(self.neg_outputs, 1)
            self.negoutputs1 = tf.nn.l2_normalize(self.negoutputs1, 1)
            self.negoutputs2 = tf.nn.l2_normalize(self.negoutputs2, 1)
        
        

    def build(self):
        self._build()

        # TF graph management
        self._loss()
        self._loss_agg()
        self._accuracy()
        self._aggregation()
        self._affinity_agg()
        
# #         self.loss = self.loss / tf.cast(self.batch_size, tf.float32) # changed 
#         if(self.fixed_neigh_weights is None and self.fixed_theta_1 is None):
#             grads_and_vars = self.optimizer.compute_gradients(self.loss)
#             clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var) 
#                     for grad, var in grads_and_vars]
#             self.grad, _ = clipped_grads_and_vars[0]
#             self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)
# #             self.opt_op = self.optimizer.minimize(self.loss)
        
     
    def _aggregation(self):
        num_samples = [layer_info.num_samples for layer_info in self.layer_infos]
        samples, support_sizes = self.sample(self.all_nodes, self.layer_infos, tf.size(self.all_nodes))
        self.aggregation, _ = self.aggregate(samples, [self.features], self.dims, num_samples,
                                          support_sizes, aggregators=self.aggregators, concat=self.concat,
                                          model_size=self.model_size, batch_size=tf.size(self.all_nodes)) #tf.nn.embedding_lookup(self.adj_info, self.all_nodes)
        if(FLAGS.flag_normalized):
            self.aggregation = self.aggregation - tf.reduce_mean(self.aggregation, axis=1)[:,tf.newaxis]
            self.aggregation = tf.nn.l2_normalize(self.aggregation, 1)
    
                   
    def _affinity_agg(self): 
#         self._loss_agg()
#         N = tf.size(self.all_nodes)
#         Z = self.aggregation
#         self.Z_tilde = tensor_repeat(Z, [N], 0)
#         self.Z_tilde_tilde = tf.tile(Z, (N,1))    
#         aff, self.theta_1 = self.link_pred_layer.affinity(self.Z_tilde, self.Z_tilde_tilde)
#         self.affinity_agg = tf.reshape(aff, (N,N))
        return
 
 
    def _loss_agg(self):
#         for aggregator in self.aggregators:
#             for var in aggregator.vars.values():
#                 self.loss_agg += FLAGS.weight_decay * tf.nn.l2_loss(var)
        self.loss_agg = 0                
        num_samples = [layer_info.num_samples for layer_info in self.layer_infos]
        
        samples, support_sizes = self.sample(self.agg_batch_Z1, self.layer_infos, self.aggbatch_size)        
        self.outagg_batch_Z1, _ = self.aggregate(samples, [self.features], self.dims, num_samples,
                                          support_sizes, aggregators=self.aggregators, concat=self.concat, model_size=self.model_size, batch_size=self.aggbatch_size)
        samples, support_sizes = self.sample(self.agg_batch_Z2, self.layer_infos, self.aggbatch_size)        
        self.outagg_batch_Z2, _ = self.aggregate(samples, [self.features], self.dims, num_samples,
                                          support_sizes, aggregators=self.aggregators, concat=self.concat, model_size=self.model_size, batch_size=self.aggbatch_size)
        
        samples, support_sizes = self.sample(self.agg_batch_Z3, self.layer_infos, self.negaggbatch_size)        
        self.outagg_batch_Z3, _ = self.aggregate(samples, [self.features], self.dims, num_samples,
                                          support_sizes, aggregators=self.aggregators, concat=self.concat, model_size=self.model_size, batch_size=self.negaggbatch_size)
        samples, support_sizes = self.sample(self.agg_batch_Z4, self.layer_infos, self.negaggbatch_size)        
        self.outagg_batch_Z4, _ = self.aggregate(samples, [self.features], self.dims, num_samples,
                                          support_sizes, aggregators=self.aggregators, concat=self.concat, model_size=self.model_size, batch_size=self.negaggbatch_size)
        
        if(FLAGS.model_size == 'small'):
            self.loss_agg += self.link_pred_layer.loss(self.outagg_batch_Z1, self.outagg_batch_Z2, self.outagg_batch_Z3, neg_samples2=self.outagg_batch_Z4,\
                                                        batch_size=self.aggbatch_size, negbatch_size=self.negaggbatch_size) 
        else:
            self.loss_agg += self.link_pred_layer.loss(self.outagg_batch_Z1, self.outagg_batch_Z2, self.neg_outputs, batch_size=self.aggbatch_size) 
        self.theta_1 = self.link_pred_layer.theta_1
    
    
    def _loss(self):
        for aggregator in self.aggregators:
            for var in aggregator.vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        if(FLAGS.model_size == 'small'):
            self.loss += self.link_pred_layer.loss(self.outputs1, self.outputs2, self.negoutputs1, neg_samples2=self.negoutputs2,\
                                                        batch_size=self.batch_size, negbatch_size=self.negbatch_size) 
        else:
            self.loss += self.link_pred_layer.loss(self.outputs1, self.outputs2, self.neg_outputs, batch_size=self.batch_size) 
        
        tf.summary.scalar('loss', self.loss)

    def _accuracy(self):
#         # shape: [batch_size]
#         aff, _ = self.link_pred_layer.affinity(self.outputs1, self.outputs2)
#         # shape : [batch_size x num_neg_samples]
#         self.neg_aff = self.link_pred_layer.neg_cost(self.outputs1, self.negoutputs1) # changed neg_outputs
#         self.neg_aff = tf.reshape(self.neg_aff, [self.batch_size, FLAGS.neg_sample_size])
#         _aff = tf.expand_dims(aff, axis=1)
#         self.aff_all = tf.concat(axis=1, values=[self.neg_aff, _aff])
        _aff, _ = self.link_pred_layer.affinity(self.outputs1, self.outputs2)
        self.neg_aff, _ = self.link_pred_layer.affinity(self.negoutputs1, self.negoutputs2)
        _aff = tf.expand_dims(_aff, axis=1)
        self.neg_aff = tf.expand_dims(self.neg_aff, axis=1)
        if(tf.shape(self.neg_aff)==tf.shape(_aff)):
            self.aff_all = tf.concat(axis=1, values=[self.neg_aff, _aff])
        else:
            self.aff_all = _aff
#         if(tf.shape(self.aff_all) == tf.size(self.aff_all)):
#             self.aff_all = self.aff_all[:,tf.newaxis]


        size = tf.shape(self.aff_all)[1]
        _, indices_of_ranks = tf.nn.top_k(self.aff_all, k=size)
        _, self.ranks = tf.nn.top_k(-indices_of_ranks, k=size)
        self.mrr = tf.reduce_mean(tf.div(1.0, tf.cast(self.ranks[:, -1] + 1, tf.float32)))
        tf.summary.scalar('mrr', self.mrr)


class Node2VecModel(GeneralizedModel):
    def __init__(self, placeholders, dict_size, degrees, name=None,
                 nodevec_dim=50, lr=0.001, **kwargs):
        """ Simple version of Node2Vec/DeepWalk algorithm.

        Args:
            dict_size: the total number of nodes.
            degrees: numpy array of node degrees, ordered as in the data's id_map
            nodevec_dim: dimension of the vector representation of node.
            lr: learning rate of optimizer.
        """

        super(Node2VecModel, self).__init__(**kwargs)

        self.placeholders = placeholders
        self.degrees = degrees
        self.inputs1 = placeholders["batch1"]
        self.inputs2 = placeholders["batch2"]

        self.batch_size = placeholders['batch_size']
        self.hidden_dim = nodevec_dim

        # following the tensorflow word2vec tutorial
        self.target_embeds = tf.Variable(
                tf.random_uniform([dict_size, nodevec_dim], -1, 1),
                name="target_embeds")
        self.context_embeds = tf.Variable(
                tf.truncated_normal([dict_size, nodevec_dim],
                stddev=1.0 / math.sqrt(nodevec_dim)),
                name="context_embeds")
        self.context_bias = tf.Variable(
                tf.zeros([dict_size]),
                name="context_bias")

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)

        self.build()

    def _build(self):
        labels = tf.reshape(
                tf.cast(self.placeholders['batch2'], dtype=tf.int64),
                [self.batch_size, 1])
        self.neg_samples, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
            true_classes=labels,
            num_true=1,
            num_sampled=FLAGS.neg_sample_size,
            unique=True,
            range_max=len(self.degrees),
            distortion=0.75,
            unigrams=self.degrees.tolist()))

        self.outputs1 = tf.nn.embedding_lookup(self.target_embeds, self.inputs1)
        self.outputs2 = tf.nn.embedding_lookup(self.context_embeds, self.inputs2)
        self.outputs2_bias = tf.nn.embedding_lookup(self.context_bias, self.inputs2)
        self.neg_outputs = tf.nn.embedding_lookup(self.context_embeds, self.neg_samples)
        self.neg_outputs_bias = tf.nn.embedding_lookup(self.context_bias, self.neg_samples)

        self.link_pred_layer = BipartiteEdgePredLayer(self.hidden_dim, self.hidden_dim,
                self.placeholders, bilinear_weights=False)

    def build(self):
        self._build()
        # TF graph management
        self._loss()
        self._minimize()
        self._accuracy()

    def _minimize(self):
        self.opt_op = self.optimizer.minimize(self.loss)

    def _loss(self):
        aff = tf.reduce_sum(tf.multiply(self.outputs1, self.outputs2), 1) + self.outputs2_bias
        neg_aff = tf.matmul(self.outputs2, tf.transpose(self.neg_outputs)) + self.neg_outputs_bias
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(aff), logits=aff)
        negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(neg_aff), logits=neg_aff)
        loss = tf.reduce_sum(true_xent) + tf.reduce_sum(negative_xent)
        self.loss = loss / tf.cast(self.batch_size, tf.float32)
        tf.summary.scalar('loss', self.loss)
        
    def _accuracy(self):
        # shape: [batch_size]
        aff = self.link_pred_layer.affinity(self.outputs1, self.outputs2)
        # shape : [batch_size x num_neg_samples]
        self.neg_aff = self.link_pred_layer.neg_cost(self.outputs1, self.neg_outputs)
        self.neg_aff = tf.reshape(self.neg_aff, [self.batch_size, FLAGS.neg_sample_size])
        _aff = tf.expand_dims(aff, axis=1)
        self.aff_all = tf.concat(axis=1, values=[self.neg_aff, _aff])
        size = tf.shape(self.aff_all)[1]
        _, indices_of_ranks = tf.nn.top_k(self.aff_all, k=size)
        _, self.ranks = tf.nn.top_k(-indices_of_ranks, k=size)
        self.mrr = tf.reduce_mean(tf.div(1.0, tf.cast(self.ranks[:, -1] + 1, tf.float32)))
        tf.summary.scalar('mrr', self.mrr)


