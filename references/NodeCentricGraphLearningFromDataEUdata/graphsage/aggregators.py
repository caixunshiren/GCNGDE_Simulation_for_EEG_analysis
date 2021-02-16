import tensorflow as tf
import numpy as np
from .layers import Layer, Dense
from .inits import glorot, zeros, repeated_variable

 
def complex_nan2zero(w):
    return tf.complex(tf.where(tf.is_nan(tf.real(w)), tf.zeros_like(tf.real(w)), tf.real(w)), \
                                tf.where(tf.is_nan(tf.imag(w)), tf.zeros_like(tf.imag(w)), tf.imag(w))) #if w is nan use 1 * NUMBER else use element in w

def complex_inf2zero(w):
    return tf.complex(tf.where(tf.debugging.is_inf(tf.real(w)), tf.zeros_like(tf.real(w)), tf.real(w)), \
                                tf.where(tf.debugging.is_inf(tf.imag(w)), tf.zeros_like(tf.imag(w)), tf.imag(w))) #if w is nan use 1 * NUMBER else use element in w


   
class MeanAggregator(Layer):
    """
    Aggregates via mean followed by matmul and non-linearity.
    """

    def __init__(self, input_dim, output_dim, adj, conv_sizes=None, neigh_input_dim=None,
            dropout=0., bias=None, act=tf.nn.relu, 
            name=None, concat=False, variables=None, **kwargs):
        super(MeanAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        if(bias is None):
            self.bias = False
        else:
            self.bias = True
            self.vars['bias'] = bias
        self.act = act
        self.concat = concat
        self.adj = adj
        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''
        
        self.input_dim = input_dim
        self.output_dim = output_dim
    
        self.vars['neigh_weights'] = variables # if variables is not None else tf.linalg.tensor_diag(repeated_variable(conv_sizes)) # glorot([neigh_input_dim, output_dim], name='neigh_weights')
#         with tf.variable_scope(self.name + name + '_variables'):
#             if self.bias:
#                 self.vars['bias'] = repeated_variable(conv_sizes, name='bias') # tf.Variable(lambda: tf.zeros([self.output_dim], dtype=tf.float32), name='bias') 

        if self.logging:
            self._log_variables()

        
    
    def _variables(self):
        print('self.vars ', self.vars)
        return self.vars
    
       
        
    def _call(self, inputs):
        
        self_vecs, neigh_vecs = inputs
        expanded_adj = tf.tile (tf.expand_dims(self.adj, -1) , [1, 1, self.input_dim])
        neigh_vecs = tf.complex(expanded_adj * tf.real(neigh_vecs),  expanded_adj * tf.imag(neigh_vecs))
                                
        neigh_means = tf.reduce_sum(neigh_vecs, axis=1)
        neigh_means = tf.complex(tf.real(neigh_means)/tf.reshape(tf.reduce_sum(self.adj, axis=1), (-1,1)), 
                                            tf.imag(neigh_means)/tf.reshape(tf.reduce_sum(self.adj, axis=1), (-1,1)))
        neigh_means = complex_nan2zero(neigh_means)
        neigh_means = complex_inf2zero(neigh_means)
        
        
        
        from_neighs_real = tf.matmul(tf.real(neigh_means), self.vars['neigh_weights'])
        from_neighs_imag = tf.matmul(tf.imag(neigh_means), self.vars['neigh_weights'])
        from_neighs = tf.complex(from_neighs_real, from_neighs_imag)
        if(self.input_dim == self.output_dim):
            from_self = self_vecs 
        else:
            from_self = self_vecs[:, 0:self.output_dim]
        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)
        if self.bias:
            output += tf.cast(self.vars['bias'], tf.complex64)
        return tf.complex(self.act(tf.real(output)), self.act(tf.imag(output))) 
        
         
class MaxPoolingAggregator(Layer):
    """ Aggregates via max-pooling over MLP functions.
    """
    def __init__(self, input_dim, output_dim, adj, conv_sizes, neigh_input_dim=None,
            dropout=0., bias=None, act=tf.nn.relu, name=None, concat=False, variables=None, **kwargs):
        super(MaxPoolingAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        if(bias is None):
            self.bias = False
        else:
            self.bias = True
            self.vars['bias'] = bias
        self.act = act
        self.concat = concat
        self.adj = adj
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim

        
        if neigh_input_dim is None:
            neigh_input_dim = input_dim
        if name is not None:
            name = '/' + name
        else:
            name = ''
        self.vars['neigh_weights'] = variables if variables is not None else repeated_variable(conv_sizes)
#         if model_size == "small":
#             hidden_dim = self.hidden_dim = 512
#         elif model_size == "big":
#             hidden_dim = self.hidden_dim = 1024
# 
#         self.mlp_layers = []
#         self.mlp_layers.append(Dense(input_dim=neigh_input_dim,
#                                  output_dim=hidden_dim,
#                                  act=tf.nn.relu,
#                                  dropout=dropout,
#                                  sparse_inputs=False,
#                                  logging=self.logging))

#         with tf.variable_scope(self.name + name + '_variables'):
#             if self.bias:
#                 self.vars['bias'] = repeated_variable(conv_sizes, name='bias') # tf.Variable(lambda: tf.zeros([self.output_dim], dtype=tf.float32), name='bias') 

        if self.logging:
            self._log_variables()

        
    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs
        
    
        expanded_adj = tf.where(self.adj==1, tf.ones_like(self.adj), -(tf.reduce_max(tf.abs(neigh_vecs))+1e9) * tf.ones_like(self.adj))
        expanded_adj = tf.tile (tf.expand_dims(expanded_adj, -1) , [1, 1, self.input_dim])
        neigh_vecs = tf.complex(expanded_adj * tf.real(neigh_vecs),  expanded_adj * tf.imag(neigh_vecs))
        exp_neigh_weight = tf.tile (tf.expand_dims(self.vars['neigh_weights'], 0) , [self.adj.shape[0], 1, 1])
        from_neighs_real = tf.matmul(tf.real(neigh_vecs), exp_neigh_weight)
        from_neighs_imag = tf.matmul(tf.imag(neigh_vecs), exp_neigh_weight)
        from_neighs = tf.complex(from_neighs_real, from_neighs_imag)
        
#         from_neighs = tf.reduce_max(from_neighs, axis=1)
        
        from_neighs = tf.transpose(from_neighs, [0,2,1])
        
        max_ind = tf.cast(tf.argmax(tf.abs(from_neighs), axis=-1), tf.int32)
        
        shapee = tf.shape(from_neighs)
        xy_ind = tf.stack(tf.meshgrid(tf.range(shapee[0]), tf.range(shapee[1]), indexing='ij'), axis=-1)
        gather_ind = tf.concat([xy_ind, max_ind[..., None]], axis=-1)
        from_neighs = tf.gather_nd(from_neighs, gather_ind)
        
#         gather_indices = tf.stack((tf.range(from_neighs.get_shape()[0], dtype=max_indices.dtype), max_indices), axis=1)
#         from_neighs = tf.gather_nd(from_neighs, gather_indices)
        # tf.reduce_max(tf.abs(from_neighs), axis=1)
#         from_neighs = tf.gather_nd(from_neighs,  tf.stack((tf.range(from_neighs.get_shape()[0], dtype=max_indices.dtype), tf.range(from_neighs.get_shape()[1], dtype=max_indices.dtype), max_indices), axis=1) ) # tf.reduce_max(tf.abs(from_neighs), axis=1)

        
        if(self.input_dim == self.output_dim):
            from_self = self_vecs   
        else:
            from_self = self_vecs[:, 0:self.output_dim]
               
        
#         neigh_h = neigh_vecs
# 
#         dims = tf.shape(neigh_h)
#         batch_size = dims[0]
#         num_neighbors = dims[1]
#         
#         h_reshaped = tf.reshape(neigh_h, (batch_size * num_neighbors, self.neigh_input_dim))
# 
#         for l in self.mlp_layers:
#             h_reshaped = l(h_reshaped)
#         
#         neigh_h = tf.reshape(h_reshaped, (batch_size, num_neighbors, self.hidden_dim))
#         neigh_h = tf.reduce_max(neigh_h, axis=1)
#         
#         from_neighs = tf.matmul(neigh_h, self.vars['neigh_weights'])
#         from_self = tf.matmul(self_vecs, self.vars["self_weights"])
        
        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)
            
        # bias
        if self.bias:
            output += tf.cast(self.vars['bias'], tf.complex64)
       
        return tf.complex(self.act(tf.real(output)), self.act(tf.imag(output))) 
    
    
class GCNAggregator(Layer):
    """
    Aggregates via mean followed by matmul and non-linearity.
    Same matmul parameters are used self vector and neighbor vectors.
    """

    def __init__(self, input_dim, output_dim, adj, neigh_input_dim=None,
            dropout=0., bias=None, act=tf.nn.relu, name=None, concat=True, fixed_neigh_weights=None, variables=None, **kwargs):
        super(GCNAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        if(bias is None):
            self.bias = False
        else:
            self.bias = True
            self.vars['bias'] = bias
        self.act = act
        self.concat = concat
        self.adj = adj
        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.vars['neigh_weights'] = variables if variables is not None else glorot([neigh_input_dim, output_dim], name='neigh_weights')
#         with tf.variable_scope(self.name + name + '_variables'):
#             if self.bias:
#                 self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_variables()

        
    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs

        neigh_vecs = tf.nn.dropout(neigh_vecs, 1-self.dropout)
        self_vecs = tf.nn.dropout(self_vecs, 1-self.dropout)
        
#         neigh_vecs = tf.py_func(assign_array, [neigh_vecs, self.adj, 0], tf.float32) 
        expanded_adj = tf.tile (tf.expand_dims(self.adj, -1) , [1, 1, self.input_dim])
        neigh_vecs = tf.complex(expanded_adj * tf.real(neigh_vecs),  expanded_adj * tf.imag(neigh_vecs))
        means = tf.reduce_sum(tf.concat([neigh_vecs, tf.expand_dims(self_vecs, axis=1)], axis=1), 
                                axis=1)/tf.reshape(tf.reduce_sum(self.adj, axis=1), (-1,1))
        
        # [nodes] x [out_dim]
        output = tf.matmul(means, self.vars['neigh_weights'])

        # bias
        if self.bias:
            output += self.vars['bias']
       
        return self.act(output)




class MeanPoolingAggregator(Layer):
    """ Aggregates via mean-pooling over MLP functions.
    """
    def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None,
            dropout=0., bias=False, act=tf.nn.relu, name=None, concat=True, fixed_neigh_weights=None, variables=None, **kwargs):
        super(MeanPoolingAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''
        self.vars['neigh_weights'] = variables if variables is not None else glorot([neigh_input_dim, output_dim], name='neigh_weights')
        if model_size == "small":
            hidden_dim = self.hidden_dim = 512
        elif model_size == "big":
            hidden_dim = self.hidden_dim = 1024

        self.mlp_layers = []
        self.mlp_layers.append(Dense(input_dim=neigh_input_dim,
                                 output_dim=hidden_dim,
                                 act=tf.nn.relu,
                                 dropout=dropout,
                                 sparse_inputs=False,
                                 logging=self.logging))

        with tf.variable_scope(self.name + name + '_variables'):
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_variables()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs
        
#         dims = tf.shape(neigh_h)
#         batch_size = dims[0]
#         num_neighbors = dims[1]
#         # [nodes * sampled neighbors] x [hidden_dim]
#         h_reshaped = tf.reshape(neigh_h, (batch_size * num_neighbors, self.neigh_input_dim))
# 
#         for l in self.mlp_layers:
#             h_reshaped = l(h_reshaped)
#         neigh_h = tf.reshape(h_reshaped, (batch_size, num_neighbors, self.hidden_dim))

#         neigh_h = tf.py_func(assign_array, [neigh_vecs, self.adj, 0], tf.float64) 
        expanded_adj = tf.tile (tf.expand_dims(self.adj, -1) , [1, 1, self.input_dim])
        neigh_vecs = tf.complex(expanded_adj * tf.real(neigh_vecs),  expanded_adj * tf.imag(neigh_vecs))
        neigh_h = tf.reduce_mean(neigh_vecs, axis=1)
        
        from_neighs = tf.matmul(neigh_h, self.vars['neigh_weights'])
        if(self.input_dim == self.output_dim):
            from_self = self_vecs   
        else:
            from_self = self_vecs[:,0:self.output_dim]
        
        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']
       
        return self.act(output)


class TwoMaxLayerPoolingAggregator(Layer):
    """ Aggregates via pooling over two MLP functions.
    """
    def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None,
            dropout=0., bias=False, act=tf.nn.relu, name=None, concat=True, fixed_neigh_weights=None, variables=None, **kwargs):
        super(TwoMaxLayerPoolingAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''
        self.vars['neigh_weights'] = variables
        if model_size == "small":
            hidden_dim_1 = self.hidden_dim_1 = 512
            hidden_dim_2 = self.hidden_dim_2 = 256
        elif model_size == "big":
            hidden_dim_1 = self.hidden_dim_1 = 1024
            hidden_dim_2 = self.hidden_dim_2 = 512

        self.mlp_layers = []
        self.mlp_layers.append(Dense(input_dim=neigh_input_dim,
                                 output_dim=hidden_dim_1,
                                 act=tf.nn.relu,
                                 dropout=dropout,
                                 sparse_inputs=False,
                                 logging=self.logging))
        self.mlp_layers.append(Dense(input_dim=hidden_dim_1,
                                 output_dim=hidden_dim_2,
                                 act=tf.nn.relu,
                                 dropout=dropout,
                                 sparse_inputs=False,
                                 logging=self.logging))


        with tf.variable_scope(self.name + name + '_variables'):
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_variables()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs
        neigh_h = neigh_vecs

#         dims = tf.shape(neigh_h)
#         batch_size = dims[0]
#         num_neighbors = dims[1]
#         # [nodes * sampled neighbors] x [hidden_dim]
#         h_reshaped = tf.reshape(neigh_h, (batch_size * num_neighbors, self.neigh_input_dim))
# 
#         for l in self.mlp_layers:
#             h_reshaped = l(h_reshaped)
#         neigh_h = tf.reshape(h_reshaped, (batch_size, num_neighbors, self.hidden_dim_2))
        neigh_vecs = tf.py_func(assign_array, [neigh_vecs, self.adj, -1e10], tf.float64) 
        neigh_h = tf.reduce_max(neigh_h, axis=1)
        
        from_neighs = tf.matmul(neigh_h, self.vars['neigh_weights'])
        if(self.input_dim == self.output_dim):
            from_self = self_vecs   
        else:
            from_self = self_vecs[:,0:self.output_dim]
        
        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']
       
        return self.act(output)

class SeqAggregator(Layer):
    """ Aggregates via a standard LSTM.
    """
    def __init__(self, input_dim, output_dim, model_size="small", neigh_input_dim=None,
            dropout=0., bias=False, act=tf.nn.relu, name=None,  concat=True, fixed_neigh_weights=None, **kwargs):
        super(SeqAggregator, self).__init__(**kwargs)

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        if name is not None:
            name = '/' + name
        else:
            name = ''
        self.vars['neigh_weights'] = variables
        if model_size == "small":
            hidden_dim = self.hidden_dim = 128
        elif model_size == "big":
            hidden_dim = self.hidden_dim = 256

        with tf.variable_scope(self.name + name + '_variables'):
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if self.logging:
            self._log_variables()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neigh_input_dim = neigh_input_dim
        self.cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_dim)

    def _call(self, inputs):
        self_vecs, neigh_vecs = inputs

        dims = tf.shape(neigh_vecs)
        batch_size = dims[0]
        initial_state = self.cell.zero_state(batch_size, tf.float32)
        used = tf.sign(tf.reduce_max(tf.abs(neigh_vecs), axis=2))
        length = tf.reduce_sum(used, axis=1)
        length = tf.maximum(length, tf.constant(1.))
        length = tf.cast(length, tf.int32)

        with tf.variable_scope(self.name) as scope:
            try:
                rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
                        self.cell, neigh_vecs,
                        initial_state=initial_state, dtype=tf.float32, time_major=False,
                        sequence_length=length)
            except ValueError:
                scope.reuse_variables()
                rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
                        self.cell, neigh_vecs,
                        initial_state=initial_state, dtype=tf.float32, time_major=False,
                        sequence_length=length)
        
        batch_size = tf.shape(rnn_outputs)[0]
        max_len = tf.shape(rnn_outputs)[1]
        out_size = int(rnn_outputs.get_shape()[2])
        index = tf.range(0, batch_size) * max_len + (length - 1)
        flat = tf.reshape(rnn_outputs, [-1, out_size])
        neigh_h = tf.gather(flat, index)

        from_neighs = tf.matmul(neigh_h, self.vars['neigh_weights'])
        if(self.input_dim == self.output_dim):
            from_self = self_vecs   
        else:
            from_self = self_vecs[:,0:self.output_dim]
         
        output = tf.add_n([from_self, from_neighs])

        if not self.concat:
            output = tf.add_n([from_self, from_neighs])
        else:
            output = tf.concat([from_self, from_neighs], axis=1)

        # bias
        if self.bias:
            output += self.vars['bias']
       
        return self.act(output)

