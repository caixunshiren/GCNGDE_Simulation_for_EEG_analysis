from __future__ import division
from __future__ import print_function

from graphsage.inits import zeros
from graphsage.layers import Layer
import tensorflow as tf
from graphsage.utils import SDP_solver, Brain_Parameter_Solver
from .inits import glorot, zeros
import sys

flags = tf.compat.v1.flags # tf.app.flags
FLAGS = flags.FLAGS


class BipartiteEdgePredLayer(Layer):
    def __init__(self, input_dim1, input_dim2, placeholders, params, fixed_theta_1 = None, dropout=False, act=tf.nn.sigmoid,
            loss_fn='custom_loss', neg_sample_weights=1.0,
            bias=False, bilinear_weights=False, **kwargs):
        """
        Basic class that applies skip-gram-like loss
        (i.e., dot product of node+target and node and negative samples)
        Args:
            bilinear_weights: use a bilinear weight for affinity calculation: u^T A v. If set to
                false, it is assumed that input dimensions are the same and the affinity will be 
                based on dot product.
        """
        super(BipartiteEdgePredLayer, self).__init__(**kwargs)
        self.fixed_theta_1 = fixed_theta_1
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.act = act
        self.bias = bias
        self.eps = 1e-7

        # Margin for hinge loss
        self.margin = 0.1
        self.neg_sample_weights = neg_sample_weights

        self.bilinear_weights = bilinear_weights

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        # output a likelihood term
        self.output_dim = 1
        self.vars['theta_1'] = glorot([input_dim1, input_dim2], name='theta_1')
        with tf.variable_scope(self.name + '_vars'):
            # bilinear form
            if bilinear_weights:
                self.vars['weights'] = tf.get_variable(
                        'pred_weights', 
                        shape=(input_dim1, input_dim2),
                        dtype=tf.float32, 
                        initializer=tf.contrib.layers.xavier_initializer())

            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')

        if loss_fn == 'xent':
            self.loss_fn = self._xent_loss
        elif loss_fn == 'skipgram':
            self.loss_fn = self._skipgram_loss
        elif loss_fn == 'hinge':
            self.loss_fn = self._hinge_loss
        elif loss_fn == 'custom_loss':
            self.loss_fn = self._custom_loss 
        elif loss_fn == 'brain_hybrid_loss':
            self.loss_fn = self._brain_hybrid_loss 
            
        
        if self.logging:
            self._log_vars()
            
        #changed from here
        self.params = params

    
    def neg_cost(self, inputs1, neg_samples, hard_neg_samples=None):
        """ For each input in batch, compute the sum of its affinity to negative samples.

        Returns:
            Tensor of shape [batch_size x num_neg_samples]. For each node, a list of affinities to
                negative samples is computed.
        """
        if self.bilinear_weights:
            inputs1 = tf.matmul(inputs1, self.vars['weights'])
        neg_aff = tf.matmul(inputs1, tf.matmul(self.theta_1, tf.transpose(neg_samples))) # cent_abs_affinity # self.affinity(inputs1, neg_samples) #

        return neg_aff

    def loss(self, inputs1, inputs2, neg_samples, neg_samples2=None, batch_size=None, negbatch_size=None):
        """ negative sampling loss.
        Args:
            neg_samples: tensor of shape [num_neg_samples x input_dim2]. Negative samples for all
            inputs in batch inputs1.
        """
        return self.loss_fn(inputs1, inputs2, neg_samples, neg_samples2=neg_samples2, batch_size=batch_size, negbatch_size=negbatch_size)
    
    def similarity(self, inputs1, inputs2):# changed, added this function
        
        return self.similarity_fn(inputs1, inputs2)
    
    
    def _xent_loss(self, inputs1, inputs2, neg_samples, hard_neg_samples=None):
        aff, _ = self.affinity(inputs1, inputs2)
        neg_aff = self.neg_cost(inputs1, neg_samples, hard_neg_samples)
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(aff), logits=aff)
        negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(neg_aff), logits=neg_aff)
        loss = tf.reduce_sum(true_xent) + self.neg_sample_weights * tf.reduce_sum(negative_xent)
        return loss

    def _skipgram_loss(self, inputs1, inputs2, neg_samples, hard_neg_samples=None):
        aff, _ = self.affinity(inputs1, inputs2)
        neg_aff = self.neg_cost(inputs1, neg_samples, hard_neg_samples)
        neg_cost = tf.log(tf.reduce_sum(tf.exp(neg_aff), axis=1))
        loss = tf.reduce_sum(aff - neg_cost)
        return loss
    
    def _hinge_loss(self, inputs1, inputs2, neg_samples, hard_neg_samples=None):
        aff, _ = self.affinity(inputs1, inputs2)
        neg_aff = self.neg_cost(inputs1, neg_samples, hard_neg_samples)
        diff = tf.nn.relu(tf.subtract(neg_aff, tf.expand_dims(aff, 1) - self.margin), name='diff')
        loss = tf.reduce_sum(diff)
        self.neg_shape = tf.shape(neg_aff)
        return loss
    
    def affinity(self, inputs1, inputs2):
        """ Affinity score between batch of inputs1 and inputs2.
        Args:
            inputs1: tensor of shape [batch_size x feature_size].
        """
        if self.bilinear_weights:
            prod = tf.matmul(inputs2, tf.transpose(self.vars['weights']))
            self.prod = prod
            result = tf.reduce_sum(inputs1 * prod, axis=1)
        prod = tf.matmul( inputs1, tf.conj(tf.matmul(self.theta_1, tf.transpose(inputs2)) ) )
        result = tf.diag_part(prod)
        return tf.transpose(result), self.theta_1
    
    
    
    def cent_abs_affinity(self, inputs1, inputs2):
#         theta_1 = SDP_solver(inputs1, inputs2, self.input_dim1, self.input_dim2)
        inputs1_cent = inputs1 - tf.reduce_mean(inputs1, axis=0)
        inputs2_cent = inputs2 - tf.reduce_mean(inputs2, axis=0)
        prod = tf.abs(tf.matmul( inputs1_cent, tf.matmul(self.theta_1, tf.transpose(inputs2_cent)))) 
        return tf.diag_part(prod), self.theta_1
        
    def brain_affinity(self, inputs1, inputs2):
        result = 0
        pre_index = 0
        for ccounter in range(len(FLAGS.brain_similarity_sizes)):
            ssize = FLAGS.brain_similarity_sizes[ccounter]
            result += self.brain_affinity_parameters[ccounter] * tf.abs(tf.matmul( inputs1[:,pre_index:pre_index+ssize], tf.conj(tf.transpose(inputs2[:,pre_index:pre_index+ssize]))))
            pre_index = ssize
        
        if(pre_index!=int(self.input_dim1)):
            sys.exit("Error in Brain Affinity, sizes and D are not equal")
        return result, self.theta_1
    
    
    def _brain_hybrid_loss(self, inputs1, inputs2, neg_samples, neg_samples2=None, hard_neg_samples=None, batch_size=None, negbatch_size=None):
        self.brain_affinity_parameters = Brain_Parameter_Solver(inputs1/tf.cast(batch_size, tf.float32), inputs2 , self.input_dim1, self.input_dim2,\
                                                                 Z3=self.neg_sample_weights * neg_samples/tf.cast(negbatch_size, tf.float32), Z4=neg_samples2)
        prod1, _ = self.brain_affinity(inputs1, inputs2)
        if(FLAGS.model_size == 'big'):
            sys.exit("brain hybrid loss must be defined for 'big' graphs")
        else:
            prod2, _ = self.brain_affinity(neg_samples, neg_samples2)
        loss = - tf.reduce_sum(prod1/tf.cast(batch_size, tf.float32)) 
        loss += self.neg_sample_weights * tf.reduce_sum(prod2/tf.cast(negbatch_size, tf.float32))
        
            
            
            
    def _custom_loss(self, inputs1, inputs2, neg_samples, neg_samples2=None, hard_neg_samples=None, batch_size=None, negbatch_size=None): # changed, added this function
        
        mode = 0
        
        
        if(self.fixed_theta_1 is not None):
            self.theta_1 = self.fixed_theta_1 
            
        else:
            
            if(FLAGS.theta_exist):
                if(mode ==0 or mode==2):
                    self.theta_1 = SDP_solver(inputs1/tf.cast(batch_size, tf.float32), inputs2 , self.input_dim1, self.input_dim2, Z3=self.neg_sample_weights * neg_samples/tf.cast(negbatch_size, tf.float32), Z4=neg_samples2)  # tf.eye(self.input_dim1) #     
                else:                        
                    self.theta_1 = SDP_solver(inputs1, inputs2, self.input_dim1, self.input_dim2, Z3=self.neg_sample_weights * neg_samples, Z4=neg_samples2) # tf.eye(self.input_dim1) #
            else:
                self.theta_1 = tf.eye(self.input_dim1)      

###                 self.vars['theta_1'] # tf.eye(self.input_dim1) #  
            
        prod1, _ = self.affinity(inputs1, inputs2)
        
            
        if(FLAGS.model_size == 'big'):
#             self.theta_1 = tf.eye(self.input_dim1)
            prod2 = self.neg_cost(inputs1, neg_samples, hard_neg_samples)
            negbatch_size = tf.size(prod2) # batch_size * tf.size(neg_samples)       
#             prod2 = tf.reshape(prod2, (tf.size(prod2),))
#             neg_aff = self.neg_cost(inputs1, neg_samples, hard_neg_samples)
#             true_xent = - prod1 # tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(prod1), logits=prod1)
#             negative_xent = neg_aff # tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(neg_aff), logits=neg_aff)
#             loss = tf.reduce_sum(true_xent) + self.neg_sample_weights * tf.reduce_sum(negative_xent)
#             loss = loss / tf.cast(batch_size, tf.float32)
        else:
#             self.theta_1 = tf.eye(self.input_dim1)
            prod2, _ = self.affinity(neg_samples, neg_samples2)
        
        
        if(mode==0):
            loss = - tf.reduce_sum(prod1/tf.cast(batch_size, tf.float32)) 
            loss += self.neg_sample_weights * tf.reduce_sum(prod2/tf.cast(negbatch_size, tf.float32))
        
        
        elif(mode==1):
            loss = - tf.reduce_sum(prod1) 
            loss += self.neg_sample_weights * tf.reduce_sum(prod2)
        
        
        elif(mode==2): # worked
            loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits( labels=tf.ones_like(prod1), logits=prod1)/tf.cast(batch_size, tf.float32))  
            loss += self.neg_sample_weights * tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(prod2), logits=prod2)/tf.cast(negbatch_size, tf.float32))
        
        
        elif(mode==3):
            loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits( labels=tf.ones_like(prod1), logits=prod1))  
            loss += self.neg_sample_weights * tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(prod2), logits=prod2))
        
        
        elif(mode==4):
            diff = tf.nn.relu(prod2 - prod1 + self.margin, name='diff')
            loss = tf.reduce_sum(diff)

        return loss
#     def custom_affinity(self, inputs1, inputs2, neg_samples1, neg_samples2=None):
    
    
    
    def weights_norm(self):
        return tf.nn.l2_norm(self.vars['weights'])

