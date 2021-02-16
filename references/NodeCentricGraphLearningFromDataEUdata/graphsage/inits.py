import tensorflow as tf
import numpy as np

# DISCLAIMER:
# Parts of this code file are derived from
# https://github.com/tkipf/gcn
# which is under an identical MIT license as GraphSAGE

def uniform(shape, scale=0.05, name=None):
    """Uniform init."""
    initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    return tf.Variable(initial, name=name)


# def tf_repeat(arr, repeats):
#     # get maximum repeat length in x
#     maxlen = np.max(repeats)
#     # tile it to the maximum repeat length, it should be of shape [xlen, maxlen] now
#     arr_tiled = tf.tile(tf.expand_dims(arr, 1), [1, maxlen])
#     # create a sequence mask using x
#     # this will create a boolean matrix of shape [xlen, maxlen]
#     # where result[i,j] is true if j < x[i].
#     mask = tf.sequence_mask(repeats, maxlen)
#     # mask the elements based on the sequence mask
#     return tf.boolean_mask(arr_tiled, mask)


# def tf_repeat(arr, repeats):
#     return tf.py_func(np.repeat, [arr, repeats], tf.float32)
    

def tf_repeat(arr, repeats):
    repeats = repeats.astype(np.int64)
    arr = tf.expand_dims(arr, 1)
    max_repeats = np.max(repeats)
    tile_repeats = [1, max_repeats]
    # If you want to support not just 1D arrays but any number of dimensions you can do:
    # tile_repeats = tf.concat(
    #     [[1], [max_repeats], tf.ones([tf.rank(arr) - 2], dtype=tf.int32)], axis=0)
    arr_tiled = tf.tile(arr, tile_repeats)
    mask = tf.less(np.arange(max_repeats), tf.expand_dims(repeats, 1))
    result = tf.reshape(tf.boolean_mask(arr_tiled, mask), [-1])
    return result

# def tf_repeat(tensor, repeats):
#     """
#     Args:
# 
#     input: A Tensor. 1-D or higher.
#     repeats: A list. Number of repeat for each dimension, length must be the same as the number of dimensions in input
# 
#     Returns:
#     
#     A Tensor. Has the same type as input. Has the shape of tensor.shape * repeats
#     """
#     with tf.variable_scope("repeat"):
#         expanded_tensor = tf.expand_dims(tensor, -1)
#         multiples = [1] + repeats
#         tiled_tensor = tf.tile(expanded_tensor, multiples = multiples)
#         repeated_tesnor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
#     return repeated_tesnor

def tf_repeat_1D(x,repeats):
#     x = tf.constant(x, dtype=tf.float64)
#     repeats = tf.constant(repeats, dtype=tf.int32)
    shape = tf.reduce_sum(repeats)
    idx = tf.concat([tf.constant([0], dtype=tf.int32), tf.cumsum(repeats[:-1])], axis=0)
    y = tf.sparse_to_dense(
        sparse_indices = idx,
        output_shape=(shape,),
        sparse_values=x - tf.concat([tf.constant([0], dtype=tf.float64), x[:-1]], axis=0)
    )
    return tf.cumsum(y)


def repeated_variable(ssizes, ffunc=None, name=None):
#     initt = tf.random_uniform((tf.size(ssizes),)) # tf.ones([tf.size(ssizes),], dtype=tf.float64)
    shape = [np.size(ssizes),]
    init_range = np.sqrt(6.0/(shape[0]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    vars_Theta_weights = tf.Variable(initial, name=name)
    if(ffunc is not None):
        vars_Theta_weights = ffunc(vars_Theta_weights)
    return tf_repeat(vars_Theta_weights, ssizes) 
    # [item for item, count in zip(self.vars_Theta_weights, self.graphL_core.conv_sizes) for i in range(count)]


def repeated2D_variable(ssizes, name=None):
    shape = [np.size(ssizes),np.size(ssizes)]
    init_range = np.sqrt(6.0/(shape[0]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    varArray = tf.Variable(initial, name=name)
    tf_repeat(varArray, ssizes)
    return  



def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(np.sum(shape)))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
#     initial = tf.ones(shape)
    return tf.Variable(initial, name=name)



def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)

def ones(shape, name=None):
    """All ones."""
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)
