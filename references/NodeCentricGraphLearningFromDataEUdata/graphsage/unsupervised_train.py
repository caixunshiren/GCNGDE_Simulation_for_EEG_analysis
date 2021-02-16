from __future__ import division
from __future__ import print_function

import os
import time
import tensorflow as tf
import numpy as np

from graphsage.models import SampleAndAggregate, SAGEInfo, Node2VecModel
from graphsage.minibatch import EdgeMinibatchIterator
from graphsage.neigh_samplers import UniformNeighborSampler
from graphsage.utils import load_data
from graphsage.utils import normal_SDP_solver, Quadratic_SDP_solver, Graph_complement

import matplotlib.pyplot as plt
import networkx as nx
#from sage.all import DiGraph

data_dir = "example_data/detection"
#input_data_prefix = "example_data/"+ target + '_' + 'ictal' #ppi, ictal, interictal
            

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

# Set random seed
seed = 123
np.random.seed(seed)
tf.random.set_seed(seed) # tf.set_random_seed(seed)
small_big_threshold = 2000
# Settings
flags = tf.compat.v1.flags # tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('train_prefix', data_dir, 'name of the object file that stores the training data. must be specified.')
tf.compat.v1.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")  # tf.app.flags
#core params..
flags.DEFINE_float('learning_rate', 0.005, 'initial learning rate.') # changed from  0.00001 to 0.001
flags.DEFINE_string("model_size", "small", "Can be big or small; model specific def'ns")



flags.DEFINE_string('model', 'graphsage_maxpool', 'model names. See README for possible values.')
flags.DEFINE_string('loss_function', 'custom_loss', 'type of loss function')

flags.DEFINE_integer('epochs', 5, 'number of epochs to train.') # changed 1 to 
flags.DEFINE_float('dropout', 0.0, 'dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.0, 'weight for l2 loss on embedding matrix.')
flags.DEFINE_integer('max_degree', 100, 'maximum node degree.')
flags.DEFINE_integer('samples_1', 20, 'number of samples in layer 1') 
flags.DEFINE_integer('samples_2', 10, 'number of users samples in layer 2') 
flags.DEFINE_integer('negsamples_1', 20, 'number of neg samples in layer 1') 
flags.DEFINE_integer('negsamples_2', 10, 'number of users neg samples in layer 2') 
flags.DEFINE_integer('dim_1', 3 , 'Size of output dim (final is 2x this, if using concat)') # changed 128 
flags.DEFINE_integer('dim_2', 3 , 'Size of output dim (final is 2x this, if using concat)')  # changed 128 
flags.DEFINE_boolean('random_context', False, 'Whether to use random context or direct edges') # changed True to False
flags.DEFINE_integer('neg_sample_size', 20, 'number of negative samples')
flags.DEFINE_integer('batch_size', 512, 'minibatch size.') # changed 512 to 8
flags.DEFINE_integer('negbatch_size', 512, 'minibatch size.')
flags.DEFINE_integer('n2v_test_epochs', 1, 'Number of new SGD epochs for n2v.')
flags.DEFINE_integer('identity_dim', 0, 'Set to positive value to use identity embedding features of that dimension. Default 0.')
flags.DEFINE_boolean('theta_exist', False , 'Theta is added or not') # changed 128 
flags.DEFINE_boolean('flag_normalized', False , 'Z becomes normalized') # changed 128 
flags.DEFINE_multi_integer('brain_similarity_sizes', [1,1,1], 'size of different parts in initial featuer vector which will lead to correlation, coherence, PLV')
#logging, saving, validation settings etc.
flags.DEFINE_boolean('save_embeddings', False, 'whether to save embeddings for all nodes after training')# changed True to False
flags.DEFINE_string('base_log_dir', '.', 'base directory for logging and saving embeddings')
flags.DEFINE_integer('validate_iter', 5000, "how often to run a validation minibatch.")
flags.DEFINE_integer('validate_batch_size', 256, "how many nodes per validation sample.")
flags.DEFINE_integer('gpu', 1, "which gpu to use.")
flags.DEFINE_integer('print_every', 10, "How often to print training info.")
flags.DEFINE_integer('max_total_steps', 10**10, "Maximum total number of iterations")

os.environ["CUDA_VISIBLE_DEVICES"]=str(FLAGS.gpu)

GPU_MEM_FRACTION = 0.8

def log_dir():
    log_dir = FLAGS.base_log_dir + "/unsup-" + FLAGS.train_prefix.split("/")[-2] # pure_filename 
    log_dir += "/{model:s}_{model_size:s}_{lr:0.6f}/".format(
            model=FLAGS.model,
            model_size=FLAGS.model_size,
            lr=FLAGS.learning_rate)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


# Define model evaluation function
def evaluate(sess, model, minibatch_iter, size=None, negminibatch_iter=None, placeholders=None):
    t_test = time.time()
    feed_dict = minibatch_iter.val_feed_dict(size)
    negfeed_dict = negminibatch_iter.val_feed_dict(size)
    if(negminibatch_iter is not None):
#         feed_dict.update({placeholders['dropout']: FLAGS.dropout})
#         feed_dict.update({placeholders['all_nodes']: all_nodes})
        feed_dict.update({placeholders['negbatch1']: negfeed_dict[placeholders['batch1']]})
        feed_dict.update({placeholders['negbatch2']: negfeed_dict[placeholders['batch2']]})
        feed_dict.update({placeholders['negbatch_size']: negfeed_dict[placeholders['batch_size']]})
    outs_val = sess.run([model.loss, model.ranks, model.mrr], 
                        feed_dict=feed_dict)
    return outs_val[0], outs_val[1], outs_val[2], (time.time() - t_test)

def incremental_evaluate(sess, model, minibatch_iter, size):
    t_test = time.time()
    finished = False
    val_losses = []
    val_mrrs = []
    iter_num = 0
    while not finished:
        feed_dict_val, finished, _ = minibatch_iter.incremental_val_feed_dict(size, iter_num)
        iter_num += 1
        outs_val = sess.run([model.loss, model.ranks, model.mrr], 
                            feed_dict=feed_dict_val)
        val_losses.append(outs_val[0])
        val_mrrs.append(outs_val[2])
    return np.mean(val_losses), np.mean(val_mrrs), (time.time() - t_test)

def save_val_embeddings(sess, model, minibatch_iter, size, out_dir, mod=""):
    val_embeddings = []
    finished = False
    seen = set([])
    nodes = []
    iter_num = 0
    name = "val"
    while not finished:
        feed_dict_val, finished, edges = minibatch_iter.incremental_embed_feed_dict(size, iter_num)
        iter_num += 1
        outs_val = sess.run([model.loss, model.mrr, model.outputs1], 
                            feed_dict=feed_dict_val)
        #ONLY SAVE FOR embeds1 because of planetoid
        for i, edge in enumerate(edges):
            if not edge[0] in seen:
                val_embeddings.append(outs_val[-1][i,:])
                nodes.append(edge[0])
                seen.add(edge[0])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    val_embeddings = np.vstack(val_embeddings)
    np.save(out_dir + name + mod + ".npy",  val_embeddings)
    with open(out_dir + name + mod + ".txt", "w") as fp:
        fp.write("\n".join(map(str,nodes)))

def construct_placeholders():
    # Define placeholders
    placeholders = {
        'batch1' : tf.placeholder(tf.int32, shape=(None), name='batch1'),
        'batch2' : tf.placeholder(tf.int32, shape=(None), name='batch2'),
        'negbatch1' : tf.placeholder(tf.int32, shape=(None), name='negbatch1'),
        'negbatch2' : tf.placeholder(tf.int32, shape=(None), name='negbatch2'),
        # negative samples for all nodes in the batch
        'neg_samples': tf.placeholder(tf.int32, shape=(None,),
            name='neg_sample_size'),
        'dropout': tf.placeholder_with_default(0., shape=(), name='dropout'),
        'batch_size' : tf.placeholder(tf.int32, name='batch_size'),
        'negbatch_size' : tf.placeholder(tf.int32, name='negbatch_size'),
        'all_nodes' : tf.placeholder(tf.int32, shape=(None), name='all_nodes'),
        #'adj_info_ph' : tf.placeholder(tf.int32, shape=(None), name="adj_info_ph")
        'agg_batch_Z1' : tf.placeholder(tf.int32, shape=(None), name='agg_batch_Z1'),
        'agg_batch_Z2' : tf.placeholder(tf.int32, shape=(None), name='agg_batch_Z2'),
        'agg_batch_Z3' : tf.placeholder(tf.int32, shape=(None), name='agg_batch_Z3'),
        'agg_batch_Z4' : tf.placeholder(tf.int32, shape=(None), name='agg_batch_Z4'),
    }
    return placeholders

def print_summary(model, feed_dict, sess, Adj_mat, title, print_flag):
    loss, final_theta_1, Z, neigh1, neigh2 = sess.run([model.loss_agg, model.theta_1, model.aggregation\
                                                    , model.aggregators[0].vars['neigh_weights'], model.aggregators[1].vars['neigh_weights']], feed_dict=feed_dict)
    U = []
    U.append(neigh1)
    U.append(neigh2)
    
#     Z = Z - np.mean(Z, axis=1)[:,np.newaxis]
#     Z = Z/np.linalg.norm(Z, axis=1)[:,np.newaxis]
    final_adj_matrix = np.matmul(Z,np.matmul(final_theta_1, Z.T))
    final_adj_matrix = (final_adj_matrix+final_adj_matrix.T)/2
#     neighbour_weights = np.where( Adj_mat==1, final_adj_matrix, 0)
#     non_neighbour_weights = np.where(Adj_mat==0, final_adj_matrix, 0)
    if(print_flag):
        print(title)
        print('    mean of neigh= %f , nonneigh= %f: ' % (np.sum(np.where( Adj_mat==1, final_adj_matrix, 0))/np.sum(np.where( Adj_mat==1, 1, 0)),\
                                                           np.sum(np.where( Adj_mat==0, final_adj_matrix, 0))/np.sum(np.where( Adj_mat==0, 1, 0))))
        print('    max of neigh= %f , nonneigh= %f: ' % (np.max(np.where( Adj_mat==1, final_adj_matrix, -1e10)), np.max(np.where( Adj_mat==0, final_adj_matrix, -1e10))))
        print('    min of neigh= %f , nonneigh= %f: ' % (np.min(np.where( Adj_mat==1, final_adj_matrix, 1e10)), np.min(np.where( Adj_mat==0, final_adj_matrix, 1e10))))
    return final_adj_matrix, final_theta_1, Z, loss, U

def Edges_to_Adjacency_mat(edges, N):
    adj = np.zeros((N,N))
    for node1,node2 in edges:
        adj[node1,node2] = 1
        adj[node2,node1] = 1
    return adj

def largest_prime_factor(n):
    i = 2
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
    return n

def batch_size_def(n): 
    for i in np.arange(int(n/500)+1)+1:
        if(n<512*i):
            return int(n/i)


def train(train_data, log_dir, Theta=None, fixed_neigh_weights=None, test_data=None, neg_sample_weights = None):
    G = train_data[0]
    Gneg = Graph_complement(G) 
    features = train_data[1]
    id_map = train_data[2]
    Adj_mat = Edges_to_Adjacency_mat(G.edges(), len(G.nodes()))
#     print('A in unsup: ', Adj_mat)
#     negAdj_mat = Edges_to_Adjacency_mat(Gneg.edges(), len(Gneg.nodes()))
    FLAGS.batch_size = batch_size_def(len(G.edges()))
    FLAGS.negbatch_size = batch_size_def(len(Gneg.edges()))
    FLAGS.samples_1 = min(25, len(G.nodes()))
    FLAGS.samples_2 = min(10, len(G.nodes()))
    FLAGS.negsamples_1 = min(25, len(Gneg.nodes()))
    FLAGS.negsamples_2 = min(10, len(Gneg.nodes()))
    FLAGS.neg_sample_size = FLAGS.batch_size
    
    if(len(G.nodes())<=small_big_threshold):
        FLAGS.model_size = 'small'
    else:
        FLAGS.model_size = 'big'
    
    if not features is None:
        # pad with dummy zero vector
        features = np.vstack([features, np.zeros((features.shape[1],))])

    context_pairs = train_data[3] if FLAGS.random_context else None
    placeholders = construct_placeholders()
    #print('placeholders: ', placeholders)
    minibatch = EdgeMinibatchIterator(G, 
            id_map,
            placeholders, batch_size = FLAGS.batch_size,
            max_degree = FLAGS.max_degree, 
            num_neg_samples = FLAGS.neg_sample_size,
            context_pairs = context_pairs)
    aggbatch_size = len(minibatch.agg_batch_Z1)
    negaggbatch_size = len(minibatch.agg_batch_Z3)
    negminibatch = EdgeMinibatchIterator(Gneg, 
            id_map,
            placeholders, batch_size = FLAGS.negbatch_size,
            max_degree = FLAGS.max_degree, 
            num_neg_samples = FLAGS.neg_sample_size,
            context_pairs = context_pairs)
        
    #adj_info = tf.Variable(placeholders['adj_info_ph'], trainable=False, name="adj_info") 
    adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.adj.shape, name="adj_info_ph")
    adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")
     
    if FLAGS.model == 'graphsage_mean':
        # Create model
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1, FLAGS.negsamples_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2, FLAGS.negsamples_2)]

        model = SampleAndAggregate(placeholders, 
                                     features,
                                     adj_info,
                                     minibatch.deg,
                                     layer_infos=layer_infos, 
                                     Adj_mat = Adj_mat,
                                     non_edges = Gneg.edges(),
                                     model_size=FLAGS.model_size,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True,  
                                     fixed_theta_1 = Theta, 
                                     fixed_neigh_weights = fixed_neigh_weights,
                                     neg_sample_weights = neg_sample_weights, 
                                     aggbatch_size=aggbatch_size, 
                                     negaggbatch_size=negaggbatch_size)
    elif FLAGS.model == 'gcn':
        # Create model
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, 2*FLAGS.dim_1, FLAGS.negsamples_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, 2*FLAGS.dim_2, FLAGS.negsamples_2)]

        model = SampleAndAggregate(placeholders, 
                                     features,
                                     adj_info,
                                     minibatch.deg,
                                     layer_infos=layer_infos, 
                                     aggregator_type="gcn",
                                     Adj_mat = Adj_mat,
                                     non_edges = Gneg.edges(),
                                     model_size=FLAGS.model_size,
                                     identity_dim = FLAGS.identity_dim,
                                     concat=False,
                                     logging=True,  
                                     fixed_theta_1 = Theta, 
                                     fixed_neigh_weights = fixed_neigh_weights,
                                     neg_sample_weights = neg_sample_weights, 
                                     aggbatch_size=aggbatch_size, 
                                     negaggbatch_size=negaggbatch_size)

    elif FLAGS.model == 'graphsage_seq':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1, FLAGS.negsamples_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2, FLAGS.negsamples_2)]

        model = SampleAndAggregate(placeholders, 
                                     features,
                                     adj_info,
                                     minibatch.deg,
                                     layer_infos=layer_infos, 
                                     identity_dim = FLAGS.identity_dim,
                                     aggregator_type="seq",
                                     Adj_mat = Adj_mat,
                                     non_edges = Gneg.edges(),
                                     model_size=FLAGS.model_size,
                                     logging=True,  
                                     fixed_theta_1 = Theta, 
                                     fixed_neigh_weights = fixed_neigh_weights,
                                     neg_sample_weights = neg_sample_weights, 
                                     aggbatch_size=aggbatch_size, 
                                     negaggbatch_size=negaggbatch_size)

    elif FLAGS.model == 'graphsage_maxpool':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1, FLAGS.negsamples_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2, FLAGS.negsamples_2)]

        model = SampleAndAggregate(placeholders, 
                                    features,
                                    adj_info,
                                    minibatch.deg,
                                    layer_infos=layer_infos, 
                                    aggregator_type="maxpool",
                                    Adj_mat = Adj_mat,
                                    non_edges = Gneg.edges(),
                                    model_size=FLAGS.model_size,
                                    identity_dim = FLAGS.identity_dim,
                                    logging=True,  
                                    fixed_theta_1 = Theta, 
                                    fixed_neigh_weights = fixed_neigh_weights,
                                    neg_sample_weights = neg_sample_weights, 
                                    aggbatch_size=aggbatch_size, 
                                    negaggbatch_size=negaggbatch_size)
        
    elif FLAGS.model == 'graphsage_meanpool':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1, FLAGS.negsamples_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2, FLAGS.negsamples_2)]

        model = SampleAndAggregate(placeholders, 
                                    features,
                                    adj_info,
                                    minibatch.deg,
                                     layer_infos=layer_infos, 
                                     aggregator_type="meanpool",
                                     Adj_mat = Adj_mat,
                                     non_edges = Gneg.edges(),
                                     model_size=FLAGS.model_size,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True,  
                                     fixed_theta_1 = Theta, 
                                     fixed_neigh_weights = fixed_neigh_weights,
                                     neg_sample_weights = neg_sample_weights, 
                                     aggbatch_size=aggbatch_size, 
                                     negaggbatch_size=negaggbatch_size)

    elif FLAGS.model == 'n2v':
        model = Node2VecModel(placeholders, features.shape[0],
                                       minibatch.deg,
                                       #2x because graphsage uses concat
                                       nodevec_dim=2*FLAGS.dim_1,
                                       lr=FLAGS.learning_rate)
    else:
        raise Exception('Error: model name unrecognized.')

    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_FRACTION
    config.allow_soft_placement = True
    
    # Initialize session
    sess = tf.Session(config=config)
    merged = tf.summary.merge_all()
    #summary_writer = tf.summary.FileWriter(log_dir(), sess.graph)
     
    # Init variables
    #minibatch.adj = minibatch.adj.astype(np.int32)
    #print('minibatch.adj.shape: %s, dtype: %s' % (minibatch.adj.shape, np.ndarray.dtype(minibatch.adj)))
    sess.run(tf.global_variables_initializer(), feed_dict={adj_info_ph: minibatch.adj})
    
    # Train model
    
    train_shadow_mrr = None
    shadow_mrr = None

    total_steps = 0
    avg_time = 0.0
    epoch_val_costs = []
    train_adj_info = tf.assign(adj_info, minibatch.adj)
    val_adj_info = tf.assign(adj_info, minibatch.test_adj)
    
    print_flag = False
    if(Theta is None):
        print_flag = True
    
    
    for epoch in range(FLAGS.epochs): 
        
        minibatch.shuffle() 
        negminibatch.shuffle() 
        iter = 0
#         print('Epoch: %04d' % (epoch + 1))
        epoch_val_costs.append(0)
        if(FLAGS.model_size == 'big'):
            whichbatch = minibatch
        elif(len(G.edges())>len(Gneg.edges())):
            whichbatch = minibatch
            opwhichbatch = negminibatch
        else:
            whichbatch = negminibatch   
            opwhichbatch = minibatch
            
        while(not whichbatch.end()):
            if(FLAGS.model_size == 'small' and opwhichbatch.end()):
                opwhichbatch.shuffle()
            # Construct feed dictionary
            
            feed_dict = minibatch.next_minibatch_feed_dict()
            negfeed_dict = negminibatch.next_minibatch_feed_dict()
            
            if(True):
                feed_dict.update({placeholders['negbatch1']: negfeed_dict[placeholders['batch1']]})
                feed_dict.update({placeholders['negbatch2']: negfeed_dict[placeholders['batch2']]})
                feed_dict.update({placeholders['negbatch_size']: negfeed_dict[placeholders['batch_size']]})
            else:
                batch1 = feed_dict[placeholders['batch1']]
                feed_dict.update({placeholders['negbatch1']: batch1})
                feed_dict.update({placeholders['negbatch2']: negminibatch.feed_dict_negbatch(batch1)})
                feed_dict.update({placeholders['negbatch_size']: negfeed_dict[placeholders['batch_size']]})
                
                
            if(Theta is not None):
                break
            if(total_steps==0):
                print_summary(model, feed_dict, sess, Adj_mat, 'first', print_flag)
            
            t = time.time()
            # Training step
            outs = sess.run([merged, model.opt_op, model.loss, model.ranks, model.aff_all, 
                             model.mrr, model.outputs1, model.outputs2, model.negoutputs1, model.negoutputs2], feed_dict=feed_dict) #, model.current_similarity   , model.learned_vars['theta_1']  # , model.aggregation
            train_cost = outs[2]
            train_mrr = outs[5]
            
#             Z = outs[8]
#             outs = np.concatenate((outs[6],outs[7]), axis=0)
#             indices = np.concatenate((feed_dict[placeholders['batch1']],feed_dict[placeholders['batch2']]))
#             Z = np.zeros((len(G.nodes()),FLAGS.dim_2*2))
#             for node in G.nodes():
#                 Z[node,:] = outs[int(np.argwhere(indices==node)[0]),:]#[0:len(G.nodes())] # 8
#             print('Z shape: ', Z.shape)

#             if train_shadow_mrr is None:
#                 train_shadow_mrr = train_mrr #
#             else:
#                 train_shadow_mrr -= (1-0.99) * (train_shadow_mrr - train_mrr)
# 
#             if iter % FLAGS.validate_iter == 0:
#                 # Validation
#                 sess.run(val_adj_info.op)
#                 val_cost, ranks, val_mrr, duration = evaluate(sess, model, minibatch, size=FLAGS.validate_batch_size, negminibatch_iter=negminibatch, placeholders=placeholders)
#                 sess.run(train_adj_info.op)
#                 epoch_val_costs[-1] += val_cost
#                  
#             if shadow_mrr is None:
#                 shadow_mrr = val_mrr
#             else:
#                 shadow_mrr -= (1-0.99) * (shadow_mrr - val_mrr)


#             if total_steps % FLAGS.print_every == 0:
#                 summary_writer.add_summary(outs[0], total_steps)
    
            # Print results
            avg_time = (avg_time * total_steps + time.time() - t) / (total_steps + 1)

            if total_steps % FLAGS.print_every == 0:
                print('loss: ', outs[2])
#                 print("Iter:", '%04d' % iter, 
#                       "train_loss=", "{:.5f}".format(train_cost),
#                       "train_mrr=", "{:.5f}".format(train_mrr), 
#                       "train_mrr_ema=", "{:.5f}".format(train_shadow_mrr), # exponential moving average
#                       "val_loss=", "{:.5f}".format(val_cost),
#                       "val_mrr=", "{:.5f}".format(val_mrr), 
#                       "val_mrr_ema=", "{:.5f}".format(shadow_mrr), # exponential moving average
#                       "time=", "{:.5f}".format(avg_time))
            
#             similarity_weights = outs[7]
#             [new_adj_info, new_batch_edges] = sess.run([model.adj_info, \
#                                                                 model.new_batch_edges], \
#                                                                   feed_dict=feed_dict)
#             minibatch.graph_update(new_adj_info, new_batch_edges)
            
            iter += 1
            total_steps += 1

            if total_steps > FLAGS.max_total_steps:
                break
        
        if(Theta is not None):
                break
        if total_steps > FLAGS.max_total_steps:
                break
           
#     print("SGD Optimization Finished!")
    

#     feed_dict = dict()
#     feed_dict.update({placeholders['batch_size'] : len(G.nodes())})

#     minibatch = EdgeMinibatchIterator(G, 
#             id_map,
#             placeholders, batch_size = len(G.edges()),
#             max_degree = FLAGS.max_degree, 
#             num_neg_samples = FLAGS.neg_sample_size,
#             context_pairs = context_pairs)
#     feed_dict = minibatch.next_minibatch_feed_dict()
#     _, Z = sess.run([merged, model.aggregation], feed_dict=feed_dict) #, model.concat   ,     aggregator_cls.vars

#     Z_tilde = np.repeat(Z, [len(G.nodes())], axis=0)
#     Z_tilde_tilde = np.tile(Z, (len(G.nodes()),1)) 
#     final_theta_1 = Quadratic_SDP_solver (Z_tilde, Z_tilde_tilde, FLAGS.dim_1*2, FLAGS.dim_2*2) 
# #     Z_centralized = Z - np.mean(Z, axis=0)
# #     final_adj_matrix = np.abs(np.matmul(Z_centralized, np.matmul(final_theta_1, np.transpose(Z_centralized))))
#     final_adj_matrix = np.matmul(Z, np.matmul(final_theta_1, np.transpose(Z)))
#     feed_dict = minibatch.next_minibatch_feed_dict()
#     feed_dict.update({placeholders['dropout']: FLAGS.dropout})
#     feed_dict.update({placeholders['all_nodes']: all_nodes})

#     FLAGS.batch_size = len(G.edges())
#     FLAGS.negbatch_size = len(Gneg.edges())
#     FLAGS.samples_1 = len(G.nodes())
#     FLAGS.samples_2 = len(G.nodes())
#     FLAGS.negsamples_1 = len(Gneg.nodes())
#     FLAGS.negsamples_2 = len(Gneg.nodes())
#     feed_dict = minibatch.batch_feed_dict(G.edges())
#     negfeed_dict = negminibatch.batch_feed_dict(Gneg.edges())
#     feed_dict.update({placeholders['negbatch1']: negfeed_dict[placeholders['batch1']]})
#     feed_dict.update({placeholders['negbatch2']: negfeed_dict[placeholders['batch2']]})
#     feed_dict.update({placeholders['negbatch_size']: negfeed_dict[placeholders['batch_size']]})
#     if(outs is not None):
#         print('loss: ', outs[2])
        
    final_adj_matrix, final_theta_1, Z, loss, U = print_summary(model, feed_dict, sess, Adj_mat, 'last', print_flag)
    
    
    #print('Z shape: ', Z.shape)
    
    if FLAGS.save_embeddings:
        sess.run(val_adj_info.op)

        save_val_embeddings(sess, model, minibatch, FLAGS.validate_batch_size, log_dir())

        if FLAGS.model == "n2v":
            # stopping the gradient for the already trained nodes
            train_ids = tf.constant([[id_map[n]] for n in G.nodes_iter() if not G.node[n]['val'] and not G.node[n]['test']],
                    dtype=tf.int32)
            test_ids = tf.constant([[id_map[n]] for n in G.nodes_iter() if G.node[n]['val'] or G.node[n]['test']], 
                    dtype=tf.int32)
            update_nodes = tf.nn.embedding_lookup(model.context_embeds, tf.squeeze(test_ids))
            no_update_nodes = tf.nn.embedding_lookup(model.context_embeds,tf.squeeze(train_ids))
            update_nodes = tf.scatter_nd(test_ids, update_nodes, tf.shape(model.context_embeds))
            no_update_nodes = tf.stop_gradient(tf.scatter_nd(train_ids, no_update_nodes, tf.shape(model.context_embeds)))
            model.context_embeds = update_nodes + no_update_nodes
            sess.run(model.context_embeds)

            # run random walks
            from graphsage.utils import run_random_walks
            nodes = [n for n in G.nodes_iter() if G.node[n]["val"] or G.node[n]["test"]]
            start_time = time.time()
            pairs = run_random_walks(G, nodes, num_walks=50)
            walk_time = time.time() - start_time

            test_minibatch = EdgeMinibatchIterator(G, 
                id_map,
                placeholders, batch_size=FLAGS.batch_size,
                max_degree=FLAGS.max_degree, 
                num_neg_samples=FLAGS.neg_sample_size,
                context_pairs = pairs,
                n2v_retrain=True,
                fixed_n2v=True)
            
            start_time = time.time()
            print("Doing test training for n2v.")
            test_steps = 0
            for epoch in range(FLAGS.n2v_test_epochs):
                test_minibatch.shuffle()
                while not test_minibatch.end():
                    feed_dict = test_minibatch.next_minibatch_feed_dict()
                    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
                    outs = sess.run([model.opt_op, model.loss, model.ranks, model.aff_all, 
                        model.mrr, model.outputs1], feed_dict=feed_dict)
                    if test_steps % FLAGS.print_every == 0:
                        print("Iter:", '%04d' % test_steps, 
                              "train_loss=", "{:.5f}".format(outs[1]),
                              "train_mrr=", "{:.5f}".format(outs[-2]))
                    test_steps += 1
            train_time = time.time() - start_time
            save_val_embeddings(sess, model, minibatch, FLAGS.validate_batch_size, log_dir(), mod="-test")
            print("Total time: ", train_time+walk_time)
            print("Walk time: ", walk_time)
            print("Train time: ", train_time)
    #del adj_info_ph, adj_info,placeholders         
    return final_adj_matrix, G, final_theta_1, Z, loss, U#, learned_vars



def post_processing_clip(adj_mat, G, save_filename):
    #print("final adjacency matrix ", adj_mat)
    #summation = np.max(np.sum(adj_mat, 1))
    th_1 = -100
    th_2 = 0
    scale = 1#/summation
    for u, v in G.edges():
        G[u][v]['weight'] = (adj_mat[u][v] * scale)
    
    N = len(G.nodes())
    DS_ratio = 5
    S = int(N/DS_ratio)
    
    plot_num_rows = int(np.floor(DS_ratio**0.5))
    plot_num_cols = int(np.ceil(DS_ratio/plot_num_rows)) 
    
    clip_num_rows = int(np.floor(S**0.5))
    clip_num_cols = int(np.ceil(S/clip_num_rows)) 
    #fig, ax = plt.subplots(num_rows, num_cols)
    #Grid=nx.grid_2d_graph(num_rows,num_cols)  
    pos = {}
    plt.figure()
    for i in range(DS_ratio):
#        ax = plt.subplot(num_rows,num_cols,i)
#        ax.plot()
        
        plt.subplot(plot_num_rows, plot_num_cols, i+1) #231+i
        H = G.subgraph(np.arange(S) + i*S)
        for j in np.arange(S):
            pos[j+ i*S] = (j%clip_num_cols,int(j/clip_num_cols))
        #print('nodes ', H.nodes())
        elarge = [(u, v) for (u, v, d) in H.edges(data=True) if (d['weight'] < th_1 )]#or d['weight'] < th_2
        esmall = [(u, v) for (u, v, d) in H.edges(data=True) if (d['weight'] <= th_2 and d['weight'] > th_1)]
        
#        Grid=nx.grid_2d_graph(5,6)                                                         
#        pos = dict(zip(Grid.nodes(),Grid.nodes())) 
#        pos = nx.spring_layout(H)  # positions for all nodes
         
        # nodes
        nx.draw_networkx_nodes(H, pos, node_size=8)
        # edges
        nx.draw_networkx_edges(H, pos, edgelist=elarge, width=1)
        #print('i = ', i)
        nx.draw_networkx_edges(H, pos, edgelist=esmall, width=1, alpha=0.5, edge_color='b', style='dashed')
        
        # labels
        #nx.draw_networkx_labels(Grid, pos, font_size=20, font_family='sans-serif')
    
        plt.axis('off')
    #print('output address: ', save_filename )
    plt.savefig(save_filename + '.png')
    #plt.show()



binarize_threshold_1 = None #-1 # 10000
binarize_threshold_2 = -100

def main(pure_filename, out_file_name, aggregator_model, D1, num_epochs, theta_exist, flag_normalized, \
                G_data=None, id_map=None, feats=None, Theta=None, fixed_neigh_weights=None, neg_sample_weights=1.0,\
                 brain_similarity_sizes=None, loss_function=None): #argv=None
    
    tf.reset_default_graph()
    FLAGS.model = aggregator_model
    FLAGS.dim_1 = D1
    FLAGS.dim_2 = D1
    FLAGS.epochs = num_epochs
    FLAGS.theta_exist = theta_exist
    FLAGS.flag_normalized = flag_normalized
    if(brain_similarity_sizes is not None):
        FLAGS.brain_similarity_sizes = brain_similarity_sizes
    if(loss_function is not None):
        FLAGS.loss_function = loss_function
        
    #print("Loading training data..")
    train_data = load_data(G_data=G_data, id_map=id_map, feats=feats, prefix=pure_filename) #changed, FLAGS.train_prefix -> and  load_walks=True -> False
    #print("Done loading training data..")    
    if(fixed_neigh_weights is not None):
#         print('fixed_neigh_weights shape ', fixed_neigh_weights[0].shape)
        Theta = tf.constant(Theta, dtype=tf.float32)
        fixed_neigh_weights = [tf.constant(a, dtype=tf.float32, name='neigh_weights') for a in fixed_neigh_weights]
    final_adj_matrix, G, final_theta_1, Z, loss, U = train(train_data, log_dir, Theta=Theta, fixed_neigh_weights=fixed_neigh_weights, neg_sample_weights=neg_sample_weights) #, learned_vars # solve(train_data, log_dir)
    #final_adj_matrix = np.abs(final_adj_matrix)
    #post_processing_clip(final_adj_matrix, G, out_file_name)
    return final_adj_matrix, final_theta_1, Z, loss, U #, learned_vars  
    
                
if __name__ == '__main__':
    tf.app.run()
