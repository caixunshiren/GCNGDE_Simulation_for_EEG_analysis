from __future__ import print_function


import random
import json
import sys
import os
from scipy.io import savemat, loadmat
import networkx as nx
from networkx.readwrite import json_graph
import tensorflow as tf
import numpy as np
import cvxopt
from cvxopt import solvers
from cvxopt import matrix
import struct
from scipy.optimize import linprog
# version_info = map(int, nx.__version__.split('.'))
# major = version_info[0]
# minor = version_info[1]
# assert (major <= 1) and (minor <= 11), "networkx major version > 1.11"
flags = tf.compat.v1.flags # tf.app.flags
FLAGS = flags.FLAGS
WALK_LEN=5
N_WALKS=50

def non_edges(edges, nodes):
    non_edges = []
    for u in nodes:
        for v in nodes:
            if((not ((u,v) in edges)) and (not ((v,u) in edges)) and u!=v):
                non_edges.append((u,v))
    return non_edges

def Graph_complement(G):
    not_edges = non_edges(G.edges(), G.nodes())
    Gc = G.copy() # nx.complement(G)
    Gc.remove_edges_from(G.edges())
    Gc.add_edges_from(not_edges)
    nx.set_edge_attributes(Gc, True, 'train_removed')
    nx.set_edge_attributes(Gc, False, 'test_removed')
    return Gc



def load_mat_data(data_dir, target,data_type):
    dir = os.path.join(data_dir, target)    
    filenames = sorted(os.listdir(dir))
    for i, filename in enumerate(filenames):
        comp_file_name = dir + '/' + filename
        if(filename.find('_' + data_type) != -1):
            data = loadmat( comp_file_name)
            yield(data)

    
def load_data(G_data=None, id_map=None, feats=None, normalize=False):
    
    if(G_data is None):
        G_data = json.load(open(prefix + "-G.json")) # G_data.get('nodes')[0:14755]
    G = json_graph.node_link_graph(G_data)
    if isinstance(G.nodes()[0], int):
        conversion = lambda n : int(n)
    else:
        conversion = lambda n : n

    if(feats is None):
        if os.path.exists(prefix + "-feats.npy"):
            feats = np.load(prefix + "-feats.npy") # numpy array of size: (14755, 50)
        else:
            feats = None
    if(id_map is None):
        id_map = json.load(open(prefix + "-id_map.json"))
    id_map = {conversion(k):int(v) for k,v in id_map.items()} #id_map.get('22') = 22
    walks = []
    broken_count = 0
    must_be_removed_nodes = []
    for node in G.nodes():
        if not 'val' in G.node[node] or not 'test' in G.node[node]:
            must_be_removed_nodes.append(node)
            broken_count += 1
    for node in must_be_removed_nodes:
        G.remove_node(node)
    for edge in G.edges():
        if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or
            G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False

    if normalize and not feats is None:
        from sklearn.preprocessing import StandardScaler
        train_ids = np.array([id_map[str(n)] for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']]) 
        train_feats = feats[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)
    
    return G, feats, id_map, walks
    

def run_random_walks(G, nodes, num_walks=N_WALKS):
    pairs = []
    for count, node in enumerate(nodes):
        if G.degree(node) == 0:
            continue
        for i in range(num_walks):
            curr_node = node
            for j in range(WALK_LEN):
                next_node = random.choice(G.neighbors(curr_node))
                # self co-occurrences are useless
                if curr_node != node:
                    pairs.append((node,curr_node))
                curr_node = next_node
        if count % 1000 == 0:
            print("Done walks for", count, "nodes")
    return pairs

def convert_dual_SDP(C,A,b):
    c = matrix([np.double(-b)]) #-b]).astype(np.double)
    A = A.astype(np.double)
    C = C.astype(np.double)
    h = [matrix(C)]
    G = [matrix(np.reshape(np.transpose(A),(A.size,1)))]
    return c,G,h
        
def normal_SDP_solver(Z1, Z2,input_dim1, input_dim2, Z3=None, Z4=None):
    C = np.matmul(np.transpose(Z2), Z1)
    
    if(FLAGS.model_size == 'small' and Z3 is not None and Z4 is not None):
        C -= np.matmul(np.transpose(Z4), Z3)
    elif(Z3 is not None):
        C -= np.matmul(np.transpose(Z3), Z1)
    
    C = (C + np.transpose(C))/2
    A = np.ones((input_dim2,input_dim1))
    b = 1  
    
    c,G,h = convert_dual_SDP(C,A,b)
    solvers.options['show_progress'] = False
    solution = solvers.sdp(c, Gs=G, hs=h)
    #print('solution zs ---- ', solution['zs'][0])	
    dual_theta = np.array(solution['zs'][0])
    #print('theta ---- ', dual_theta)
    dual_theta = np.reshape(dual_theta, (input_dim1, input_dim2))
    return dual_theta

def convert_quadratic_SDP(C,A,b):
    tilde_C = np.zeros(C.shape)
    tilde_C = np.concatenate((np.concatenate((tilde_C,tilde_C),axis=1),np.concatenate((C,tilde_C),axis=1)),axis=0) 
    return convert_dual_SDP(tilde_C,A,b)

        
def NP_Brain_Parameter_Solver(Z1, Z2, input_dim1, input_dim2, Z3=None, Z4=None):
    pre_index = 0
    c = []
    for ccounter in range(len(FLAGS.brain_similarity_sizes)):
        ssize = FLAGS.brain_similarity_sizes[ccounter]
        c.append(-tf.abs(tf.matmul( Z1[:,pre_index:pre_index+ssize], tf.conj(tf.transpose(Z2[:,pre_index:pre_index+ssize]))))+\
                 tf.abs(tf.matmul( Z3[:,pre_index:pre_index+ssize], tf.conj(tf.transpose(Z4[:,pre_index:pre_index+ssize])))))
        pre_index = ssize
    A_eq = np.ones((len(FLAGS.brain_similarity_sizes),1))
    b_eq = 1
    A_ub = np.concatenate((np.eye(len(FLAGS.brain_similarity_sizes)), -np.eye(len(FLAGS.brain_similarity_sizes))), axis=0)
    b_ub = np.concatenate((np.ones((len(FLAGS.brain_similarity_sizes),1)), np.zeros((len(FLAGS.brain_similarity_sizes),1))), axis=0)
    
    res = linprog(c, A_ub, b_ub, A_eq, b_eq)
    return res.x
           
def Brain_Parameter_Solver(Z1, Z2, input_dim1, input_dim2, Z3=None, Z4=None):
    if(Z4 is not None):
        solution = tf.py_func(NP_Brain_Parameter_Solver, [Z1, Z2, input_dim1, input_dim2, Z3, Z4], tf.float64) #normal_SDP_solver, Quadratic_SDP_solver
    else:
        solution = tf.py_func(NP_Brain_Parameter_Solver, [Z1, Z2, input_dim1, input_dim2, Z3], tf.float64) #normal_SDP_solver, Quadratic_SDP_solver
    return tf.cast(solution, tf.float32)
    
    
    
def Quadratic_SDP_solver(Z1, Z2, input_dim1, input_dim2, Z3=None, Z4=None):
    C = np.matmul(np.transpose(Z2), Z1)
    if(FLAGS.model_size == 'small' and Z3 is not None and Z4 is not None):
        C -= np.matmul(np.transpose(Z4), Z3)
    elif(Z3 is not None):
        C -= np.matmul(np.transpose(Z3), Z1)
#     elif(Z3 is None and Z4 is None):
#         C = C
    C = -(C + np.transpose(C))/2
    b = 1

    A = np.eye(np.shape(C)[0]*2)
    c,G,h = convert_quadratic_SDP(C,A,b)
    solvers.options['show_progress'] = False
    solution = solvers.sdp(c, Gs=G, hs=h)
    dual_Z = np.array(solution['zs'][0])
    dual_theta = dual_Z[0:input_dim1,input_dim1:input_dim1+input_dim2]
    dual_theta = np.reshape(dual_theta, (input_dim1, input_dim2))
    dual_theta /= np.sqrt(np.sum(dual_theta**2))
    return dual_theta
 
       
        
def SDP_solver( Z1, Z2, input_dim1, input_dim2, Z3=None, Z4=None): 
    
    '''
    min_(X) tr(CX)
    s.t.    tr(AX) = b
            X >= 0  
    '''
    if(Z4 is not None):
        solution = tf.py_func(Quadratic_SDP_solver, [Z1, Z2, input_dim1, input_dim2, Z3, Z4], tf.float64) #normal_SDP_solver, Quadratic_SDP_solver
    else:
        solution = tf.py_func(Quadratic_SDP_solver, [Z1, Z2, input_dim1, input_dim2, Z3], tf.float64) #normal_SDP_solver, Quadratic_SDP_solver
    return tf.cast(solution, tf.float32)

def tensor_repeat(a,b,axis):
    rep = tf.py_func(np.repeat, [a,b,axis], tf.float32)
    return tf.cast(rep, tf.float32)


if __name__ == "__main__":
    """ Run random walks """
    graph_file = sys.argv[1]
    out_file = sys.argv[2]
    G_data = json.load(open(graph_file))
    G = json_graph.node_link_graph(G_data)
    nodes = [n for n in G.nodes() if not G.node[n]["val"] and not G.node[n]["test"]]
    G = G.subgraph(nodes)
    pairs = run_random_walks(G, nodes)
    with open(out_file, "w") as fp:
        fp.write("\n".join([str(p[0]) + "\t" + str(p[1]) for p in pairs]))


#             c = matrix([1.,-1.,1.])
#             G = [ matrix([[-7., -11., -11., 3.],
#                           [ 7., -18., -18., 8.],
#                           [-2.,  -8.,  -8., 1.]]) ]
#             G += [ matrix([[-21., -11.,   0., -11.,  10.,   8.,   0.,   8., 5.],
#                            [  0.,  10.,  16.,  10., -10., -10.,  16., -10., 3.],
#                            [ -5.,   2., -17.,   2.,  -6.,   8., -17.,  8., 6.]]) ]
#             h = [ matrix([[33., -9.], [-9., 26.]]) ]
#             h += [ matrix([[14., 9., 40.], [9., 91., 10.], [40., 10., 15.]]) ]
#             sol = solvers.sdp(c, Gs=G, hs=h)
#             print("first solution", sol['x'])
