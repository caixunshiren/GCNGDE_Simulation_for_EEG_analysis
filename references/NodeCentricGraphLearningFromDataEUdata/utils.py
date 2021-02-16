from supervised_tasks import train_test
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from mpl_toolkits.axes_grid1 import AxesGrid
from graphsage.unsupervised_train import main
import json
import numpy as np
import os.path
import sys
import random
from sklearn.preprocessing import normalize
from PIL import Image
import networkx as nx
import math
import os
import scipy
from numpy import fft
from scipy.signal import hilbert
from scipy.fftpack import rfft, irfft, fftfreq
from scipy.signal import butter, lfilter
import scipy.io
import tensorflow as tf
from tensorflow.python.ops import metrics
from tensorflow.python.ops import resources
from collections import namedtuple
import h5py
# Ignore all GPUs, tf random forest does not benefit from it.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

title_fontSize = 40



        
def parameter_selection(N, D, rep_mode, graph_const_mode, population_coeff, theta_exist, mean_U_scale=0):
    D1 = int(D/2)
    concat_flag=2
    
    small_big_threshold = 100
    if(N<=small_big_threshold):
        model_size = 'small'
    else:
        model_size = 'big'
        
    if model_size == "small":
        hidden_dim = 512
    elif model_size == "big":
        hidden_dim = 1024
    
    if(theta_exist):
        Theta = np.random.multivariate_normal(np.ones(D,)*mean_U_scale, np.eye(D,D), size=(D,))
        Theta = (Theta+Theta.T)/2
        Theta = Theta/np.linalg.norm(Theta, ord='fro')
    else:
        Theta = np.eye(D)
    
    layer_infos = [D1,D1] # min(25, n_train_samples), min(10,n_train_samples)
    dims = [D1]
    num_layers = len(layer_infos)
    dims.extend([layer_infos[i] for i in range(num_layers)])
    U = []
    D1 = int(D/concat_flag)
    for layer in range(num_layers):
        dim_mult = 2 if ((concat_flag==2) and (layer != 0)) else 1
        
        if(rep_mode=='graphsage_mean'):
            dim1 = dim_mult*dims[layer]
            
        elif(rep_mode=='graphsage_maxpool'):
            dim1 = hidden_dim
            
        dim2 = dims[layer+1]
#         UL = np.random.random_sample(size=(dim1, dim2)).astype(np.float32)
        UL = np.random.multivariate_normal(np.ones(dim1,)*mean_U_scale, np.eye(dim1,dim1), size=(dim2,)).T
        U.append(UL)
#             U_tilde_GT = (np.random.random_sample(size=(D,D))) # dim_mult*dims[layer], dims[layer+1]
#     X_test = np.random.random_sample(size=(N,D1))
#     W, _, _, _, _ = hybrid_repres_estimation(X_test, A, input_prefix, output_prefix, 'test1', aggregator_model, \
#                                            num_epochs, iteration_flag, Theta=Theta, fixed_neigh_weights=U)
    A = Graph_Construction(N, graph_const_mode, population_coeff)
    return Theta, U, A , population_coeff#, W, X_test


def sampler(A, Theta, U, num_samples, input_prefix, output_prefix, aggregator_model, theta_exist, flag_normalized, num_epochs=10, x_samp_model=0, scale_x_sampler=3):
    n_train_samples,_ = np.shape(A)
    D,_ = np.shape(Theta)
    D1 = int(D/2)
    more_num_samples = scale_x_sampler*num_samples
    probs = [] #np.zeros((more_num_samples,1));
    samples = []
    Weights = []
    for i in range(more_num_samples):
        if(x_samp_model==0):
            X = np.random.random_sample(size=(n_train_samples,D1)) 
        elif(x_samp_model==1):
            X = np.random.multivariate_normal(np.zeros(n_train_samples,), 1-A + 0.1, size=(D1,)).T  # 
#         X = np.where(A==1, X+2, X)
        if(flag_normalized):
            X = X - np.mean(X)
            X = normalize(X, axis=1)
        W, Theta_no, Z, A_hybrid, loss, U_no = hybrid_repres_estimation(X, A, input_prefix, output_prefix, 'test1', aggregator_model, \
                                           num_epochs, 0, theta_exist, flag_normalized, Theta=Theta, fixed_neigh_weights=U)
        samples.append(X)
        Weights.append(W)
        prob = -loss # calc_prob(samples, A, Theta, U)
        probs.append(prob)
    probs = np.array(probs)
    which = (probs.argsort()[-num_samples:][::-1]).astype(np.int)
    return [samples[i] for i in which], probs[which], [Weights[i] for i in which]



def W_map_A(W, th):
    A = np.where(W>th, 1, 0)
#     A = A - np.diag(np.diag(A))
    return A
    
    
def hybrid_repres_estimation(X, A, input_prefix, output_prefix, filename, aggregator_model, \
                             num_epochs, iteration_flag, theta_exist, flag_normalized, Theta=None,\
                              fixed_neigh_weights=None, population_coeff=None, neg_sample_weights=1.0,\
                               th=None, brain_similarity_sizes=None, loss_function=None):
#     print(np.get_printoptions())
#     np.set_printoptions(threshold='nan') #1000
    print('X shape ', X.shape)
    counter = 0
    indices = []
    if not os.path.exists(input_prefix):
        os.makedirs(input_prefix)
    if(iteration_flag == 0):
        G_data, id_map, feats = json_gen(X, A, input_prefix + filename, False)
        if not os.path.exists(output_prefix):
            os.makedirs(output_prefix)
        W, Theta_out, Z, loss, U = main(input_prefix + filename, output_prefix + filename, aggregator_model,\
                                            X.shape[1], num_epochs, theta_exist, flag_normalized, G_data=G_data,\
                                             id_map=id_map, feats=feats, Theta=Theta, fixed_neigh_weights=fixed_neigh_weights,\
                                              neg_sample_weights=neg_sample_weights, brain_similarity_sizes=brain_similarity_sizes,\
                                              loss_function = loss_function)
    elif(iteration_flag == 1):
        num_to_remove = 5
        A_pre = np.zeros(A.shape)
        while((np.sum(np.abs(A_pre-A))>(np.size(A)/8))): #  counter <= num_to_remove and not (A==0).all()
            G_data, id_map, feats = json_gen(X, A, input_prefix + filename, False)
            if not os.path.exists(output_prefix):
                os.makedirs(output_prefix)
            # W, Theta_out, Z, loss = main(input_prefix + filename, output_prefix + filename, aggregator_model, X.shape[1], num_epochs, Theta=Theta, fixed_neigh_weights=fixed_neigh_weights)
            W, Theta_out, Z, loss, U = main(input_prefix + filename, output_prefix + filename, aggregator_model,\
                                                X.shape[1], num_epochs, theta_exist, flag_normalized, G_data=G_data, id_map=id_map, feats=feats,\
                                                Theta=Theta, fixed_neigh_weights=fixed_neigh_weights, neg_sample_weights=neg_sample_weights) #, vars
            A_pre = np.copy(A)
            if(True): # counter == 0 and th is None
                th = np.mean((W)) #(np.max(W) + np.min(W))/2 # (np.max(softmax(W)) + np.min(softmax(W)))/2 # np.mean(softmax(W))
            A = W_map_A(W, th) #convert(W, mode='normal')
            A[[(i) for i in range(A.shape[0])],[(i) for i in range(A.shape[0])]]= 1
            print('diff A, A_pre: ', np.sum(np.abs(A_pre-A)))
            print('A population ', np.sum(np.abs(A))/np.size(A))
            counter += 1


    elif(iteration_flag == 2): 
        N = A.shape[0]
        A[[(i) for i in range(N)],[(i) for i in range(N)]]= 1
        W = A 
        if(population_coeff!=None):
            num_to_remove = int(np.floor(N*(1-population_coeff)))
        else:
            num_to_remove = int(np.floor(N/2))
        A_init = A
        while(counter<= num_to_remove): # ((A_pre-A).any() and not A.all())
            G_data, id_map, feats = json_gen(X, A, input_prefix + filename, False)
            if not os.path.exists(output_prefix):
                os.makedirs(output_prefix)
            W_pre = W
            W, Theta_out, Z, loss, U = main(input_prefix + filename, output_prefix + filename, aggregator_model,\
                                                X.shape[1], num_epochs, theta_exist, flag_normalized, G_data=G_data, id_map=id_map, feats=feats,\
                                                Theta=Theta, fixed_neigh_weights=fixed_neigh_weights, neg_sample_weights=neg_sample_weights) #, vars
            W += (W.max() - W.min())* 2 * np.eye(W.shape[0]) 
            indice1, indice2 = np.unravel_index(np.argpartition(np.reshape(W,(W.size,)),counter*2+1)[0:counter*2+2], W.shape) # np.unravel_index(np.argmin(W), W.shape)
            A = A_init
            A[indice1, indice2] = 0   
#             print('pre : ', [W_pre[indice_pair] for indice_pair in indices])
#             print('next: ', [    W[indice_pair] for indice_pair in indices])
            print(np.argsort(np.reshape(W,(W.size,)))[:40])
            counter += 1
            
            
    elif(iteration_flag == 3): 
        N = A.shape[0]
        A[[(i) for i in range(N)],[(i) for i in range(N)]]= 1
        W = A 
        if(population_coeff!=None):
            num_to_remove = int(np.floor(N*(1-population_coeff)))
        else:
            num_to_remove = int(np.floor(N/2))
        A_init = A
        edge_stack = []
        while(counter<= num_to_remove): # ((A_pre-A).any() and not A.all())
            G_data, id_map, feats = json_gen(X, A, input_prefix + filename, False)
            if not os.path.exists(output_prefix):
                os.makedirs(output_prefix)
            W_pre = W
# W, Theta_out, Z, loss = main(input_prefix + filename, output_prefix + filename, aggregator_model, X.shape[1], num_epochs, Theta=Theta, fixed_neigh_weights=fixed_neigh_weights)
            W, Theta_out, Z, loss, U = main(input_prefix + filename, output_prefix + filename, aggregator_model,\
                                                X.shape[1], num_epochs, theta_exist, flag_normalized, G_data=G_data, id_map=id_map, feats=feats,\
                                                Theta=Theta, fixed_neigh_weights=fixed_neigh_weights, neg_sample_weights=neg_sample_weights) #, vars
            W += (W.max() - W.min())*2*np.eye(W.shape[0]) 
            indices1, indices2 = np.unravel_index(np.argpartition(np.reshape(W,(W.size,)),len(edge_stack)+1)[0:len(edge_stack)+2], W.shape) # np.unravel_index(np.argmin(W), W.shape)
            edge_stack_new = []
            for i in range(len(indices1)):
                edge_stack_new.append((indices1[i], indices2[i]))
            edge_stack_new = list(set(edge_stack_new))
            A = A_init
            if(set(edge_stack) <= set(edge_stack_new)):
                A[indices1, indices2] = 0
                edge_stack = set(edge_stack_new)
                counter += 1
            else:
                edge_stack = []
                counter = 0
            print(np.argsort(np.reshape(W,(W.size,)))[:40])
    return W, Theta_out, Z, A, loss, U  # np.mean(np.abs(Z[:,100:200])) # np.mean(np.abs(Z[:,0:100]))



def json_gen(data, A, input_prefix, save_flag=True):
    N,D = data.shape
    test_flag = False
    G_jason_data = {}  
    id_map_jason_data = {}
    G_jason_data['directed'] = False
    G_jason_data['graph'] = {
        "name": "disjoint_union( ,  )"
        }
    G_jason_data['nodes'] = []
    for n in range(N):
        feature_vector = data[n,:]
        feature_vector = np.reshape(feature_vector, (feature_vector.size,) )
        id_map_jason_data[str(n)] = n
        label_vector = []
        if(n > N):
            test_flag = True
        G_jason_data['nodes'].append({  
            'test': test_flag,
            'id': n , 
            'feature': feature_vector.tolist(),
            'val': test_flag,
            'label': label_vector,
        })
    
    G_jason_data['links'] = []
    test_flag = False
    for i in range(N):
        if(i > N):
            test_flag = True
        for j in i + np.arange(N-i):
            if(A[i,j]!=0):
                G_jason_data['links'].append({  
                    'test_removed': test_flag,
                    'train_removed': not test_flag,
                    'target': int(i),
                    'source': int(j),
                    
                })
    G_jason_data['multigraph'] = False 
#     if not os.path.exists(input_prefix):
#         os.makedirs(input_prefix)
    if(save_flag):
        with open( input_prefix +'-G.json', 'w') as outfile:  
            json.dump(G_jason_data, outfile)
        
        with open( input_prefix +'-id_map.json', 'w') as outfile:  
            json.dump(id_map_jason_data, outfile)
        
        np.save(input_prefix +'-feats', data)
    #print(input_prefix +'-feats saved!')
    return G_jason_data, id_map_jason_data, data

def Graph_Construction(N, mode, population_coeff):
    if(mode == 'full'):
        A = np.ones((N,N)) # - np.eye(num_vertices)
    elif(mode == 'random'):
        A = 10* np.ones((N,N))
        A[np.triu_indices(N)] = np.random.rand(np.triu_indices(N)[0].size,)
        A = np.where(A<=population_coeff, 1, 0)
        A = (A + A.T)
#         A[np.tril_indices(N)] = A[np.triu_indices(N)].T
        A[[(i) for i in range(A.shape[0])],[(i) for i in range(A.shape[0])]]= 1
    return A

def softmax(a):
    return np.exp(a) / np.sum(np.exp(a))#, axis=axis

def normal_scale(a, axis=None):
    if(axis is None):
        meann = np.mean(a, axis=axis)
    else:
        meann = np.mean(a, axis=axis)[:,np.newaxis]
    a = a - meann
    a /= (np.var(a)**0.5)
    return a


def convert(a, mode=None, A=None, par=None): 
    max_val = np.max(a)
    min_val = np.min(a)
    if(mode == 'l1_norm_shift'):
        a = a - np.min(a)
        result = a/np.sum(a)
        
    elif(mode == 'normal') :    
        result = normal_scale(a)
    
    elif(mode == 'RGB') :    
        a = a - np.min(a)
        a /= np.max(a)
        a *= 255
        result = a
        
    elif(mode == 'shift_pos') :    
        result = a - np.min(a)
        
    elif(mode == 'softmax') :    
        result = softmax(a)
        
    elif(mode == 'noChange') :    
        result = (a)
        
    elif(mode == 'abs' or (mode == None and max_val<=1 and min_val>=-1)):
        result = np.abs(a)
        
    elif(mode == 'l1_norm' or (mode == None and max_val>1e2)):
        a = np.abs(a)
        result = a/np.sum(a)
        
    elif(mode == 'froNorm'):
        result = np.abs(a)/np.linalg.norm(a, ord='fro')
    
    elif(mode == 'lin_scale'):
        result = a*par
    
#     if(A is not None):    
#         result = np.where(A==0, 0, result)
    return result

def comp_weights(A,B): 
#     if(np.var(A)!=0 and np.var(B)!=0): 
    A = convert(A, mode='normal')
    B = convert(B, mode='normal') 
    comp = np.linalg.norm(A - B ,ord='fro')/np.linalg.norm(B ,ord='fro')
    return comp   

def eval_performance(B_all, B_star_all, title):
    comparison_error = np.zeros((len(B_all),))
    for counter in range(len(B_all)):
        b = B_star_all[counter] # convert(, mode='noChange')
        a = B_all[counter]
#         a = convert(a, mode='lin_scale', par=np.max(b)/np.max(a))
        comparison_error[counter] =  comp_weights(a,b)
    #print('comparison error of ' + title + ': ', comparison_error)
    return np.mean(comparison_error), np.var(comparison_error)

def add_subplot_border(ax, width=1, color=None ):

    fig = ax.get_figure()

    # Convert bottom-left and top-right to display coordinates
    x0, y0 = ax.transAxes.transform((0, 0))
    x1, y1 = ax.transAxes.transform((1, 1))

    # Convert back to Axes coordinates
    x0, y0 = ax.transAxes.inverted().transform((x0, y0))
    x1, y1 = ax.transAxes.inverted().transform((x1, y1))

    rect = plt.Rectangle(
        (x0, y0), x1-x0, y1-y0,
        color=color,
        transform=ax.transAxes,
        zorder=-1,
        lw=2*width+1,
        fill=None,
    )
    fig.patches.append(rect)
    
    
def plot_graph_windows(i, adj_mat, th_1, th_2, plot_num_rows, plot_num_cols, intervals_seizure):
    pos = {}
    S = adj_mat.shape[0]
    clip_num_rows = int(np.floor(S**0.5))
    clip_num_cols = int(np.ceil(S/clip_num_rows))
    rows, cols = np.where(adj_mat >= -1e20)
    edges = zip(rows.tolist(), cols.tolist())
    H = nx.Graph()
    H.add_edges_from(edges)
    for u, v in H.edges():
        H[u][v]['weight'] = (adj_mat[u][v])
    N = len(H.nodes())
    ax = plt.subplot(plot_num_rows, plot_num_cols, i+1) #231+i
    if(intervals_seizure != ''):
        ax.set_title(intervals_seizure)
        add_subplot_border(ax, color='r')
    for j in np.arange(S):
        pos[j+ i*S] = (j%clip_num_cols,int(j/clip_num_cols))
    return pos


def node_size_scale(a):
    a = a - np.min(a)    
    return a*300/np.max(a)


def plot_nodes(eigenVectors, plot_num_rows, plot_num_cols, intervals_seizures, save_filename, estimated_states):
    num_wind = len(eigenVectors) 
    pos = {}
    S = eigenVectors[0].size
    clip_num_rows = int(np.floor(S**0.5))
    clip_num_cols = int(np.ceil(S/clip_num_rows))
    
    plt.figure(num=None, figsize=(60, 30), dpi=80)
    for i in np.arange(num_wind):
#         plot_graph_windows(i, adj_mat[i], th_1, th_2, plot_num_rows, plot_num_cols, intervals_seizures[i]) 
        eigVec_i = eigenVectors[i]
        H = nx.Graph()
        H.add_nodes_from(np.arange(S))
        ax = plt.subplot(plot_num_rows, plot_num_cols, i+1) #231+i
#         if(estimated_states!= None):
        ax.set_title(str(estimated_states[i]), fontsize=title_fontSize)
#         if(intervals_seizures!= None and intervals_seizures[i] == 1):
        if(intervals_seizures[i] == 1):
            ax.set_title('--'+ str(estimated_states[i])+'--', fontsize=title_fontSize)
            add_subplot_border(ax, color='r')
        else:
            add_subplot_border(ax, 0.1, color='k')
        for j in np.arange(S):
            pos[j + i*S] = (j % clip_num_cols, int(j/clip_num_cols))
        nx.draw_networkx(H, pos, node_size = node_size_scale(eigVec_i), with_labels=False) # nodelist = nodelist,       
        plt.axis('off')      
        
    plt.savefig(save_filename + '.png')    
    
    

def plot_edges(W, plot_num_rows, plot_num_cols, intervals_seizures, savefolder, filename, estimated_states):
    S = np.int(np.sqrt(W[0].shape[0]))
    ictal_indices = np.argwhere(intervals_seizures==1).astype(np.int)
    ictal_indices = ictal_indices.reshape((ictal_indices.size,))
    
#     num_wind = len(W)
#     pos = {}
#     th_2 = 0 # np.mean(means) #1/(S*S*2) #0.2 #
#     th_1 = 0.5 #(th_2+np.max(means))/2 #1/(S*S) # 0.5 # 
#     clip_num_rows = int(np.floor(S**0.5))
#     clip_num_cols = int(np.ceil(S/clip_num_rows))         
#     plt.figure(num=None, figsize=(60, 40), dpi=120)
#     for i in np.arange(num_wind):
# #         plot_graph_windows(i, W[i], th_1, th_2, plot_num_rows, plot_num_cols, intervals_seizures[i]) 
#         adj_mat_i = W[i]
#         adj_mat_i = convert(adj_mat_i, mode='normal')
#         rows, cols = np.where(adj_mat_i >= -1e20)
#         edges = zip(rows.tolist(), cols.tolist())
#         H = nx.Graph()
#         H.add_edges_from(edges)
#         for u, v in H.edges():
#             H[u][v]['weight'] = (adj_mat_i[u][v])
#         N = len(H.nodes())
#         ax = plt.subplot(plot_num_rows, plot_num_cols, i+1) #231+i
#         
#         if(estimated_states is not None):
#             ax.set_title(str(estimated_states[i]), fontsize=title_fontSize)
#             
#         if(intervals_seizures is not None and intervals_seizures[i] == 1):
#             ax.set_title('--'+ str(estimated_states[i])+'--', fontsize=title_fontSize)
#             add_subplot_border(ax, color='r')
#             
#         for j in np.arange(S):
#             pos[j+ i*S] = (j%clip_num_cols,int(j/clip_num_cols))
#             
#         elarge = [(u, v) for (u, v, d) in H.edges(data=True) if (d['weight'] > th_1 )]#or d['weight'] < th_2
#         esmall = [(u, v) for (u, v, d) in H.edges(data=True) if (d['weight'] >= th_2 and d['weight'] < th_1)]        
#         nx.draw_networkx_nodes(H, pos, node_size=8)
#         nx.draw_networkx_edges(H, pos, edgelist=elarge, width=1)
#         nx.draw_networkx_edges(H, pos, edgelist=esmall, width=1, alpha=0.5, edge_color='b', style='dashed')       
#         plt.axis('off')
#     plt.savefig(save_filename + '_adjacency.png')
    
    
    fig = plt.figure(num=None, figsize=(60, 40), dpi=120)
    grid = AxesGrid(fig, 111,
                nrows_ncols=(plot_num_rows, plot_num_cols),
                axes_pad=1,
                cbar_mode='single',
                cbar_location='right',
                cbar_pad=0.1
                )
    i = 0
    for ax in grid: # i in np.arange(num_wind):
#         plt.subplot(plot_num_rows, plot_num_cols, i+1)
        if(i+1>len(W)):
            break
#         print('i: ', i)
#         print('len W: ', len(W))
        im = ax.imshow(np.reshape(W[i],(S,S))) # convert(adj_mat[i],mode='RGB')
        if(intervals_seizures is not None and intervals_seizures[i] == 1 and estimated_states is not None):
            ax.set_title('----'+ str(estimated_states[i])+'----', fontsize=title_fontSize)
#             add_subplot_border(ax, color='r')
        elif(intervals_seizures is not None and intervals_seizures[i] == 0 and estimated_states is not None):
            ax.set_title(str(estimated_states[i]), fontsize=title_fontSize)
        i += 1
#     fig.colorbar(im)
    cbar = ax.cax.colorbar(im)
    cbar.ax.tick_params(labelsize=30) 
    cbar = grid.cbar_axes[0].colorbar(im)
    
    plt.savefig(savefolder + '/W_' + filename + '.png')
    
#     EVC = np.zeros((S,num_wind))
#     for i in np.arange(num_wind):
#         EVC[:,i] = EigenVec_Centrality(adj_mat[i])
#     fig = plt.figure(num=None, figsize=(60, 40), dpi=120)
#     plt.imshow(EVC) # convert(EVC,mode='RGB')
#     plt.colorbar()
#     plt.savefig(savefolder + '/EVC_' + filename + '.png')
    
#     plt.show()
#     plt.figure(num=None, figsize=(60, 30), dpi=80)
#     plt.axis('off')
#     plt.savefig(save_filename + '_EVC.png')
    
#     plt.figure(num=None, figsize=(60, 30), dpi=80)
#     densities = np.array([network_density(a, th_1) for a in adj_mat])
#     plt.plot(np.arange(num_wind),densities ,'b', lw=4)
#     plt.plot(ictal_indices, densities[ictal_indices] , color='r', lw=7)
#     plt.savefig(save_filename + '_density.png')
    
#     plt.figure(num=None, figsize=(60, 30), dpi=80)
#     densities_binary = np.array([network_density(np.where(a>th_1, 1, 0)) for a in adj_mat])
#     plt.plot(np.arange(num_wind),densities_binary,'b', lw=4)
#     plt.plot(ictal_indices, densities_binary[ictal_indices], color='r', lw=7)
#     plt.savefig(save_filename + '_density_binary.png')
    
def plotting_figure(arr, title, x_array=None, show_flag=False):
    if(x_array is None):
        x_array = np.arange(arr.size)
    plt.figure(num=None, figsize=(60, 40), dpi=120)
    plt.plot(x_array, arr)
    plt.savefig(title + '.png')
    if(show_flag):
        plt.show()
    
def plotting_weights(save_folder, filename, W, intervals_seizures = None, estimated_states = None):
    num_wind = len(W)
    plot_num_rows = int(np.ceil(num_wind**0.5))
    plot_num_cols = int(np.ceil(num_wind/plot_num_rows)) 
    if not os.path.exists(save_folder + '/' ):
        os.makedirs(save_folder + '/' )
    plot_edges(W, plot_num_rows, plot_num_cols, intervals_seizures, save_folder , filename, estimated_states )
#     eigenVectors = [np.real(np.linalg.eig(a)[1][:,0]) for a in adj_mat]
#     plot_nodes(eigenVectors, plot_num_rows, plot_num_cols, intervals_seizures,  save_folder + '/' + 'Eigen_' + filename, estimated_states )

def EigenVec_Centrality(a):
    a = np.nan_to_num(a)
#     a[a==None] = 1e10
#     a[a==np.nan] = 1e10
#     a[a== np.inf] = 1e10
#     a[a== -np.inf] = -1e10
    landa, V = np.linalg.eig(a)
    return V[:,-1]

plotting_colors = ['r>-', 'ks-', 'b>-', 'gd:','c>-','md-']

def plot_performance(xrange, measures, xlabel, title, save_filename, num_approaches, name_weight_matrices):
    np.savetxt(save_filename, measures)
    np.savetxt(save_filename + 'xrange', xrange)
    plt.figure()
    for appr in range(num_approaches):
        plt.plot(xrange, measures[appr], plotting_colors[appr], linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel('Relative Error')
    plt.legend(name_weight_matrices)
    plt.title(title)
    plt.savefig(save_filename + '.png')
#     plt.show()


def coherence_func(data, freqBand, wilch_window_len, fs, D_overlapping=0.5):
    S = np.shape(data)[0]
    coherence_mat = -np.ones((S,S))
    for i in range(S):
        for j in range(S):
            freqCoh, Cxy = scipy.signal.coherence(data[i,:], data[j,:], fs=fs, nperseg=wilch_window_len, noverlap=int(D_overlapping*wilch_window_len))
            Cxy = np.where(freqCoh>=freqBand[0],Cxy,0)
            Cxy = np.where(freqCoh<freqBand[1],Cxy,0)
            coherence_mat[i,j] = np.sum(Cxy)
    return coherence_mat

def plot_eeg(data, seizure_start_time_offsets, seizure_lengths):
    plt.figure()
    xrange = np.arange(data.shape[1])
    for s in range(data.shape[0]):
        plt.plot(xrange, data[s,:]) 

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


class Corr_PLV_Coh(): 
    def __init__(self, feature_mode, freq_bands, initial_freq_band):
        self.feature_mode = feature_mode
        self.freq_bands = freq_bands
        self.initial_freq_band = initial_freq_band
        
    def apply(self, data, fs):
        # correlation
        data_ext = down_sampling(data, 10) # data
        sizes=[data.shape[1]]
           
#         #PLV
#         hilberted = hilbert(data)
#         hilberted /= np.abs(hilberted)
#         data_ext = np.concatenate((data_ext, hilberted), axis=1)
#         sizes.append(hilberted.shape[1])
#         #coherence
#         f_signal = rfft(data)   
#         W = fftfreq(f_signal.shape[-1], d=1/fs)
#         W = np.tile(W,[f_signal.shape[0],1])
#         for i in np.arange(len(self.freq_bands)):
#             if(i>0):
#                 lowcut = self.freq_bands[i-1]
#             else:
#                 lowcut = self.initial_freq_band
#             highcut = self.freq_bands[i]
#     #         butter_bandpass_filter(data, lowcut, highcut, fs)
#             cut_f_signal = f_signal.copy()
#             cut_signal = np.where(W<highcut, cut_f_signal,0 ) 
#             cut_signal = np.where(W>=lowcut, cut_signal,0 )            
#     #         cut_signal = irfft(cut_signal)
#     #         cut_signal = hilbert(cut_signal)
#     #         fft_cut_signal = np.abs(cut_signal)       
# #             data_ext = np.concatenate((data_ext, fft_cut_signal), axis=1)
#             data_ext = np.concatenate((data_ext, np.angle(cut_signal)), axis=1)
#         sizes.append(np.sum(np.array(sizes)))
        return data_ext, sizes
    
    
def down_sampling(X, ratio):
    S,T = X.shape
    output = np.zeros((S,ratio))
    window_len = int(T/ratio)
    for i in range(ratio):
        output[:,i] = np.mean(X[:,i * window_len:(i+1) * window_len], axis=1)    
    return output    
    
def get_inv_dir (sub_id):
    inv_subs=[1084, 1096, 1146, 253, 264, 273, 375, 384, 548, 565, 583, 590, 620, 862, 916, 922, 958, 970]
    inv_subs2=[1073, 1077, 1125, 115, 1150, 139, 442, 635, 818]
    inv_subs3=[13089, 13245, 732]
    if (sub_id in inv_subs):
        inv_dir='inv'
    elif (sub_id in inv_subs2):
        inv_dir='inv2'
    elif (sub_id in inv_subs3):
        inv_dir='inv3'
    return inv_dir
  

def load_EU_features(input_prefix, target, load_Core):
    """
        main function that loads data features from a saved .mat file, 
        whose content will be decoded by Matlab_Load_Core class in supervised_tasks.py
    """
    dir = input_prefix + '/' + get_inv_dir (target) +'/' + 'pat_FR_' + str(target) + '.mat'
    try:
        return scipy.io.loadmat(dir)
    except Exception as e:
        return h5py.File(dir, 'r')
#         with h5py.File(dir, 'r') as f:
#             a_group_key = list(f.keys())
#             # Get the data
#             data = list(f[a_group_key])
#             return data

     

def load_EU_settings(input_prefix, target, load_Core):
    dir = input_prefix + '/' + get_inv_dir (target) +'/' + 'pat_FR_' + str(target)+ '_setting' + '.mat'
    if os.path.exists(dir):
        matFile = scipy.io.loadmat(dir)
        return matFile['train_files'][0].size, matFile['test_files'][0].size
    else:
        return None

Structural_Side_Info = namedtuple('Structural_Side_Info', ['adj_means', 'adj_vars'])

def load_side_adj(task_core):
    input_prefix = task_core.sidinfo_dir
    target = task_core.target
    adj_calc_modes = task_core.adj_calc_mode
    
    adj_means = []
    adj_vars = []
#     adj_threshold = 0
    for adj_calc_mode in adj_calc_modes:
        dir = input_prefix + '/' + get_inv_dir (target) +'/' + 'pat_FR_' + str(target) + '_sideAdj' +'_' + adj_calc_mode + '.mat'
        side_inf = scipy.io.loadmat(dir)
#         if(not np.any(side_inf['adj_means']<0)):
#             adj_threshold = np.mean(side_inf['adj_means'])
        A = side_inf['adj_means']
        A_sorted = np.sort(np.reshape(A, -1))
        adj_threshold = A_sorted[int(A_sorted.size*task_core.A_density_ratio)]
        print('adj_threshold = ', adj_threshold)
        A = np.where(A >= adj_threshold, 1, 0) 
        print('A thresholded : ', A)
        adj_means.append(A)
        adj_vars.append(float(side_inf['adj_vars']))
    out = Structural_Side_Info(adj_means = adj_means, adj_vars = adj_vars)
    return out


def inpython_online_wrapper(matlab_engin, matlab_load_core, file_num_array, mode, data_dir, load_Core=None):
    X, Y, conv_sizes, sel_win_nums, clip_sizes = matlab_engin.python_online_wrapper(matlab_load_core.target, file_num_array, mode, data_dir, nargout=5)
    X = np.asarray(X)
    if(X.ndim == 3):
        X = X[..., np.newaxis]
    Y = class_relabel(np.reshape(np.asarray(Y),[-1]), clip_sizes, matlab_load_core, preszr_sec=load_Core.preszr_sec if load_Core is not None else 10, \
                                    postszr_sec=load_Core.postszr_sec if load_Core is not None else 100000)
    sel_win_nums = np.reshape(np.asarray(sel_win_nums),[-1])
    clip_sizes = np.asarray(clip_sizes)
    return X, Y, sel_win_nums, conv_sizes, clip_sizes
    
def fileNum2winNum(clip_sizes_total, idx):
    win_idx = None
#     if(clip_sizes_total.shape[0]!=idx.size):
#         raise NotImplementedError('fileNum2winNum not equal sizes')
    for counter in idx:
        if(np.size(clip_sizes_total,0)==2):
            start_idx = int(clip_sizes_total[0, counter])
            end_idx = int(clip_sizes_total[1, counter])
        else:
            start_idx = clip_sizes_total[counter, 0]
            end_idx = clip_sizes_total[counter, 1]
        in_arr = start_idx + np.arange(end_idx-start_idx)
        win_idx = in_arr if win_idx is None else np.hstack((win_idx, in_arr))
    return win_idx
                
def rolling_window_list (a, window, step_size):
    return [a[..., i: i + window] for i in range(0, (np.shape(a)[-1]+1)-window, step_size)]


def rolling_window(a, window, step_size):
    return np.array(rolling_window_list (a, window, step_size))

    
def EU_online_fileNum_RollingWindow(matlab_load_core, task_core):
    file_numbers = matlab_load_core.settings_TrainNumFiles + matlab_load_core.settings_TestNumFiles
    if(matlab_load_core.y_test.size != matlab_load_core.settings_TestNumFiles):
#         raise NotImplementedError('Test files were not loaded.')
        file_numbers = matlab_load_core.settings_TrainNumFiles
    file_num_arr = np.arange(file_numbers)
    in_train_idx = []
    in_test_idx = []
    if('24h-1h' in task_core.TrainTest_mode):
        in_train_idx = rolling_window_list(file_num_arr, 24, 1)[:-1]
        in_test_idx = [[int(a[-1]+1)] for a in in_train_idx]
    
    return zip(in_train_idx, in_test_idx)



def upper_triangle(matt):
#         return matt[np.triu_indices(self.graphL_core.num_nodes)]
    ones = tf.ones_like(matt)
    mask_a = tf.matrix_band_part(ones, 0, -1) # Upper triangular matrix of 0s and 1s
#         mask_b = tf.matrix_band_part(ones, 0, 0)  # Diagonal matrix of 0s and 1s
    mask = tf.cast(mask_a, dtype=tf.bool) # - mask_b Make a bool mask
    return tf.boolean_mask(matt, mask)
    
def class_relabel(y, clip_sizes, matlab_load_core, preszr_sec, postszr_sec):    
    y = np.where(y<0, 0, y) 
#     print('clip_sizes: ', clip_sizes)
#     print('clip_sizes shape: ', clip_sizes.shape)
#     try:
#         ynew = None
#         for counter in np.arange(clip_sizes.shape[1]):
#             start_idx = int(clip_sizes[0, counter])
#             end_idx = int(clip_sizes[1, counter])
#             inn_y = y[start_idx:end_idx]
#             inn_y_new = np.zeros_like(inn_y)
#             
#             if(np.any(inn_y!=0)):
#                 preszr_onset = int(np.argwhere(inn_y==1)[0])
#                 szr_offset = int(np.argwhere(inn_y==1)[-1])
#                 if(preszr_onset==0):
#                     print('!!!!!!!!!!!Preszr labels started from the beginning of the file.')
#                 szr_onset = preszr_onset + matlab_load_core.pre_ictal_num_win
#                 inn_y_new[np.max((szr_onset - sec2win(preszr_sec, matlab_load_core), 0)):np.min((szr_onset + sec2win(postszr_sec, matlab_load_core), szr_offset))] = 1
#             ynew = inn_y_new if ynew is None else np.concatenate((ynew, inn_y_new), axis=0)
#         ynew = np.array(ynew)
#     except:
#         ynew = y
#     return ynew
    return y
                  
def neural_net(x, classifier_properties, num_classes):
    final_layer = tf.layers.dense(x, classifier_properties.n_hidden_1)
    if(classifier_properties.n_hidden_2 is not None):
        layer_1 = final_layer
        final_layer = tf.layers.dense(layer_1, classifier_properties.n_hidden_2)
        if(classifier_properties.n_hidden_3 is not None):
            layer_2 = final_layer
            final_layer = tf.layers.dense(layer_2, classifier_properties.n_hidden_3)
            if(classifier_properties.n_hidden_4 is not None):
                layer_3 = final_layer
                final_layer = tf.layers.dense(layer_3, classifier_properties.n_hidden_4)
                
    out_layer = tf.layers.dense(final_layer, num_classes)
    return out_layer
#     model = keras.Sequential([
#         keras.layers.Flatten(input_shape=(28, 28)),
#         keras.layers.Dense(128, activation=tf.nn.relu),
#         keras.layers.Dense(10, activation=tf.nn.softmax)
#     ])
#     return model
    
    
def NN_classifier(features, labels, classif_core):
    logits = neural_net(features, classif_core.classifier_properties, classif_core.num_classes)
    pred_classes = tf.argmax(logits, axis=1)
    pred_probas = tf.nn.softmax(logits)
    if(classif_core.classifier_properties.loss_type == 'softmax'):
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                    logits=logits, labels=labels)) # tf.cast(, dtype=tf.int32)
    return loss, pred_classes, pred_probas, None, None


def RandomForest_classifier(features, labels, classif_core):
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        
        forest_graph = classif_core.classifier_properties.forest_graph
        # Get training graph and loss
        train_op = forest_graph.training_graph(features, labels)
        loss_op = forest_graph.training_loss(features, labels)
        logits, _, _ = forest_graph.inference_graph(features)
        pred_classes = tf.argmax(logits, 1)
        pred_probas = tf.nn.softmax(logits)
#         accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        if(classif_core.classifier_properties.loss_type == 'softmax'):
            loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(
                                        labels=labels, logits=logits))
    return loss, pred_classes, pred_probas, train_op, loss_op

# 
def sec2win( total_len, matlab_load_core):
    return int(np.floor((total_len-matlab_load_core.window_size_sec)/matlab_load_core.stride_sec) +1)


def win2sec(win_num, matlab_load_core):
    return ((np.abs(win_num)-1)*matlab_load_core.stride_sec + matlab_load_core.window_size_sec)*np.sign(win_num)

def half1D_to_2D(arr, num_nodes):  
    tri = np.zeros((num_nodes, num_nodes))
    tri[np.triu_indices(num_nodes)] = arr
    tri = (tri+tri.T)/2
    return tri


class Network_Stats (object):
    def __init__(self, net):
        N, K = net.order(), net.size()
        self.avg_deg = float(K) / N
#         self.SCC = nx.number_strongly_connected_components(net)
#         self.WCC = nx.number_weakly_connected_components(net)
        # Clustering coefficient of all nodes (in a dictionary)
        self.clust_coefficients = np.fromiter(nx.clustering(net).values(), dtype=float)
        
        # Connected components are sorted in descending order of their size
#         self.cam_net_components = nx.connected_component_subgraphs(net)
#         self.cam_net_mc = self.cam_net_components[0]
        # Betweenness 
        self.centralitybet_cen = np.fromiter(nx.betweenness_centrality(net).values(), dtype=float)
        
        # Closeness 
        self.centralityclo_cen = np.fromiter(nx.closeness_centrality(net).values(), dtype=float)
        
        # Eigenvector 
        self.centralityeig_cen = np.fromiter(nx.eigenvector_centrality(net).values(), dtype=float)
        
        
    def get_all(self):
        return  [self.avg_deg,  np.mean(self.clust_coefficients),np.min(self.clust_coefficients),np.max(self.clust_coefficients),\
                                np.mean(self.centralitybet_cen), np.min(self.centralitybet_cen), np.max(self.centralitybet_cen),\
                                np.mean(self.centralityclo_cen), np.min(self.centralityclo_cen), np.max(self.centralityclo_cen),\
                                np.mean(self.centralityeig_cen), np.min(self.centralityeig_cen), np.max(self.centralityeig_cen)]   
        
def network_stats_calc(W_total, matlab_load_core):
    all_stats = []
    for i in range(W_total.shape[0]):
        W = half1D_to_2D(W_total[i,...], matlab_load_core.num_nodes) 
        adj_threshold = 0
        if(not np.any(W<0)):
            adj_threshold = np.min((np.mean(W), np.min(np.diag(W))))
        A = np.where(W>=adj_threshold, 1, 0)
        rows, cols = np.where(A == 1)
        edges = zip(rows.tolist(), cols.tolist())
        G = nx.Graph()
        G.add_edges_from(edges)
        net = G
        all_stats.append(Network_Stats(net).get_all())
    return np.vstack(all_stats), None
        