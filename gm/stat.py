# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 15:55:15 2019

@author: Xiaoyang Guo, Adam Duncan
"""

import numpy as np
import networkx as nx
from time import time
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from joblib import Parallel,delayed

from .match import permutate_adjmat,match_extended_nx


def undirected_graphmat_to_vector(A):
    """Convert a symmetric adjacency matrix to the vector of entries above diagonal
    """
    n = A.shape[0]
    p = n*(n-1)//2

    v = np.empty(p)
    k0 = 0
    for i in range(n-1):
        k1 = k0+n-i-1
        v[k0:k1] = A[i,i+1:]
        k0 = k1
    return v

def undirected_graphmats_to_vectors(A):
    """Take a list of symmetric adjacency matrix to a matrix of entries above diagonal
    """
    N = len(A)
    X0 = undirected_graphmat_to_vector(A[0])
    p = np.size(X0)
    X = np.empty((N,p))
    X[0,:] = X0
    for i in range(1,N):
        X[i,:] = undirected_graphmat_to_vector(A[i])
    return X

def undirected_graph_to_vectors(G):
    N = len(G)
    X0 = undirected_graphmat_to_vector(nx.to_numpy_matrix(G[0]))
    p = np.size(X0)
    X = np.empty((N,p))
    X[0,:] = X0
    for i in range(1,N):
        X[i,:] = undirected_graphmat_to_vector(nx.to_numpy_matrix(G[i]))
    return X

def vectorize_graph_nd(G, p, attr='v', w=1.0):
    n = G.number_of_nodes()
    d = np.array(G.nodes[0][attr]).size

    Ne = n*(n-1)//2
    N = Ne + n*d
    V = np.empty(N)

    A = nx.to_numpy_matrix(G)
    A = permutate_adjmat(p,A)
    V[:Ne] = undirected_graphmat_to_vector(A)
    k0,k1 = Ne,Ne+d
    for i in range(n):
        V[k0:k1] = w*G.nodes[i][attr]
        k0=k1; k1+=d
    return V

def vector_to_undirected_graphmat(v):
    p = np.size(v)
    n = int(np.sqrt(2*p))+1
    A = np.zeros((n,n))

    k0 = 0
    for i in range(n-1):
        k1 = k0+n-i-1
        A[i,i+1:] = v[k0:k1]
        k0 = k1
    A = A+A.T
    return A

def unvectorize_graph_nd(V, n,d, attr='v', w=1.0):
    Ne = n*(n-1)//2
#    N = Ne + n*d
    A = vector_to_undirected_graphmat(V[:Ne])
    G = nx.from_numpy_matrix(A)

    k0,k1 = Ne,Ne+d
    for i in range(n):
        G.nodes[i][attr] = V[k0:k1]/w
        k0=k1; k1+=d
    return G

## Karcher Mean

def original_mean(G):
    """Compute the adjacency mean
    """
    G0 = G.copy()

    N = len(G0)
    A = [None]*N

    nm = np.max([g.number_of_nodes() for g in G0])
    for i,g in enumerate(G0):
        A[i] = np.zeros((nm,nm))
        n = g.number_of_nodes()
        g.add_nodes_from(range(n,nm))
        A[i] = nx.to_numpy_matrix(g)

    return nx.from_numpy_matrix(np.mean(A,axis = 0))

def muG_aligned(G,P,num_attr=False,attr='v'):
    """Average aligned graphs's edge weights and node attributes

    Args:


    """
    N = len(G)
    n = [g.number_of_nodes() for g in G]
    nmax = max(n)

    # average edge weights
    A = [None]*N
    v = [0]*nmax
    vct = [0]*nmax
    for i in range(N):
        A[i] = np.zeros((nmax,nmax))
        A[i][:n[i],:n[i]] = permutate_adjmat(P[i],nx.to_numpy_matrix(G[i]))
        if num_attr:
            for j, nd in enumerate(G[i]):
                if 'fict' not in G[i].nodes[j]:
                    v[j] += np.array(G[i].nodes[j][attr])
                    vct[j] += 1

    muA = np.mean(A,0)
    muG = nx.from_numpy_matrix(muA)

    if not num_attr:
        return muG

    # average node coordinate
    dead_nodes = []
    for nd in muG:
        if vct[nd]>0:
            muG.nodes[nd][attr] = v[nd]/vct[nd]
        else:
            dead_nodes.append(nd)

    if dead_nodes:
        print('dead nodes: ', dead_nodes)

    # fill up input graphs to match cardinalities
    for i in range(N):
        for nd in range(n[i],nmax):
                G[i].add_node(nd)
                G[i].nodes[nd]['fict']=True
                #if nd not in dead_nodes:
                G[i].nodes[nd][attr]=muG.nodes[nd][attr]

    muG.remove_nodes_from(dead_nodes)
    for g in G:
        g.remove_nodes_from(dead_nodes)

    muG = nx.convert_node_labels_to_integers(muG, ordering='sorted')
    for i in range(N):
        G[i] = nx.convert_node_labels_to_integers(G[i], ordering='sorted')

    return muG

def iterative_mean_graph_ext_nx(G,mu_init=None,max_itr=30,two_way=False,
                                use_node=False,w=1.0,num_attr=False,attr='v',
                                algo='umeyama',max_hc=None):
    if mu_init is None:
        i = np.argmax([g.number_of_nodes() for g in G])
        muG = G[i].copy()
    else:
        muG = mu_init

    N = len(G)
    Gp = [None]*N
    E = [0]*(max_itr+2)

    print(f"first pass: muG has {muG.number_of_nodes()} nodes")
    start=time()
    P0 = []
    for k in range(N):
        Gp[k],muG,p,d,d0=match_extended_nx(G[k],muG,two_way=two_way,
        use_node=use_node,w=w, attr=attr,algo = algo,max_hc=max_hc)
        for nd in muG:
            if p[nd] not in G[k]:#null node of Gp
                Gp[k].nodes[nd]['fict'] = True
        P0.append(p)
        E[0] += d0*d0
        E[1] += d*d
        #print ("G{:d} has {:d} nodes and muG has {:d} nodes".format(k,Gp[k].number_of_nodes(),muG.number_of_nodes()))
    print("first pass time: {:.2f}s".format(time()-start))

    muG = muG_aligned(Gp, P0, num_attr=num_attr,attr=attr)
    mu_list = [muG.copy()]

    start=time()
    for m in range(max_itr):
        Ptm = []
        print(f"starting iteration {m+1}/{max_itr}, muG has {muG.number_of_nodes()} nodes")
        for k in range(N):
            Gp[k],muG,p,d,_=match_extended_nx(G[k],muG,two_way=two_way,
            use_node=use_node,w=w, attr=attr,algo = algo,max_hc=max_hc)
            Ptm.append(p)
            for nd in muG:
                if p[nd] not in G[k]:
                    Gp[k].nodes[nd]['fict'] = True
        E[m+2] += d*d
            #print ("iteration: {:d}/{:d}; G{:d} has {:d} nodes and muG has {:d} nodes".format(m+1,max_itr,k,Gp[k].number_of_nodes(),muG.number_of_nodes()))

        muG = muG_aligned(Gp,Ptm,num_attr=num_attr,attr=attr)
        mu_list.append(muG.copy())
        print("finished iteration {:d}/{:d} and time so far {:.2f}s".format(m+1,max_itr,time()-start))

    return muG, Gp, E, mu_list, Ptm

## PCA

def pcaG_aligned(G, P=None, attr='v', w=1.0):

    nodes=[]
    for g in G:
        nodes.append(g.number_of_nodes())
    #print('nodes number are: ', nodes)
    if len(set(nodes))>1:
        # two way null nodes padding is not always converged
        print('null nodes will be added')
    else:
        print('graphs are equal size')
    nmax=max(nodes)
    d = np.array(G[0].nodes[0][attr]).size

    V = np.empty((len(G),nmax*(nmax-1)//2+nmax*d))

    for i,g in enumerate(G):
        if not P:
            p=[]
            for nd in range(g.number_of_nodes()):
                p.append(np.where(nd == np.array(g.nodes))[0][0])
        else:
            p = P[i]

        V[i,:] = vectorize_graph_nd(g, p, attr=attr, w=w)

    pca = PCA()
    scores = pca.fit_transform(V)
    return pca,scores, V

def pcaG_aligned_edge(Gp, P=None):
    """PCA for aligned graphs, edge only
    """
    nodes=[]
    for g in Gp:
        nodes.append(g.number_of_nodes())
    print('nodes number are: ', nodes)
    if len(set(nodes))>1:
        print('null nodes will be added')
    nmax=max(nodes)
    #vectorize
    V = np.empty((len(Gp),nmax*(nmax-1)//2))
    for i,g in enumerate(Gp):
        A = np.zeros((nmax,nmax))
        if not P:
            p=[]
            for nd in range(g.number_of_nodes()):
                p.append(np.where(nd == np.array(g.nodes))[0][0])
        else:
            p = P[i]
        A[:nodes[i],:nodes[i]] = permutate_adjmat(p,nx.to_numpy_matrix(g))
        V[i,:] = undirected_graphmat_to_vector(A)

    pca = PCA()
    scores = pca.fit_transform(V)

    return pca,scores, V

def pca_graphs_to_scores(pca,G, attr='v', w=1.0):
    nGraphs = len(G)
    nDims = pca.components_.shape[1]
    V = np.empty((nGraphs,nDims))
    for i in range(nGraphs):
        Vi = vectorize_graph_nd(G[i], attr=attr,w=w)
        if Vi.size != nDims:
            raise ValueError('Input graph {:d} has wrong shape for pca object.'.format(i))
        else:
            V[i,:] = Vi
    return pca.transform(V)

def pca_scores_to_graphs(pca,scores,n,d, attr='v', w=1.0):
    nGraph = scores.shape[0]
    nComp = pca.components_.shape[0]
    if scores.shape[1] < nComp:
        pad = np.zeros((nGraph,nComp-scores.shape[1]))
        scores = np.concatenate((scores,pad),axis=1)
    elif scores.shape[1] > nComp:
        raise ValueError('Score vectors too long for pca object.')
    V = pca.inverse_transform(scores)
    G = []
    for i in range(nGraph):
        G.append( unvectorize_graph_nd(V[i,:],n,d, attr=attr,w=w) )
    return G

def pca_scores_to_graphs_structure(pca,scores):
    """Reconstruct the pca score to graph, edge only.
    """
    nGraph = scores.shape[0]
    nComp = pca.components_.shape[0]
    if scores.shape[1] < nComp:
        pad = np.zeros((nGraph,nComp-scores.shape[1]))
        scores = np.concatenate((scores,pad),axis=1)
    elif scores.shape[1] > nComp:
        raise ValueError('Score vectors too long for pca object.')
    V = pca.inverse_transform(scores)
    G = []
    for i in range(nGraph):
        G.append( nx.from_numpy_matrix(vector_to_undirected_graphmat(V[i,:]) ))
    return G

## distance matrix

def compute_distmat(G1,G2=None,two_way=False,k_print=None,
                    use_node=False, w=1.0, attr='v',
                    algo = 'umeyama', max_hc=None):
    """Compute pairwise distance matrix in Graph Space for G1 (List) and G2 (List)
    
    Returns:
        D: graph distance matrix
        D0: original distance matrix
    """
    n1 = len(G1)
    symm = (G2 is None)
    if symm:
        G2 = G1
        n2 = n1
        ncmp = int(n1*(n1-1)/2)
    else:
        n2 = len(G2)
        ncmp = n1*n2

    print ('computing distance matrix with {:d} comparisons.'.format(ncmp))

    if k_print is None:
        k_print = ncmp+1

    D = np.empty((n1,n2)) # graph distance
    D0 = np.empty((n1,n2)) # original distance

    start = time()
    j0 = 0
    k=0
    for i in range(n1):
        if symm:
            j0=i+1
            D[i,i]=0
        for j in range(j0,n2):
            #print(i,j)
            g1 = G1[i].copy()
            g2 = G2[j].copy()
            _,_,_,D[i,j],D0[i,j]= match_extended_nx(g1,g2,two_way=two_way,
            w = w,use_node=use_node, attr = attr,algo = algo, max_hc=max_hc)
            if symm:
                D[j,i] = D[i,j]
                D0[j,i] = D0[i,j]
            k += 1
            if k%k_print==0:
                print ('finished {:d} of {:d} comparisons'.format(k,ncmp))
                print ('time so far: {:5.3f}s'.format(time()-start))

    print('done! time so far: {:5.3f}s'.format(time()-start))

    return D, D0

def compute_distmat_paral(G1,G2=None,n_jobs=4,two_way=False,
                         use_node=False,w=1.0,attr='v',
                         algo='umeyama',max_hc=None):
    """parallel version of extended distance matrix.
    """
    n1 = len(G1)
    symm = (G2 is None)
    if symm:
        G2 = G1
        n2 = n1
        ncmp = int(n1*(n1-1)/2)
    else:
        n2 = len(G2)
        ncmp = n1*n2

    print (f'parallel computing distmat with {ncmp} comparisons using {n_jobs} cores')

    j0 = 0
    result=[None]*n1
    start = time()
    for i in range(n1):
        if i%(n1/5)==0: 
            print(f'computing row {i+1}/{n1}')
        if symm:
            j0=i+1
        result[i] = Parallel(n_jobs=n_jobs)(delayed(match_extended_nx)(G1[i].copy(),G2[j].copy(),
              use_node=use_node,w = w,attr = attr,paral=True,algo=algo) for j in range(j0,n2))
    
    print('done! time so far: {:5.3f}s'.format(time()-start))

    ##
    D = np.full((n1,n2),0)
    if symm:
        for i in range(n1):
            for j in range(i+1,n1):
                D[i,j]=result[i][j-i-1]
                D[j,i]=D[i,j]
    else:
        D = np.array(result)

    return D


## Classification

def svm_rbf_distmat(D_train,D_valid,D_test,
                    y_train,y_valid,y_test,
                    C_vec,gam_vec):
    #For kernel=”precomputed”, the expected shape of X is [n_samples_test, n_samples_train]
    n_train = D_train.shape[0]
    if D_valid.shape[1]!=n_train:
        D_valid = D_valid.T
    if D_test.shape[1]!=n_train:
        D_test = D_test.T
    
    N_C = C_vec.size
    N_gamma = gam_vec.size
    
    acc_grid = np.empty((N_C,N_gamma))
    
    print ('fitting {:d} models...'.format(N_C*N_gamma))
    for i in range(N_C):
        C = C_vec[i]
        for j in range(N_gamma):
            gamma = gam_vec[j]
            
            Grbf_train = np.exp(-gamma*D_train**2)
            Grbf_valid = np.exp(-gamma*D_valid**2)
            
            svc = SVC(C=C,kernel='precomputed',class_weight='balanced')
            svc.fit(Grbf_train,y_train)
            y_valid_pred = svc.predict(Grbf_valid)

            acc_grid[i,j] = accuracy_score(y_valid,y_valid_pred)

    I = np.argmax(acc_grid)
    i,j = (I//acc_grid.shape[1], I%acc_grid.shape[1])
#    acc_max_valid = acc_grid[i,j]

    C = C_vec[i]
    gamma = gam_vec[j]

    Grbf_train = np.exp(-gamma*D_train**2)
    Grbf_valid = np.exp(-gamma*D_valid**2)
    Grbf_test = np.exp(-gamma*D_test**2)

    svc = SVC(C=C,kernel='precomputed',class_weight='balanced')
    svc.fit(Grbf_train,y_train)

    y_train_pred = svc.predict(Grbf_train)
    y_valid_pred = svc.predict(Grbf_valid)
    y_test_pred = svc.predict(Grbf_test)

    return y_train_pred,y_valid_pred,y_test_pred, acc_grid,C,gamma

def kNN_tunek(D_valid,D_test,y_train,y_valid,y_test,kmax=None):
    if kmax is None:
        k = []
        for y in [y_train,y_valid,y_test]:
            k.extend(np.sqrt(np.bincount(y)[np.unique(y)]))
        kmax = int(np.min(k))
    nval = len(y_valid)
    ntst = len(y_test)
    kNN_valid_inds = np.argsort(D_valid,0) #axis = 0, column
    kNN_valid_pred = y_train[kNN_valid_inds]

    kNN_test_inds = np.argsort(D_test,0)
    kNN_test_pred = y_train[kNN_test_inds]

    y_valid_pred = np.empty((kmax,nval),int)
    kNN_ValSuccess = np.empty(kmax)
    for k in range(1,kmax+1):

        for i in range(nval):
            y_valid_pred[k-1,i] = np.argmax(np.bincount(kNN_valid_pred[:k,i]))

        kNN_ValSuccess[k-1] = np.mean(y_valid==y_valid_pred[k-1,:])

    k = np.argmax(kNN_ValSuccess)+1
    y_test_pred = np.empty(ntst,int)
    for i in range(ntst):
        y_test_pred[i] = np.argmax(np.bincount(kNN_test_pred[:k,i]))
    return y_test_pred,k
