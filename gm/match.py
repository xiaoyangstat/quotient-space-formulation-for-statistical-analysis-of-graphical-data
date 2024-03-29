# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 15:54:43 2019

@author: Xiaoyang Guo, Adam Duncan
"""

import numpy as np
import networkx as nx
from scipy.linalg import eig
from scipy.optimize import linear_sum_assignment

#import matlab.engine # to run faq Matlab code in Python; this is deprecated
from .faq import sfw # the Python version of FAQ

def mat_sqdist(A,B):
    return np.sum(np.power(A-B,2) )/2 # divided by 2 because of symmetry

def eig_sorted(A):
    L,U = eig(A)
    s = np.argsort(L) #ascending
    s = s[::-1] #descending
    U[:,s]
    return L[s],U[:,s]

def perm_list_to_mat(p):
    n = len(p)
    P = np.zeros((n,n))
    for i in range(n):
        P[i,p[i]] = 1
    return P

def perm_mat_to_list(P):
    return np.where(P)[1]

def permutate_adjmat(p,A):
    """Permutate adjacency matrix A by permutation list/matrix p
    """
    p = np.array(p)

    if p.ndim==1:
        return A[p,:][:,p]
    elif p.ndim==2:
        return np.dot(p, np.dot(A,p.T))

def permutate_nx(p,G):
    """Permutate networkx graph object.

    Not working for nx.to_numpy_matrix()!!!
    """
    p = np.array(p)

    if p.ndim==2:
        p = perm_mat_to_list(p)

    return nx.relabel_nodes(G, dict(zip(p,range(len(p)))))

def umeyama_perm_mat(A,B,D_nd=None):
    """Spectral based graph matching by Umeyama 1988
    """
    La,Ua = eig_sorted(A)
    Lb,Ub = eig_sorted(B)

    X = np.dot( np.abs(Ub), np.abs(Ua.T) )

    if D_nd is None:
        p = linear_sum_assignment(-X)
    else:
        p = linear_sum_assignment(D_nd-X)

    P = perm_list_to_mat(p[1])
    Ap = permutate_adjmat(P,A)

    return Ap,P

def hill_climb_mat(A,B,P0=None,D_nd=None,max_hc=None):
    """Local node exchange.

    Improve the matching after umeyama_perm_mat.
    """
    n = A.shape[0]
    def step(A,B,D,P): # find best swap
        Ap = permutate_adjmat(P,A)
        E = mat_sqdist(Ap,B) + np.sum(D[P==1])
        I,J = 0,0 # default is no swap
        for i in range(n-1):
            for j in range(i+1,n):
                P[[i,j],:] = P[[j,i],:]
                Ap = permutate_adjmat(P,A)
                Ep = mat_sqdist(Ap,B) + np.sum(D[P==1])
                if Ep < E:
                    E,I,J = Ep,i,j
                P[[i,j],:] = P[[j,i],:]
        P[[I,J],:] = P[[J,I],:]
        return E

    if D_nd is None:
        D_nd = np.zeros((n,n))
    if P0 is None:
        P = np.identity(n)
    else:
        P = P0.copy()
    Ap = permutate_adjmat(P,A)
    E = mat_sqdist(Ap,B) + np.sum(D_nd[P==1])
    
    if E==0:
        return Ap,P,E
    
    Ep = step(A,B,D_nd,P)

    k = 0
    while Ep < E and (max_hc is None or k<max_hc):
        E = Ep
        Ap = permutate_adjmat(P,A)
        Ep = step(A,B,D_nd,P)
        k += 1
    return Ap,P,E

def umeyama_then_hill_climb_mat(A,B,D_nd=None, max_hc=None):
    Ap,P = umeyama_perm_mat(A,B, D_nd)
    Ap,P, E = hill_climb_mat(A,B, P,D_nd, max_hc=max_hc)

    return Ap,P,E

def match_extended_nx(G1,G2,laplacian=False,two_way=False,
                      use_node=False,w=1.0,attr='v',
                      algo='umeyama',max_hc=None,paral=False):
    """Graph Matching

    Args:
        w: tuning parameter to balance edge attributes and node attributes

    Returns:
        G1p: permutated graph
        G2p: G2 with some null nodes added
        p: permutation list
        d: distance after matching
        d0: distance before matching
    """
    
    n1 = G1.number_of_nodes()
    n2 = G2.number_of_nodes()

    if two_way:
        N = n1+n2
    else:
        N = np.max([n1,n2])

    ## distance matrix for node attributes
    D = np.zeros((N,N))
    if use_node:
        D = use_node(G1,G2,attr=attr,two_way=two_way)
        
    # null nodes padding
    if two_way:
        G1.add_nodes_from(range(n1,N))
        G2.add_nodes_from(range(n2,N))
    else:
        if n1<=n2:
            G1.add_nodes_from(range(n1,n2))
        else:
            G2.add_nodes_from(range(n2,n1))

    if laplacian:
        #print('using laplacian matrix instead of adjacency matrix')
        A1 = nx.laplacian_matrix(G1).todense()
        A2 = nx.laplacian_matrix(G2).todense()
    else:
        A1 = nx.to_numpy_matrix(G1)
        A2 = nx.to_numpy_matrix(G2)

    w = w*w
    d0 = np.sqrt(mat_sqdist(A1,A2) + w*sum(np.diagonal(D))) # orignal distance

    if algo == 'faq':
        #print('matching graphs using fast approximate quadratic programming')
        # Matlab
        #eng = matlab.engine.start_matlab()
        #eng.addpath(r'./gm/faq/',nargout=0)
        #A2m = -1*A2
        #A1m = matlab.double(A1.tolist())
        #A2m = matlab.double(A2m.tolist())
        # = eng.sfw(A2m,A1m,30)
        #p = np.array(p)[0].astype('int')
        #p = [i-1 for i in p]

        # Python
        ##debug##
        # import scipy.io as sio
        # sio.savemat('test.mat', {'A1':A1,'A2':A2})
        ##
        p = sfw(-1*A2,A1,w*D.transpose())
        A1p = permutate_adjmat(p,A1)
        P = perm_list_to_mat(p)
    else:
        #print('matching graphs using umeyama and hill climb')
        A1p,P, _= umeyama_then_hill_climb_mat(A1,A2,w*D,max_hc=max_hc)
        p = perm_mat_to_list(P)

    if two_way:
        pinv = perm_mat_to_list(P.T)
        n0 = n1
        for n in range(n1,N):
            if pinv[n]<n2: # null to node of g2
                P[:,[n0,n]] = P[:,[n,n0]] #exchange two colums
                n0 += 1
        n0 = n2
        for n in range(n2,N):
            if p[n]<n1: # node of g1 to null
                P[[n0,n],:] = P[[n,n0],:] #exchange two rows
                n0 += 1
        p = perm_mat_to_list(P)

    d = np.sqrt(mat_sqdist(A1p,A2) + w*sum(D[P==1]))

    if paral: # for parallel computing purpose
        return d

    G1p = permutate_nx(p,G1)
    G2p = G2.copy()

    # remove null nodes
    for n in range(n2):
        if p[n] >= n1: # nth node of G1 is null
            # copy attributes of n from G2 to G1
            #G1p.node[n] = G2.node[n].copy()
            G1p.nodes[n].update(G2.nodes[n])
    for n in range(n2,N):
        if p[n] >= n1: #both is null
            # remove node n from both
            G1p.remove_node(n)
            G2p.remove_node(n)
        else: #nth node of G2 is null
            # copy position of node n from G1 to G2
            #G2p.node[n] = G1p.node[n].copy()
            G2p.nodes[n].update(G1p.nodes[n])
            
    if two_way:
        p = p[:n0]
                
    G1.remove_nodes_from(range(n1,N))
    G2.remove_nodes_from(range(n2,N))
    
    return G1p,G2p,p,d,d0