# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 09:47:38 2019

@author: Xiaoyang Guo, Adam Duncan
"""

import numpy as np

def node_xy_to_vec(G,attr='v'):
    pos = {}
    for n in G:
        x = float(G.nodes[n]['x'])
        y = float(G.nodes[n]['y'])

        pos[n] = (x,y)

        G.nodes[n][attr] = np.array((x,y))
        del G.nodes[n]['x']
        del G.nodes[n]['y']
    return pos

def node_vec_pos(G,attr='v'):
    pos = {}
    for n in G:
        pos[n] = tuple(G.nodes[n][attr])
    return pos

def node_square_dists(G1,G2,attr='v',two_way=False):
    """Compute node distance matrix 
    
    NxN matrix whose left top is n2xn1 real node distance
    """
    n1 = G1.number_of_nodes()
    n2 = G2.number_of_nodes()

    if two_way:
        N = n1+n2
    else:
        N = np.max([n1,n2])

    D = np.zeros((N,N))

    for i in range(n1):
        for j in range(n2):
            attr1 = np.array(G1.nodes[i][attr])
            attr2 = np.array(G2.nodes[j][attr])
            D[j,i] = np.sum((attr1-attr2)**2)

    for i in range(n2,N):
        for j in range(n1):
            D[i,j] = 0#np.sum((G1.node[i-n2][attr]-np.zeros(np.shape(G1.node[i-n2][attr])))**2)
    for i in range(n2):
        for j in range(n1,N):
            D[i,j] = 0#np.sum((G2.node[i][attr]-np.zeros(np.shape(G2.node[i][attr])))**2)

    #D[n2:,n1:] = np.zeros((n1,n2))

    return D

def node_binary_dists(G1,G2,attr='symbol',two_way=False):

    n1 = G1.number_of_nodes()
    n2 = G2.number_of_nodes()

    if two_way:
        N = n1+n2
    else:
        N = np.max([n1,n2])
        
    D = np.zeros((N,N))

    for i in range(n1):
        for j in range(n2):
#            if G1.node[nd1[i]]['symbol']==G2.node[nd2[j]]['symbol']:
            if G1.nodes[i][attr]==G2.nodes[j][attr]:
                D[j,i] = 0
            else:
                D[j,i] = 1
                
    return D

def protein_node_dist(G1,G2):
    """return a (n2+n1)x(n2+n1) matrix whose left top is n2xn1 real node distance
    """
    n1 = G1.number_of_nodes()
    n2 = G2.number_of_nodes()

    N = n1+n2
    D = np.empty((N,N))

    for i in range(n1):
        for j in range(n2):
            D[j,i] = abs(G1.node[i]['length'] - G2.node[j]['length'])


    for i in range(n2,N):
        D[i,:n1] = [G1.node[j]['length'] for j in range(n1)]

    for i in range(n1,N):
        D[:n2,i] = [G2.node[j]['length'] for j in range(n2)]

    D[n2:,n1:] = np.zeros((n1,n2))

    return D
