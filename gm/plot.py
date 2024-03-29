# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 15:56:42 2019

@author: Xiaoyang Guo, Adam Duncan
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout

def fill_pos(pos,n,deviation = 2):
    """fill out the missing position by mean+-deviation*std

    Args:
        n: number of total nodes
    """

    pos_mu = (np.mean([pos[i][0] for i in pos]),np.mean([pos[i][1] for i in pos]))
    pos_std = (np.std([pos[i][0] for i in pos]),np.std([pos[i][1] for i in pos]))

    pos1 = pos.copy()

    for i in range(n):
        if i not in pos:
            pos1[i] = (pos_mu[0]+np.random.uniform(-deviation,deviation)*pos_std[0],
                       pos_mu[1]+np.random.uniform(-deviation,deviation)*pos_std[1])
    return pos1

def remove_no_pos_nodes(G,pos):
    """remove nodes whose positions are not in pos

    Args:
        G: a networkx graph
        pos: dictionary, positions
    """
    g = G.copy()
    tmp = list(G.nodes())
    for nd in tmp:
        if nd not in pos:
            g.remove_node(nd)
    return g

def remove_weak_nodes(G,thr=0,verbose=False):
    """remove weak nodes of G

    Args:
        thr: threshold of weight under which the edge need to remove
        while graph edge weight are transformed to absolute value;
    """
    if thr<0:
        return G

    G1=G.copy()

    total_weight = 0
    removed_weight = 0
    for (u,v,d) in G.edges(data=True):
        total_weight += d['weight']
        if d['weight']<thr:
            removed_weight += d['weight']
            G1.edges[u,v]['weight']=0
            G1.remove_edge(u,v)

    if verbose:
        print('removed {:.2f}% weak edges'.format(100-100*len(G1.edges)/len(G.edges)))
        print(f'removed {removed_weight/total_weight*100}% weights')

    G2 = G1.copy()
    for i in nx.isolates(G2):
        G1.remove_node(i)
    return G1

def draw_weighted(W, title='', pos=None, width_factor=1,
                  thr=-1, thr2=None, draw=True, prog='neato', args='',
                  return_pos=False,label_name=None,**kwargs):
    if thr2 is None:
        thr2=thr
    else:
        thr2 = min(thr,thr2)

    E = W.edges(data=True)
    E1 = [(u,v,d) for (u,v,d) in E if thr<=d['weight']]
    wd1 = width_factor*np.array([d['weight'] for (u,v,d) in E1])
    if thr2<thr:
        E2 = [(u,v,d) for (u,v,d) in E if thr2<=d['weight']<thr]
        wd2 = width_factor*np.array([d['weight'] for (u,v,d) in E2])
    else:
        E2 = []; wd2 = []

    Wthr = nx.Graph()
    Wthr.add_nodes_from(W.nodes())
    Wthr.add_edges_from(E1)
    Wthr.add_edges_from(E2)

    if label_name == False:
        with_labels = False
        nodeDict = None
    elif label_name != None:
        with_labels = True
        nodeDict = {}
        for i in W.nodes:
            if label_name in W.nodes[i]:
                nodeDict[i] = W.nodes[i][label_name]
            else:
                nodeDict[i] = '*'
    else:
        with_labels = True
        nodeDict = None

    if pos is None:
        pos = graphviz_layout(Wthr, prog=prog, args=args)
    if draw:
        plt.title(title,fontdict={'fontsize':30,'weight':'bold','color':'navy'})
        nx.draw_networkx(Wthr,pos,edgelist=E1,width=wd1,
                         with_labels=with_labels,labels=nodeDict,node_color='skyblue',
                         **kwargs)
        if thr2<thr:
            nx.draw_networkx_edges(Wthr,pos,edgelist=E2,width=wd2,style='dashed')
        plt.axis('off')

    if return_pos:
        return pos
    else:
        return None
