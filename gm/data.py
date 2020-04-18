# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 15:49:13 2019

@author: Xiaoyang Guo, Adam Duncan
"""

import numpy as np
import networkx as nx
from xml.etree.ElementTree import ElementTree as ET

from .node import node_xy_to_vec

def make_binary_weights(G):
    """Assign binary weights to edges

    Parameters
    ----------
    G : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    for u,v in G.edges():
        G[u][v]['weight'] = 1

def make_valence_weights(G):
    """Assign valence weights to edges

    Parameters
    ----------
    G : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    for u,v in G.edges():
        G[u][v]['weight'] = int(G[u][v]['valence'])

# def make_frequency_weights(G):
    
#     for u,v in G.edges():
#         G[u][v]['weight'] = int(G[u][v]['frequency'])

def fix_atom_nodes(G):
    for n in G.nodes():
        G.node[n]['symbol'] = str(G.node[n]['symbol'].strip())
        G.node[n]['charge'] = int(G.node[n]['charge'])

def get_labels_cxl(infile,y_map=None):
    labelET = ET().parse(infile)
    n = int(labelET[0].attrib['count'])
    names = [None]*n
    y = np.empty(n,dtype=int)
    if y_map:
        for i in range(n):
            e = labelET[0][i]
            names[i] = e.attrib['file'][:-4]
            y[i] = y_map[e.attrib['class']]
    else:
        for i in range(n):
            e = labelET[0][i]
            names[i] = e.attrib['file'][:-4]
            y[i] = e.attrib['class']
    return y,names,n

def load_gv_nodexy(path,names):
    n = len(names)
    G = [None]*n
    pos = [None]*n
    for i in range(n):
        G[i] = nx.nx_agraph.read_dot(path+names[i]+'.gv')
        G[i] = nx.Graph(G[i])
        G[i] = nx.convert_node_labels_to_integers(G[i])
        make_binary_weights(G[i])
        pos[i] = node_xy_to_vec(G[i])
    return G,pos

def load_gv_molecule(path,names):
    n = len(names)
    G = [None]*n
    pos = [None]*n
    for i in range(n):
        G[i] = nx.nx_agraph.read_dot(path+names[i]+'.gv')
        G[i] = nx.Graph(G[i])
        G[i] = nx.convert_node_labels_to_integers(G[i])
        make_valence_weights(G[i])
        pos[i] = node_xy_to_vec(G[i])
        fix_atom_nodes(G[i])
    return G,pos

def load_gxl_protein(path,names):
    n = len(names)
    G = []
    for i in range(n):
        g = nx.Graph()
        tree = ET().parse(path+names[i]+'.gxl')
        for node in tree.findall(".//node"):
            for nodeattr in node.findall("./attr[2]/int"):
                g.add_node(node.get('id'),length=int(nodeattr.text))
        for edge in tree.findall(".//edge"):
            start = edge.get('from')
            end = edge.get('to')
            for weight in edge.findall("./attr[3]/double"): # distance0
                g.add_edge(start,end,weight=float(weight.text))
        G.append( nx.convert_node_labels_to_integers(g))
    return G
