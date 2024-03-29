# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 15:49:13 2019

@author: Xiaoyang Guo, Adam Duncan
"""

import numpy as np
import networkx as nx
from xml.etree.ElementTree import ElementTree as ET
from collections import defaultdict

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

## the following codes are copied/modified from: https://github.com/tvayer/FGW

def per_section(it, is_delimiter=lambda x: x.isspace()):
    ret = []
    for line in it:
        if is_delimiter(line):
            if ret:
                yield ret  # OR  ''.join(ret)
                ret = []
        else:
            ret.append(line.rstrip())  # OR  ret.append(line)
    if ret:
        yield ret

def graph_label_list(path,name):
    graphs=[]
    with open(path+name) as f:
        sections = list(per_section(f))
        k=1
        for elt in sections[0]:
            graphs.append((k,int(elt)))
            k=k+1
    return graphs

def compute_adjency(path,name):
    adjency= defaultdict(list)
    with open(path+name) as f:
        sections = list(per_section(f))
        for elt in sections[0]:
            adjency[int(elt.split(',')[0])].append(int(elt.split(',')[1]))
    return adjency


def graph_indicator(path,name):
    data_dict = defaultdict(list)
    with open(path+name) as f:
        sections = list(per_section(f))
        k=1
        for elt in sections[0]:
            data_dict[int(elt)].append(k)
            k=k+1
    return data_dict

def node_labels_dic(path,name):
    node_dic=dict()
    with open(path+name) as f:
        sections = list(per_section(f))
        k=1
        for elt in sections[0]:
            node_dic[k]=int(elt)
            k=k+1
    return node_dic

def compute_weighted_adjency(path,name1,name2):
    adjency ={}
    with open(path+name1) as f1:
        with open(path+name2) as f2:
            sections1 = list(per_section(f1))
            sections2 = list(per_section(f2))
            for i,elt in enumerate(sections1[0]):
                adjency[(int(elt.split(',')[0]),int(elt.split(',')[1]))]=int(sections2[0][i].split(',')[0])
    return adjency

def node_attr_dic(path,name):
    node_dic=dict()
    with open(path+name) as f:
        sections = list(per_section(f))
        k=1
        for elt in sections[0]:
            node_dic[k]=[float(x) for x in elt.split(',')]
            k=k+1
    return node_dic

def indices_to_one_hot(number, nb_classes,label_dummy=-1):
    """Convert an iterable of indices to one-hot encoded labels."""
    
    if number==label_dummy:
        return np.zeros(nb_classes)
    else:
        return np.eye(nb_classes)[number]
    
def build_IMDB_dataset(path,s='MULTI',use_node_deg=False):
    graphs=graph_label_list(path,'IMDB-'+s+'_graph_labels.txt')
    adjency=compute_adjency(path,'IMDB-'+s+'_A.txt')
    data_dict=graph_indicator(path,'IMDB-'+s+'_graph_indicator.txt')

    data=[]
    for i in graphs:
        g=nx.Graph()
        for node in data_dict[i[0]]:
            g.graph['id'] = i[0]
            g.add_node(node)
            for node2 in adjency[node]:
                g.add_edge(node,node2)
        if use_node_deg:
            node_degree_dict=dict(g.nx_graph.degree())
            normalized_node_degree_dict={k:v/len(g.nx_graph.nodes()) for k,v in node_degree_dict.items() }
            nx.set_node_attributes(g.nx_graph,normalized_node_degree_dict,'attr_name')
        g = nx.convert_node_labels_to_integers(g)
        data.append((g,i[1]))
        
    return data


def build_MUTAG_dataset(path,one_hot=True):
    graphs=graph_label_list(path,'MUTAG_graph_labels.txt') # id and label
    adjency=compute_adjency(path,'MUTAG_A.txt')
    data_dict=graph_indicator(path,'MUTAG_graph_indicator.txt')
    node_dic=node_labels_dic(path,'MUTAG_node_labels.txt') # ya aussi des nodes attributes ! The fuck ?
    edge_dic=compute_weighted_adjency(path,'MUTAG_A.txt','MUTAG_edge_labels.txt')
    edge_dic_transform = {0:1.5,1:1,2:2,3:3}

    data=[]
    for i in graphs:
        g=nx.Graph()
        for node in data_dict[i[0]]:
            g.graph['id'] = i[0]
            g.add_node(node)
            if one_hot:
                attr=indices_to_one_hot(node_dic[node],7)
                g.nodes[node]['attr'] = attr
            else:
                g.nodes[node]['attr'] = node_dic[node]
            for node2 in adjency[node]:
                g.add_edge(node,node2,weight=edge_dic_transform[edge_dic[(node,node2)]])

        g = nx.convert_node_labels_to_integers(g)
        data.append((g,i[1]))

    return data


def build_BZR_dataset(path,type_attr='label',use_node_deg=False):
    graphs=graph_label_list(path,'BZR_graph_labels.txt')
    if type_attr=='label':
        one_hot=True
        node_dic=node_labels_dic(path,'BZR_node_labels.txt') # A voir pour les attributes
        node2vec=dict(zip([1,6,7,8,9,15,16,17,35,53],range(10)))
    if type_attr=='real':
        one_hot=False
        node_dic=node_attr_dic(path,'BZR_node_attributes.txt')
    adjency=compute_adjency(path,'BZR_A.txt')
    data_dict=graph_indicator(path,'BZR_graph_indicator.txt')
    data=[]
    for i in graphs:
        g=nx.Graph()
        for node in data_dict[i[0]]:
            g.graph['id'] = i[0]
            g.add_node(node)
            if not use_node_deg:
                if one_hot:
                    attr=indices_to_one_hot(node2vec[node_dic[node]],10)
                else:
                    attr=node_dic[node]
                g.nodes[node]['attr'] = attr
            for node2 in adjency[node]:
                g.add_edge(node,node2)
        if use_node_deg:
            node_degree_dict=dict(g.degree())
            normalized_node_degree_dict={k:v/len(g.nx_graph.nodes()) for k,v in node_degree_dict.items() }
            nx.set_node_attributes(g,normalized_node_degree_dict,'attr')
        g = nx.convert_node_labels_to_integers(g)
        data.append((g,i[1]))

    return data


def build_ENZYMES_dataset(path,type_attr='label',use_node_deg=False):
    graphs=graph_label_list(path,'ENZYMES_graph_labels.txt')
    if type_attr=='label':
        one_hot=True
        node_dic=node_labels_dic(path,'ENZYMES_node_labels.txt') # A voir pour les attributes
        node2vec=dict(zip([1,2,3],range(3)))
    if type_attr=='real':
        one_hot=False
        node_dic=node_attr_dic(path,'ENZYMES_node_attributes.txt')
    adjency=compute_adjency(path,'ENZYMES_A.txt')
    data_dict=graph_indicator(path,'ENZYMES_graph_indicator.txt')
    data=[]
    for i in graphs:
        g=nx.Graph()
        for node in data_dict[i[0]]:
            g.graph['id'] = i[0]
            g.add_node(node)
            if not use_node_deg:
                if one_hot:
                    attr=indices_to_one_hot(node2vec[node_dic[node]],len(node2vec))
                else:
                    attr=node_dic[node]
                g.nodes[node]['attr'] = attr
            for node2 in adjency[node]:
                g.add_edge(node,node2)
        if use_node_deg:
            node_degree_dict=dict(g.nx_graph.degree())
            normalized_node_degree_dict={k:v/len(g.nx_graph.nodes()) for k,v in node_degree_dict.items() }
            nx.set_node_attributes(g.nx_graph,normalized_node_degree_dict,'attr_name')
        g = nx.convert_node_labels_to_integers(g)
        data.append((g,i[1]))

    return data


def build_PROTEIN_dataset(path,type_attr='label',use_node_deg=False):
    if type_attr=='label':
        one_hot=True
        node_dic=node_labels_dic(path,'PROTEINS_full_node_labels.txt') # A voir pour les attributes
    if type_attr=='real':
        one_hot=False
        node_dic=node_attr_dic(path,'PROTEINS_full_node_attributes.txt')
    graphs=graph_label_list(path,'PROTEINS_full_graph_labels.txt')
    adjency=compute_adjency(path,'PROTEINS_full_A.txt')
    data_dict=graph_indicator(path,'PROTEINS_full_graph_indicator.txt')
    data=[]
    for i in graphs:
        g=nx.Graph()
        for node in data_dict[i[0]]:
            g.graph['id'] = i[0]
            g.add_node(node)
            if not use_node_deg:
                if one_hot:
                    attr=indices_to_one_hot(node_dic[node],3)
                else:
                    attr=node_dic[node]
                g.nodes[node]['attr'] = attr
            for node2 in adjency[node]:
                g.add_edge(node,node2)
        if use_node_deg:
            node_degree_dict=dict(g.nx_graph.degree())
            normalized_node_degree_dict={k:v/len(g.nx_graph.nodes()) for k,v in node_degree_dict.items() }
            nx.set_node_attributes(g.nx_graph,normalized_node_degree_dict,'attr_name')
        g = nx.convert_node_labels_to_integers(g)
        data.append((g,i[1]))

    return data

def build_PTC_FM_dataset(path,one_hot=True):
    graphs=graph_label_list(path,'PTC_FM_graph_labels.txt') # id and label
    adjency=compute_adjency(path,'PTC_FM_A.txt')
    data_dict=graph_indicator(path,'PTC_FM_graph_indicator.txt')
    node_dic=node_labels_dic(path,'PTC_FM_node_labels.txt') # ya aussi des nodes attributes ! The fuck ?
    edge_dic=compute_weighted_adjency(path,'PTC_FM_A.txt','PTC_FM_edge_labels.txt')
    edge_dic_transform = {0:3,1:1,2:2,3:1.5}

    data=[]
    for i in graphs:
        g=nx.Graph()
        for node in data_dict[i[0]]:
            g.graph['id'] = i[0]
            g.add_node(node)
            if one_hot:
                attr=indices_to_one_hot(node_dic[node],18)
                g.nodes[node]['attr'] = attr
            else:
                g.nodes[node]['attr'] = node_dic[node]
            for node2 in adjency[node]:
                g.add_edge(node,node2,weight=edge_dic_transform[edge_dic[(node,node2)]])

        g = nx.convert_node_labels_to_integers(g)
        data.append((g,i[1]))

    return data

def build_PTC_FR_dataset(path,one_hot=True):
    graphs=graph_label_list(path,'PTC_FR_graph_labels.txt') # id and label
    adjency=compute_adjency(path,'PTC_FR_A.txt')
    data_dict=graph_indicator(path,'PTC_FR_graph_indicator.txt')
    node_dic=node_labels_dic(path,'PTC_FR_node_labels.txt') # ya aussi des nodes attributes ! The fuck ?
    edge_dic=compute_weighted_adjency(path,'PTC_FR_A.txt','PTC_FR_edge_labels.txt')
    edge_dic_transform = {0:3,1:2,2:1,3:1.5}

    data=[]
    for i in graphs:
        g=nx.Graph()
        for node in data_dict[i[0]]:
            g.graph['id'] = i[0]
            g.add_node(node)
            if one_hot:
                attr=indices_to_one_hot(node_dic[node],19)
                g.nodes[node]['attr'] = attr
            else:
                g.nodes[node]['attr'] = node_dic[node]
            for node2 in adjency[node]:
                g.add_edge(node,node2,weight=edge_dic_transform[edge_dic[(node,node2)]])

        g = nx.convert_node_labels_to_integers(g)
        data.append((g,i[1]))

    return data

def build_PTC_MR_dataset(path,one_hot=True):
    graphs=graph_label_list(path,'PTC_MR_graph_labels.txt') # id and label
    adjency=compute_adjency(path,'PTC_MR_A.txt')
    data_dict=graph_indicator(path,'PTC_MR_graph_indicator.txt')
    node_dic=node_labels_dic(path,'PTC_MR_node_labels.txt') # ya aussi des nodes attributes ! The fuck ?
    edge_dic=compute_weighted_adjency(path,'PTC_MR_A.txt','PTC_MR_edge_labels.txt')
    edge_dic_transform = {0:3,1:2,2:1,3:1.5}

    data=[]
    for i in graphs:
        g=nx.Graph()
        for node in data_dict[i[0]]:
            g.graph['id'] = i[0]
            g.add_node(node)
            if one_hot:
                attr=indices_to_one_hot(node_dic[node],18)
                g.nodes[node]['attr'] = attr
            else:
                g.nodes[node]['attr'] = node_dic[node]
            for node2 in adjency[node]:
                g.add_edge(node,node2,weight=edge_dic_transform[edge_dic[(node,node2)]])

        g = nx.convert_node_labels_to_integers(g)
        data.append((g,i[1]))

    return data

def build_NCI1_dataset(path):
    node_dic=node_labels_dic(path,'NCI1_node_labels.txt')
    node_dic2={}
    for k,v in node_dic.items():
        node_dic2[k]=v-1
    node_dic=node_dic2
    graphs=graph_label_list(path,'NCI1_graph_labels.txt')
    adjency=compute_adjency(path,'NCI1_A.txt')
    data_dict=graph_indicator(path,'NCI1_graph_indicator.txt')
    data=[]
    for i in graphs:
        g=nx.Graph()
        for node in data_dict[i[0]]:
            g.graph['id'] = i[0]
            g.add_node(node)
            attr=indices_to_one_hot(node_dic2[node],37)
            g.nodes[node]['attr'] = attr
            for node2 in adjency[node]:
                g.add_edge(node,node2)
 
        g = nx.convert_node_labels_to_integers(g)
        data.append((g,i[1]))

    return data