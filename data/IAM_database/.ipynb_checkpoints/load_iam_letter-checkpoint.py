# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 13:06:25 2019

@author: Xiaoyang
"""

import sys
import numpy as np

from gm.utils import get_labels_cxl,load_gv_nodexy

level = 'LOW'
level = 'MED'
level = 'HIGH'

if sys.platform=='win32':
    datapath0 = 'C:\\Users\\student\\Dropbox\\GraphMetrics\\data\\iam_database\\letter\\'
    datapath = datapath0 + level+'\\'
else:
    datapath0 = '/Users/Xiaoyang/Dropbox/GraphMetrics/data/iam_database/letter/'
    datapath = datapath0 + level+'/'
    
y_map = {'A':0, 'E':1, 'F':2, 'H':3, 'I':4, 'K':5, 'L':6, 'M':7, 'N':8, 
         'T':9, 'V':10, 'W':11, 'X':12, 'Y':13, 'Z':14}

y_train,train_names,ntrn = get_labels_cxl(datapath+'train.cxl',y_map)
y_valid,valid_names,nval = get_labels_cxl(datapath+'validation.cxl',y_map)
y_test,test_names,ntst = get_labels_cxl(datapath+'test.cxl',y_map)

# load graphs
G_train,pos_train = load_gv_nodexy(datapath,train_names)
G_valid,pos_valid = load_gv_nodexy(datapath,valid_names)
G_test,pos_test = load_gv_nodexy(datapath,test_names)
np.save(datapath0+'letter_train_'+level,[G_train,pos_train,y_train])
np.save(datapath0+'letter_valid_'+level,[G_valid,pos_valid,y_valid])
np.save(datapath0+'letter_test_'+level,[G_test,pos_test,y_test])