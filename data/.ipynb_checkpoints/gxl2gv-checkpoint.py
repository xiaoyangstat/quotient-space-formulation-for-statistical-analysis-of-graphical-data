# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 13:11:44 2016

@author: Adam Duncan
"""

import os
from subprocess import call

def gxl2gv_dir(path,name):
    os.chdir(path)
    files = os.listdir('.')
    for f in files:
        if f[-4:] == '.gxl':
            print name + ": Converting " + f
            outstr = "-o" + f[:-4] + ".gv"
            call(["gxl2gv", outstr, f])

gxl2gv_dir('/home/adam/Dropbox/GraphMetrics/Datasets/IAM_Graph_Database/AIDS/data/',
           'AIDS')

gxl2gv_dir('/home/adam/Dropbox/GraphMetrics/Datasets/IAM_Graph_Database/Letter/Letter/HIGH',
           'Letter (HIGH)')
gxl2gv_dir('/home/adam/Dropbox/GraphMetrics/Datasets/IAM_Graph_Database/Letter/Letter/MED',
           'Letter (MED)')
gxl2gv_dir('/home/adam/Dropbox/GraphMetrics/Datasets/IAM_Graph_Database/Letter/Letter/LOW',
           'Letter (LOW)')

gxl2gv_dir('/home/adam/Dropbox/GraphMetrics/Datasets/IAM_Graph_Database/Protein/data',
           'Protein')
