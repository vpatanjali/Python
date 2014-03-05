# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 21:50:01 2014

@author: patanjali
"""

import os, gc
import numpy

workspace = 'G:/Kaggle/Default/'
filenames = [['train_v2_part1', 'train_v2_part2']]#,
#             ['test_v2_part1'], ['test_v2_part2'], 
#             ['test_v2_part3'], ['test_v2_part4']]
DLM = ','

os.chdir(workspace)

for file_group in filenames:
    data_pieces = [0 for i in xrange(len(file_group))]
    for i, file_ in enumerate(file_group):
        print file_
        data_pieces[i] = numpy.genfromtxt(file_ + ".csv", delimiter=DLM)
    data = numpy.vstack(data_pieces)
    numpy.save(file_ + ".npy", data)
    del data_pieces, data
    gc.collect()
