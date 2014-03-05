# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 21:45:16 2014

@author: patanjali
"""
#%%
workspace = 'G:/Kaggle/Default/'
train_filename = 'train_v2'
test_filenames = ['test_v2_part1', 'test_v2_part2', 'test_v2_part3', 
                  'test_v2_part4']

impute = True
scale = False

import os, gc
import numpy
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import Imputer

eps = 1e-12

#transformer = lambda x: numpy.log(numpy.maximum(eps,x))
#transformer.__name__ = 'log_rlu'
transformer = lambda x: x
transformer.__name__ = 'none'

suffix = '_' + transformer.__name__ + '_scale'*scale + '_impute'*impute
imputer = Imputer(copy = False)
scaler = MinMaxScaler((0,100), copy = False)

#%% Reading the raw data

os.chdir(workspace)

data = numpy.load(train_filename + '.npy')

#%% Sampling and splitting
idvs, dvs = numpy.hsplit(data,[-1])

del data
gc.collect()

#%% Imputing and scaling
if impute:
    idvs = imputer.fit_transform(idvs)
if scale:
    idvs = scaler.fit_transform(idvs)[:,1:]

#%% Transforming data

idvs = transformer(idvs)
idvs = numpy.nan_to_num(idvs)

gc.collect()

numpy.save(train_filename + suffix + ".npy", idvs)
numpy.save(train_filename  + suffix + "_dvs.npy", dvs)

#%%

for file_ in test_filenames:
    data = numpy.load(file_ + ".npy")
    if impute:
        data = imputer.transform(data)
    if scale:
        data = scaler.transform(data)[:,1:]
    data = transformer(data)
    data = numpy.nan_to_num(data)
    gc.collect()
    numpy.save(file_ + suffix + ".npy", data)
    