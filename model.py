# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 22:45:16 2014

@author: patanjali
"""

workspace = 'G:/Kaggle/Default/'
suffix = '_none_impute'
dev_filename = 'train_v2%s_dev' %(suffix)
val_filename = 'train_v2%s_val' %(suffix)
test_filenames = ['test_v2_part%s%s' %(i,suffix) for i in xrange(1,5)]
thresh = 0
test = True
eps = .3
inf = 1e20
#%%

import os, time, gc
import math
import numpy

from sklearn.preprocessing import Binarizer

from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import Imputer

from sklearn import metrics

from xtravars import diffs,sums,products

#%%

types = {'RFC' : 'single',
         'RFR' : 'single',
         'GBR' : 'list', 
         'GBC' : 'list', 
         'ABR' : 'list', 
         'ABC' : 'list', 
         'Lasso' : 'single',
         'LR': 'single',
         'LSVC' : 'single',
         'SVC' : 'single'
         }

methods = {'RFC' : RandomForestClassifier, 
           'RFR' : RandomForestRegressor, 
           'GBR' : GradientBoostingRegressor, 
           'GBC' : GradientBoostingClassifier, 
           'ABR': AdaBoostRegressor,
           'ABC': AdaBoostClassifier,
           'Lasso' : Lasso,
           'LR': LogisticRegression,
           'LSVC' : LinearSVC,
           'SVC' : SVC
           }

#%%
    
os.chdir(workspace)

dev_idvs_all = numpy.nan_to_num(numpy.load(dev_filename + ".npy"))
val_idvs_all = numpy.nan_to_num(numpy.load(val_filename + ".npy"))

dev_dvs = numpy.nan_to_num(numpy.load(dev_filename + "_dvs.npy"))
val_dvs = numpy.nan_to_num(numpy.load(val_filename + "_dvs.npy"))

binarizer = Binarizer(copy=True, threshold=thresh)
imputer = Imputer(copy = False)

dev_dvs_binary = binarizer.transform(dev_dvs).reshape((dev_dvs.shape[0],))
val_dvs_binary = binarizer.transform(val_dvs).reshape((val_dvs.shape[0],))

"""
from statsmodels.regression import quantile_regression

dev_idvs2 = dev_idvs[:10000,:]
inds = [i for i in xrange(dev_idvs2.shape[1]) if len(unique(dev_idvs2[:,i])) > 1]
dev_dvs2 = dev_dvs[:10000,:].reshape((10000,))

model = quantile_regression.QuantReg(dev_dvs2, dev_idvs2)
model.fit()
"""

"""
#plot(mae_dev)
print "Finished loading data"
"""
#%% manual feature selection

dev_idvs = numpy.zeros((dev_idvs_all.shape[0],4))
val_idvs = numpy.zeros((val_idvs_all.shape[0],4))

#665 [] [47 53 54 66]

dev_idvs[:,0] = dev_idvs_all[:,522] - dev_idvs_all[:,521]
dev_idvs[:,1] = dev_idvs_all[:,522] - dev_idvs_all[:,272]
dev_idvs[:,2] = dev_idvs_all[:,713] / dev_idvs_all[:,665]
dev_idvs[:,3] = dev_idvs_all[:,719] / dev_idvs_all[:,665]
val_idvs[:,0] = val_idvs_all[:,522] - val_idvs_all[:,521]
val_idvs[:,1] = val_idvs_all[:,522] - val_idvs_all[:,272]
val_idvs[:,2] = val_idvs_all[:,713] / val_idvs_all[:,665]
val_idvs[:,3] = val_idvs_all[:,719] / val_idvs_all[:,665]

dev_idvs = imputer.fit_transform(dev_idvs)
val_idvs = imputer.transform(val_idvs)
#dev_idvs = numpy.nan_to_num(dev_idvs)
#val_idvs = numpy.nan_to_num(val_idvs)

dev_idvs_all = numpy.nan_to_num(dev_idvs_all)
val_idvs_all = numpy.nan_to_num(val_idvs_all)

dev_idvs = numpy.hstack((dev_idvs_all, dev_idvs))
val_idvs = numpy.hstack((val_idvs_all, val_idvs))

dev_idvs[numpy.where(dev_idvs>inf)] = inf
val_idvs[numpy.where(val_idvs>inf)] = inf
dev_idvs[numpy.where(dev_idvs<-1*inf)] = -1*inf
val_idvs[numpy.where(val_idvs<-1*inf)] = -1*inf

#%%
i = 1
for i in xrange(2,10):
    start_time = time.time()
    params_dict = {'RFC' : {'n_estimators' : int(10*i), 'max_depth': 5},
                   'RFR' : {'n_estimators' : int(100-10*i), 'max_depth': 10},
                   'GBR' : {'subsample' : 0.7, 'n_estimators' : 1, 'loss': 'lad',
                           'max_depth': 1, 'verbose' : 1},
                   'GBC' : {'subsample' : 0.7, 'n_estimators' : 10*(i+1),
                           'max_depth': 3, 'verbose' : 1},
                   'ABR' : {'n_estimators' : 10*i, 'loss': 'linear'},
                   'ABC' : {'base_estimator': SVC(kernel = 'linear', 
                                                  max_iter = 10,
                                                  probability = True), 
                            'n_estimators' : i, 'random_state' : 1,
                            'algorithm' : 'SAMME'},
                   'Lasso' : {'alpha' : 1.0/(2**i)},
                   'LR' : {'penalty': 'l1',
                           'C' : 1, 'dual' : False},
                   'LSVC' : {'C' : 0.01*i, 'penalty': 'l1',
                             'dual' : False, 'verbose' : 0},
                   'SVC' : {'kernel' : 'linear', 'max_iter' : 1000*i, 
                            'probability' : True, 'verbose' : 2}
                   }
    
    method = 'GBC'
    modelingmethod = methods[method]
    typ = types[method]
    params = params_dict[method]
    
    print method, params
    model = modelingmethod(**params)
    model.fit(dev_idvs, dev_dvs_binary)
    print "Finished fitting the model"
    score_dev = model.predict_proba(dev_idvs)[:,1]
    score_val = model.predict_proba(val_idvs)[:,1]
    """
    score_dev = dev_idvs[:,522] - dev_idvs[:,272]
    score_val = val_idvs[:,522] - val_idvs[:,272]
    """
    zipper_dev = [(score_dev[j], dev_dvs[j]) for j in xrange(score_dev.shape[0])]
    zipper_val = [(score_val[j], val_dvs[j]) for j in xrange(score_val.shape[0])]
    
    zippersorted_dev = sorted(zipper_dev, key = lambda x:x[0])
    zippersorted_val = sorted(zipper_val, key = lambda x:x[0])
    
    x_dev = numpy.array([x[0] for x in zippersorted_dev])
    y_dev = numpy.array([x[1] for x in zippersorted_dev])
    x_val = numpy.array([x[0] for x in zippersorted_val])
    y_val = numpy.array([x[1] for x in zippersorted_val])
    yrev_dev = y_dev[::-1]
    yrev_val = y_val[::-1]
    o_dev = numpy.cumsum(y_dev)
    o_val = numpy.cumsum(y_val)
    
    x = binarizer.transform(y_dev)
    y = 1-x
    xx = numpy.cumsum(x)/sum(x)
    yy = numpy.cumsum(y)/sum(y)
    ks = yy - xx
    plot(xx)
    plot(yy)
    ks_score_thresh = x_dev[numpy.where(ks==max(ks))[0][0]]
    x2 = binarizer.transform(y_val)
    y2 = 1-x2
    xx2 = numpy.cumsum(x2)/sum(x2)
    yy2 = numpy.cumsum(y2)/sum(y2)
    ks2 = yy2 - xx2
    
    print "Model has a dev KS %s, val KS %s and val KS at dev_cutoff %s" %(max(yy-xx)*100, max(yy2-xx2)*100, ks2[numpy.where(x_val<ks_score_thresh)[0].max()]*100)
    # 
    dev_idvs_ks = dev_idvs[numpy.where(score_dev>ks_score_thresh)[0],:]
    dev_dvs_ks = dev_dvs[numpy.where(score_dev>ks_score_thresh)[0],:]
    dev_dvs_ks = dev_dvs_ks.reshape((dev_dvs_ks.shape[0],))
    val_idvs_ks = val_idvs[numpy.where(score_val>ks_score_thresh)[0],:]
    val_dvs_ks = val_dvs[numpy.where(score_val>ks_score_thresh)[0],:]
    val_dvs_ks = val_dvs_ks.reshape((val_dvs_ks.shape[0],))
 
    #%% Picking most preditive variables for the residue...
    """
    corrs = numpy.array([numpy.abs(numpy.corrcoef(dev_idvs_ks2[:,j].T,dev_dvs_ks.T)[0,1]) for j in xrange(dev_idvs_ks2.shape[1])])
    dev_idvs_ks = dev_idvs_ks2[:,numpy.where((corrs>eps))[0]]
    val_idvs_ks = val_idvs_ks2[:,numpy.where((corrs>eps))[0]]
    print dev_idvs_ks.shape
    """
    src = [dev_idvs_ks, val_idvs_ks]
    dst = src
    
    for ds in xrange(len(src)):
        for id1 in diffs:
            for id2 in diffs[id1]:
                dst[ds] = numpy.hstack((dst[ds],(src[ds][:,id1+id2+1] - src[ds][:,id1]).reshape(src[ds].shape[0],1)))
    
    for ds in xrange(len(src)):
        for id1 in sums:
            for id2 in sums[id1]:
                dst[ds] = numpy.hstack((dst[ds],(src[ds][:,id1+id2+1] + src[ds][:,id1]).reshape(src[ds].shape[0],1)))
    
    for ds in xrange(len(src)):
        for id1 in products:
            for id2 in products[id1]:
                dst[ds] = numpy.hstack((dst[ds],(src[ds][:,id1+id2+1] + src[ds][:,id1]).reshape(src[ds].shape[0],1)))
    
    dev_idvs_ks, val_idvs_ks = dst
    #dev_idvs_ks = numpy.hstack((dev_idvs_ks,dev_idvs_ks2))
    #val_idvs_ks = numpy.hstack((val_idvs_ks,val_idvs_ks2))
    
    maxs = numpy.array([numpy.abs(dev_idvs_ks[:,j]).max() for j in xrange(dev_idvs_ks.shape[1])])
    dev_idvs_ks = dev_idvs_ks[:,numpy.where((maxs<inf))[0]]
    val_idvs_ks = val_idvs_ks[:,numpy.where((maxs<inf))[0]]
    """
    maxs = []
    eps = 0.35
    for i in xrange(dev_idvs_ks2.shape[1]):
        dev_diff = (dev_idvs_ks2*dev_idvs_ks2[:,i].reshape(dev_idvs_ks2.shape[0],1)).T
        corrs = numpy.array([numpy.corrcoef(dev_diff[j,:],dev_dvs_ks.T)[0,1] for j in xrange(i+1,dev_idvs_ks2.shape[1])])
        if numpy.where(corrs > eps)[0].shape[0] > 0 or  numpy.where(corrs < -1*eps)[0].shape[0] > 0:
            print i, numpy.where(corrs > eps)[0], numpy.where(corrs < -1*eps)[0], corrs[numpy.where(corrs > eps)[0]], corrs[numpy.where(corrs < -1*eps)[0]]
            maxs.append(i)
    print maxs
    """
    #%% Linear dv prediction model post KS point, 0 below that.
       
    for l in xrange(7,8):
        start_time = time.time()
        params_dict = {'RFC' : {'n_estimators' : int(10*l), 'max_depth': 5},
                       'RFR' : {'n_estimators' : int(100-10*l), 'max_depth': 10},
                       'GBR' : {'subsample' : 0.7, 'n_estimators' : 10*(l+1), 'loss': 'lad',
                               'max_depth': 3, 'verbose' : 1},
                       'GBC' : {'subsample' : 0.7, 'n_estimators' : 10*l,
                               'max_depth': 3, 'verbose' : 1},
                       'ABR' : {'n_estimators' : 10*l, 'loss': 'linear'},
                       'ABC' : {'base_estimator': SVC(kernel = 'linear', 
                                                      max_iter = 10,
                                                      probability = True), 
                                'n_estimators' : l, 'random_state' : 1,
                                'algorithm' : 'SAMME'},
                       'Lasso' : {'alpha' : 1.0/(2**l)},
                       'LR' : {'penalty': 'l1',
                               'C' : 10**l, 'dual' : False},
                       'LSVC' : {'C' : 0.01*l, 'penalty': 'l1',
                                 'dual' : False, 'verbose' : 0},
                       'SVC' : {'kernel' : 'linear', 'max_iter' : 1000*l, 
                                'probability' : True, 'verbose' : 2}
                       }
        
        method2 = 'GBR'
        modelingmethod = methods[method2]
        typ = types[method2]
        params = params_dict[method2]
        model2 = modelingmethod(**params)
        res = model2.fit(dev_idvs_ks, dev_dvs_ks)
        score_dev_linear = numpy.zeros(score_dev.shape)
        score_dev_linear[numpy.where(score_dev>ks_score_thresh)[0]] = model2.predict(dev_idvs_ks)
        mae_dev = metrics.mean_absolute_error(score_dev_linear, dev_dvs)
        
        score_val_linear = numpy.zeros(score_val.shape)
        score_val_linear[numpy.where(score_val>ks_score_thresh)[0]] = model2.predict(val_idvs_ks)
        mae_val = metrics.mean_absolute_error(score_val_linear, val_dvs)
        print "Performance in dev and val at i = %s and l = %s are %s, %s " %(i, l, mae_dev, mae_val)
        
        """
        # Median replacement method
        arr = []
        cutoffs = []
        
        for j in xrange(1,100):
            j_dev = numpy.cumsum(abs(yrev_dev-j))[::-1]
            j_val = numpy.cumsum(abs(yrev_val-j))[::-1]
            mae_dev = (o_dev + j_dev)/o_dev.shape[0]
            mae_val = (o_val + j_val)/o_val.shape[0]
            arr.append(mae_val.min())
            cutoffs.append(x_val[numpy.where(mae_val==mae_val.min())[0][0]])
            
        min_ = min(arr)
        min_ind = arr.index(min(arr))
        cutoff_min = cutoffs[min_ind]
        cutoff_max = -10790
        
        predictions_dev = numpy.zeros(score_dev.shape)
        predictions_dev[numpy.where(score_dev>=cutoff_min)[0]] = min_ind+1
        mae_dev = metrics.mean_absolute_error(predictions_dev, dev_dvs)
        
        print "Dev performance, ", mae_dev
            
        predictions_val = numpy.zeros(score_val.shape)
        predictions_val[numpy.where(score_val>=cutoff_min)[0]] = min_ind+1
        mae_val = metrics.mean_absolute_error(predictions_val, val_dvs)
        print "Val performance, ", mae_val
        print "Time taken %s", time.time() - start_time
    
        """
        
        #%%    
        if test == True:
            ind = 105472
            print "Writing output to file ", 'output_%s_%s_%s_%s_%s_%s_%s.csv' %(method, method2, suffix, i, eps, mae_dev, mae_val)
            output_file = open('output_%s_%s_%s_%s_%s_%s_%s_%s.csv' %(method, method2, suffix, i, l, eps, mae_dev, mae_val),'w')
            output_file.write('id,loss\n')
            
            for file_ in test_filenames:
                #print file_
                idvs = numpy.load(file_ + ".npy")
                # Median replacement method
                idvs2 = numpy.zeros((idvs.shape[0],4))
                idvs2[:,0] = idvs[:,522] - idvs[:,521]
                idvs2[:,1] = idvs[:,522] - idvs[:,272]
                idvs2[:,2] = idvs[:,713] / idvs[:,665]
                idvs2[:,3] = idvs[:,719] / idvs[:,665]
                idvs2 = imputer.transform(idvs2)
                idvs2 = numpy.hstack((idvs,idvs2))
                idvs2 = numpy.nan_to_num(idvs2)
                idvs2[numpy.where(idvs2>inf)] = inf
                idvs2[numpy.where(idvs2<-1*inf)] = -1*inf
                score_binary = model.predict_proba(idvs2)[:,1]
                idvs4 = idvs2[numpy.where(score_binary>ks_score_thresh)[0],:]
                #idvs3 = idvs4[:,numpy.where((corrs>eps))[0]]
                src = [idvs4]
                dst = src
                for ds in xrange(len(src)):
                    for id1 in diffs:
                        for id2 in diffs[id1]:
                            dst[ds] = numpy.hstack((dst[ds],(src[ds][:,id1+id2+1] - src[ds][:,id1]).reshape(src[ds].shape[0],1)))
        
                for ds in xrange(len(src)):
                    for id1 in sums:
                        for id2 in sums[id1]:
                            dst[ds] = numpy.hstack((dst[ds],(src[ds][:,id1+id2+1] + src[ds][:,id1]).reshape(src[ds].shape[0],1)))
                
                for ds in xrange(len(src)):
                    for id1 in products:
                        for id2 in products[id1]:
                            dst[ds] = numpy.hstack((dst[ds],(src[ds][:,id1+id2+1] + src[ds][:,id1]).reshape(src[ds].shape[0],1)))
                idvs3 = dst[0]
                #idvs3 = numpy.hstack((idvs3,idvs4))
                idvs3 = idvs3[:,numpy.where((maxs<inf))[0]]
                idvs3[numpy.where(idvs3>inf)] = inf
                idvs3[numpy.where(idvs3<-1*inf)] = -1*inf
                score_linear = numpy.zeros(score_binary.shape)
                score_linear[numpy.where(score_binary>ks_score_thresh)[0]] = model2.predict(idvs3)
                for j in xrange(idvs.shape[0]):
                    output_file.write(str(ind) + ',' + str(int(score_linear[j])) + '\n')
                    ind += 1
            output_file.close()
            
            #del model
            gc.collect()

#%%
"""
maxs = []
eps = 0.05
for i in xrange(dev_idvs.shape[1]):
    dev_diff = (dev_idvs*dev_idvs[:,i].reshape(dev_idvs.shape[0],1)).T
    corrs = numpy.array([numpy.corrcoef(dev_diff[j,:],dev_dvs.T)[0,1] for j in xrange(i+1,dev_idvs.shape[1])])
    if numpy.where(corrs > eps)[0].shape[0] > 0 or  numpy.where(corrs < -1*eps)[0].shape[0] > 0:
        print i, numpy.where(corrs > eps)[0], numpy.where(corrs < -1*eps)[0]
"""
#%%
# div 665 [] [47 53 54 66]
"""

This is the sandbox ;)

all_min = 1
all_min_ind = 0
all_min_k = 0
for i in xrange(dev_dvs.shape)
corrs = [(i,numpy.corrcoef(dev_idvs[:,i].T,dev_dvs.T)[0,1]) for i in xrange(dev_idvs.shape[1])]

corrs = [(i,numpy.corrcoef(dev_idvs[:,i].T,dev_dvs.T)[0,1]) for i in xrange(dev_idvs.shape[1])]
corrs_binary = [(i,numpy.corrcoef(dev_idvs[:,i].T,dev_dvs_binary.T)[0,1]) for i in xrange(dev_idvs.shape[1])]

selectedvars = [x[0] for x in corrs if x[1] > 0.01]

dev_idvs2 = dev_idvs[:,selectedvars]
val_idvs2 = val_idvs[:,selectedvars]

for k in xrange(dev_idvs.shape[1]):

#%%
k = 463
score_dev = predictions_dev#dev_idvs[:,k]
score_val = predictions_val#val_idvs[:,k]

zipper_dev = [(score_dev[i], dev_dvs[i]) for i in xrange(score_dev.shape[0])]
zipper_val = [(score_val[i], val_dvs[i]) for i in xrange(score_val.shape[0])]

zippersorted_dev = sorted(zipper_dev, key = lambda x:x[0])
zippersorted_val = sorted(zipper_val, key = lambda x:x[0])

x_dev = numpy.array([i[0] for i in zippersorted_dev])
y_dev = numpy.array([i[1] for i in zippersorted_dev])
x_val = numpy.array([i[0] for i in zippersorted_val])
y_val = numpy.array([i[1] for i in zippersorted_val])
yrev_dev = y_dev[::-1]
yrev_val = y_val[::-1]
o_dev = numpy.cumsum(y_dev)
o_val = numpy.cumsum(y_val)

arr = []

for i in xrange(6,7):
    i_dev = numpy.cumsum(abs(yrev_dev-i))[::-1]
    i_val = numpy.cumsum(abs(yrev_val-i))[::-1]
    mae_dev = (o_dev + i_dev)/o_dev.shape[0]
    mae_val = (o_val + i_val)/o_val.shape[0]
    
    #ind = x_dev[numpy.where(mae_dev==mae_dev.min())][0]
    arr.append(mae_dev.min())
curr_min = min(arr)
curr_min_ind = arr.index(min(arr))

if(curr_min < all_min):
    print curr_min, all_min
    all_min = curr_min
    all_min_ind = curr_min_ind
    all_min_k = k

print k, all_min, all_min_ind, all_min_k
#%%

idv = dev_idvs[:,895].reshape((dev_dvs.shape[0],))
dv = dev_dvs

unsorted = zip(idv,dv)
sorted_ = sorted(unsorted, key=lambda x:x[0])
idv = numpy.array([x[0] for x in sorted_])
dv = numpy.array([x[1] for x in sorted_])

breaks = []
medians = []
median = 0
predictions = numpy.zeros(dv.shape)

for j in xrange(80000,idv.shape[0]):
    median_ = numpy.median(dv[j:])
    if median_ > median and median_ == int(median_):
        breaks.append(j)
        median = median_
        medians.append(median)
        print j, median
    predictions[j] = median
    if j % 1000 == 0:
        print j
print sum(abs(dv-predictions))/dv.shape[0]

breaks_idv = idv[breaks]
pred_val = numpy.zeros(val_dvs.shape)
idv2 = val_idvs[:,895]
for i, break_ in enumerate(breaks_idv):
    pred_val[numpy.where(idv2>=break_)[0]] = medians[i]

print sum(abs(pred_val-val_dvs))/val_dvs.shape[0]


breaks_idv = [3.314386962572522]
medians = [5]
if test == True:
    ind = 105472
    output_file = open('output_463_log_univariate_single_break.csv', 'w')
    output_file.write('id,loss\n')
    
    for file_ in test_filenames:
        print file_
        idvs = numpy.load(file_ + ".npy")
        predictions = numpy.zeros((idvs.shape[0],))
        idv2 = idvs[:,463]
        for i, break_ in enumerate(breaks_idv):
            predictions[numpy.where(idv2>=break_)[0]] = medians[i]
        for j in xrange(idvs.shape[0]):
            output_file.write(str(ind) + ',' + str(int(predictions[j])) + '\n')
            ind += 1
    output_file.close()
    
    #del model
    gc.collect()
"""