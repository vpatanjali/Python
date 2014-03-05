# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 22:45:16 2014

@author: patanjali
"""

workspace = 'G:/Kaggle/Default/'
suffix = '_none_impute'
datafilename = 'train_v2%s' %(suffix)
test_filenames = ['test_v2_part%s%s' %(i,suffix) for i in xrange(1,5)]
thresh = 0
test = True
eps = .3
inf = 1e20
corr_thresh = 0.15

#%%

import os, time, gc
import logging, logging.config
import math
import numpy
from sklearn.preprocessing import Binarizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn import metrics
import Models
from Models import InteractionFeatures, Model, Bounder, RemoveDuplicateCols, ReturnSame, f1, lad

from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

#%%

os.chdir(workspace)

logging.config.fileConfig('loggerConfig.properties')

logger = logging.getLogger('alllog')
logger.debug("Starting...")

binarizer = Binarizer(copy=True, threshold=thresh)

featureunion1 = FeatureUnion([
                              #('duplicater',ReturnSame()),
                              ('if+',InteractionFeatures(method = lambda x,y:(x+y), threshold = corr_thresh,subsample = 1,logger=logger)),
                              ('if-',InteractionFeatures(method = lambda x,y:(x-y), threshold = corr_thresh,subsample = 1,logger=logger)),
                              ('if*',InteractionFeatures(method = lambda x,y:(x*y), threshold = corr_thresh,subsample = 1,logger=logger)),
                              ('if/',InteractionFeatures(method = lambda x,y:(x/y), threshold = corr_thresh,subsample = 1,logger=logger)),
                              ('if|',InteractionFeatures(method = lambda x,y:(y/x), threshold = corr_thresh,subsample = 1,logger=logger))
                               ])
                             
pp_pipeline = Pipeline([
                        ('removedupes',RemoveDuplicateCols(logger=logger)),
                        ('featureextraction',featureunion1),
                        ('bounder',Bounder(inf,-inf))
                        ])

#%%

idvs_raw = numpy.load(datafilename + ".npy")

dvs = numpy.load(datafilename + "_dvs.npy")

dvs_binary = binarizer.transform(dvs).reshape((dvs.shape[0],))

idvs = pp_pipeline.fit_transform(idvs,dvs_binary)

logger.debug("Building models with %s idvs", idvs.shape[1])

#%% Loss models

#corrs = numpy.array([numpy.abs(numpy.corrcoef(dvs_binary.T,idvs[:,i])[0,1]) for i in xrange(idvs.shape[1])])
#corrs2 = numpy.array([numpy.abs(numpy.corrcoef(dvs_binary.T,idvs2[:,i])[0,1]) for i in xrange(idvs2.shape[1])])

#idvs3 = numpy.hstack((idvs[:,numpy.where(corrs>0.145)[0]],idvs2[:,numpy.where(corrs2>0.11)[0]],))

#print idvs3.shape

idvs = Bounder(inf,-inf).transform(idvs)

tuned_parameters = {'n_estimators' : [50],
                    'max_depth': [5], 
                    'verbose' : [2]}

clf2 = GridSearchCV(GradientBoostingClassifier(), tuned_parameters, 
                   cv=StratifiedKFold(dvs_binary,n_folds = 5), 
                   scoring=f1,verbose = 2)
clf2.fit(idvs, dvs_binary)

for s in clf2.grid_scores_:
    print s

#%% Loss given default models

idvs_lgd = numpy.hstack(idvs, clf.predict_proba(idvs))

ind = numpy.where(dvs_binary>0)

idvs_lgd = idvs_lgd[ind]
dvs_lgd = dvs[ind]

featureunion2 = FeatureUnion([('duplicater',ReturnSame()),
                              ('if+',InteractionFeatures(method = lambda x,y:(x+y), threshold = 0.5,subsample = 1,logger=logger)),
                              ('if-',InteractionFeatures(method = lambda x,y:(x-y), threshold = 0.5,subsample = 1,logger=logger)),
                              ('if*',InteractionFeatures(method = lambda x,y:(x*y), threshold = 0.5,subsample = 1,logger=logger)),
                              ('if/',InteractionFeatures(method = lambda x,y:(x/y), threshold = 0.5,subsample = 1,logger=logger)),
                              ('if|',InteractionFeatures(method = lambda x,y:(y/x), threshold = 0.5,subsample = 1,logger=logger))
                               ])
"""                             
pp_pipeline2 = Pipeline([('featureextraction',featureunion2),
                        ('bounder',Bounder(inf,-inf)),])

dev_idvs_lgd = pp_pipeline2.fit_transform(dev_idvs_lgd,dev_dvs_lgd)
val_idvs_lgd = pp_pipeline2.transform(val_idvs_lgd)

logger.debug("Starting linear model with a dataset of size %s", (val_idvs_lgd.shape))
"""
#%% Loss given default models

tuned_parameters_lgd = {'n_estimators' : [10,50,100],
                    'max_depth': [2,3,4,5], 
                    'verbose' : [2]}

clf_lgd = GridSearchCV(GradientBoostingRegressor(loss='lad'), 
                       tuned_parameters_lgd, 
                       cv=StratifiedKFold(dvs,n_folds = 5), 
                       scoring=lad,verbose = 2)
clf_lgd.fit(idvs, dvs)

for s in clf_lgd.grid_scores_:
    print s

#%%
"""
for j in xrange(10):
    figure()
    test = score_dev2[numpy.where(dev_dvs_lgd==j)[0]]
    print numpy.median(test)
    plot(numpy.histogram(test,101)[1][0:-1],numpy.histogram(test,101)[0])
"""    
#%% Ensemble on the previous 2
"""
dev_idvs_ensemble = numpy.hstack((score_dev0, score_dev1, score_dev2))
val_idvs_ensemble = numpy.hstack((score_val0, score_val1, score_val2))

#%%

ensembles = {}
for i in xrange(10):
    ensembles[i] = Model('GBR',n_estimators = 10*i+1,verbose = 2,subsample = 0.5,max_depth=1,logger = logger)
    ensembles[i].fit(dev_idvs_ensemble, dev_dvs1)
    score_dev_ensemble = ensembles[i].model.predict(dev_idvs_ensemble)#.reshape(dev_dvs1.shape)
    score_val_ensemble = ensembles[i].model.predict(val_idvs_ensemble)#.reshape(val_dvs1.shape)
    logger.debug("Ensemble model has dev %s and val %s" %(metrics.mean_absolute_error(score_dev_ensemble, dev_dvs1),metrics.mean_absolute_error(score_val_ensemble, val_dvs1)))
    score_dev_ensemble = bounder.transform(score_dev_ensemble)
    score_val_ensemble = bounder.transform(score_val_ensemble)
    logger.debug("Capped ensemble model has dev %s and val %s" %(metrics.mean_absolute_error(score_dev_ensemble, dev_dvs1),metrics.mean_absolute_error(score_val_ensemble, val_dvs1)))
    linear_score_dev[dev_ind] = score_dev2.reshape((score_dev2.shape[0],))#_ensemble#.reshape((score_dev_ensemble.shape[0],1))
    linear_score_val[val_ind] = score_val2.reshape((score_val2.shape[0],))#_ensemble#.reshape((score_val_ensemble.shape[0],1))
    logger.debug("Overall model has dev %s and val %s" %(metrics.mean_absolute_error(linear_score_dev, dev_dvs),metrics.mean_absolute_error(linear_score_val, val_dvs)))
"""
#%%
"""    
    if test == True:
        ind = 105472
        print "Writing output to file ", 'output_%s_%s_%s_%s_%s_%s_%s.csv' %(method, method2, suffix, i, eps, mae_dev, mae_val)
        output_file = open('output_roundcapped,%s_%s_%s_%s_%s_%s_%s_%s.csv' %(method, method2, suffix, i, l, eps, mae_dev, mae_val),'w')
        output_file.write('id,loss\n')
"""        
for file_ in test_filenames:
    #print file_
    idvs_ = numpy.load(file_ + ".npy")
    idvs__ = numpy.load(file_ + '_idvs_corr_gt_.14.npy')
    idvs_ = numpy.hstack((idvs_,idvs__))
    idvs_ = Bounder(inf,-inf).transform(idvs_)
    idvs___ = pp_pipeline2.transform(idvs_)[:,872:]
    #score = clf.predict_proba(idvs_)[:,1]
    numpy.save(file_+'_idvs_corr_gt_0.5.npy',idvs__)
"""
            idvs = numpy.hstack((idvs,featureunion1.transform(idvs)))
            idvs[numpy.where(idvs>inf)] = inf
            idvs[numpy.where(idvs<-1*inf)] = -1*inf
            score_binary = model.predict_proba(idvs)[:,1]
            idvs2 = idvs[numpy.where(score_binary>ks_score_thresh)[0],:]
            idvs2 = numpy.hstack((idvs2,featureunion2.transform(idvs2)))
            idvs2[numpy.where(idvs2>inf)] = inf
            idvs2[numpy.where(idvs2<-1*inf)] = -1*inf
            score_linear = numpy.zeros(score_binary.shape)
            score_linear[numpy.where(score_binary>ks_score_thresh)[0]] = model2.predict(idvs2)
            score_linear = (score_linear.round()+numpy.abs(score_linear.round()))/2
            for j in xrange(idvs.shape[0]):
                output_file.write(str(ind) + ',' + str(int(score_linear[j])) + '\n')
                ind += 1
        output_file.close()
        
        #del model
        gc.collect()
"""
#%%
"""
maxs = []
eps = 0.05

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
    
#665 [] [47 53 54 66]

#dev_idvs[:,0] = dev_idvs_all[:,522] - dev_idvs_all[:,521]
#dev_idvs[:,1] = dev_idvs_all[:,522] - dev_idvs_all[:,272]
#dev_idvs[:,2] = dev_idvs_all[:,713] / dev_idvs_all[:,665]
#dev_idvs[:,3] = dev_idvs_all[:,719] / dev_idvs_all[:,665]
#val_idvs[:,0] = val_idvs_all[:,522] - val_idvs_all[:,521]
#val_idvs[:,1] = val_idvs_all[:,522] - val_idvs_all[:,272]
#val_idvs[:,2] = val_idvs_all[:,713] / val_idvs_all[:,665]
#val_idvs[:,3] = val_idvs_all[:,719] / val_idvs_all[:,665]

a1 = (numpy.cov(x.T,z.T)[0,1]**2)
b1 = 2*(numpy.cov(x.T,z.T)[0,1])*(numpy.cov(y.T,z.T)[0,1])
c1 = (numpy.cov(y.T,z.T)[0,1]**2)
a2 = x.var()
b2 = -2*x.mean()*y.mean()
c2 = y.var()

a = a1*b2 - a2*b1
b = 2*(a1*c2-a2*c1)
c = b1*c2 - b2*c1

v1 = (-1*b - sqrt(b*b-4*a*c))/2/a

#dev_idvs = imputer.fit_transform(dev_idvs)
#val_idvs = imputer.transform(val_idvs)
#dev_idvs = numpy.nan_to_num(dev_idvs)
#val_idvs = numpy.nan_to_num(val_idvs)
"""