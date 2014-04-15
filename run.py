# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 21:50:01 2014

@author: patanjali

#TODO: Truncate the training set to make it similar to the test set
#TODO: Add more idvs(location profiles)
#TODO: Use other modeling methods
#DONE: Split to Dev-val or use cross validation
#DONE: Cleanup genIDVs function and make it more modular
#TODO: Add logging

"""

#%%

import os, gc
import numpy, pandas
import fun
from sklearn.ensemble import GradientBoostingClassifier

workspace = 'G:/Kaggle/AllState'
trainfile = 'train.csv'
testfile = 'test_v2.csv'

DVs = ['A','B','C','D','E','F','G']

PROFILE_LIMIT = 0
DEV_LIMIT = 10000

#%%

os.chdir(workspace)

train_data,test_data = fun.loadData(trainfile,testfile)
actuals = fun.getDVs(train_data)
#fun.plotFrequencies(train_data)
#fun.minMaxFirstLastStats(train_idvdata,actuals)
#fun.twoWayFreqs(actuals)

#%%

train_idvdata = fun.getIDVData(train_data)
#train_idvdata_trunc = fun.truncate_series2(train_idvdata,pct=100)

#%%

mostRecent = fun.predictMostRecent(train_idvdata)
mostRecent['plan'] = fun.plan(mostRecent)

keys = ['state','location','risk_factor','C_previous']
"""
,'group_size','homeowner','car_age','car_value',
        ,'age_oldest','age_youngest','married_couple',,
        'duration_previous']
"""
for key in keys:
    train_idvdata = train_idvdata.merge(fun.genIDVs_profiles(train_idvdata,key,DVs),how='left',on=key).sort(('customer_ID','shopping_pt'))
    print key

idvs = fun.genIDVs_cust(train_idvdata)

idvs = idvs.fillna(method='backfill')
idvs = idvs.fillna(method='ffill')

predictions = pandas.DataFrame(mostRecent['A'])

model = {}

#%%

#from sklearn.decomposition import PCA

#pca = PCA(n_components = 200)

#idvs2 = idvs.T.drop_duplicates().T#pca.fit_transform(idvs)

#%%

predictions_train_raw = {}

for dv in DVs:
    model[dv] = GradientBoostingClassifier(verbose = 2, n_estimators = 100)
    model[dv].fit(idvs[PROFILE_LIMIT:DEV_LIMIT],actuals[dv][PROFILE_LIMIT:DEV_LIMIT])
    predictions[dv] = model[dv].predict_proba(idvs).argmax(1) + min(actuals[dv])
    predictions_train_raw[dv] = model[dv].predict_proba(idvs)
    print "Finished fitting ", dv
    print "Performance", sum(predictions[dv][PROFILE_LIMIT:DEV_LIMIT]==actuals[dv][PROFILE_LIMIT:DEV_LIMIT]), sum(idvs[dv+'_last'][PROFILE_LIMIT:DEV_LIMIT]==actuals[dv][PROFILE_LIMIT:DEV_LIMIT])
    print "Performance", sum(predictions[dv][DEV_LIMIT:]==actuals[dv][DEV_LIMIT:]), sum(idvs[dv+'_last'][DEV_LIMIT:]==actuals[dv][DEV_LIMIT:])
    print "Performance", sum(predictions[dv]==actuals[dv]), sum(idvs[dv+'_last']==actuals[dv])

#%%

predictions['plan'] = fun.plan(predictions)
actuals['plan'] = fun.plan(actuals)
print sum(mostRecent['plan']==actuals['plan'])/(actuals.shape[0]*1.0)
print sum(predictions['plan']==actuals['plan'])/(actuals.shape[0]*1.0)
#%%

for key in keys:
    test_data = test_data.merge(fun.genIDVs_profiles(train_idvdata,key,DVs),how='left',on=key)
    print key

test_idvs = fun.genIDVs_cust(test_data)
test_idvs = test_idvs.fillna(method='backfill')
test_idvs = test_idvs.fillna(method='ffill')
test_idvs = test_idvs[idvs.columns]

mostRecent_test = fun.predictMostRecent(test_data)

preds_test = pandas.DataFrame(mostRecent_test['A'])

#%%

preds_test_raw = {}

for dv in DVs:
    preds_test_raw[dv] = model[dv].predict_proba(test_idvs)
    preds_test[dv] = model[dv].predict_proba(test_idvs).argmax(1) + min(actuals[dv])

preds_test['plan'] = fun.plan(preds_test)
preds_test.to_csv('test10.csv',cols=['plan'])

#%%
"""
import cPickle

data4 = open('idvs.pkl','wb')

cPickle.dump(idvs,data4)

data4.close()
"""
#print fun.error(predictions,actuals), 