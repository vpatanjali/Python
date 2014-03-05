# -*- coding: utf-8 -*-
"""
Created on Mon Feb 03 19:53:25 2014

@author: patanjali
"""

"""

Beating the Benchmark :::::: Kaggle Loan Default Prediction Challenge.
__author__ : Abhishek

"""

import os

os.chdir('G:/Kaggle/Default/')

import pandas as pd
import numpy as np
import cPickle
from sklearn import preprocessing
from sklearn.svm import LinearSVC
import  scipy.stats as stats
import sklearn.linear_model as lm

#%%

def testdata(filename):
 X = pd.read_table(filename, sep=',', warn_bad_lines=True, error_bad_lines=True)

 X = np.asarray(X.values, dtype = float)

 col_mean = stats.nanmean(X,axis=0)
 inds = np.where(np.isnan(X))
 X[inds]=np.take(col_mean,inds[1])
 data = np.asarray(X[:,1:-3], dtype = float)

 return data
 
def data(filename):
 X = pd.read_table(filename, sep=',', warn_bad_lines=True, error_bad_lines=True)

 X = np.asarray(X.values, dtype = float)

 col_mean = stats.nanmean(X,axis=0)
 inds = np.where(np.isnan(X))
 X[inds]=np.take(col_mean,inds[1])

 labels = np.asarray(X[:,-1], dtype = float)
 data = np.asarray(X[:,1:-4], dtype = float)
 return data, labels


def createSub(clf, traindata, labels, testdata):
 sub = 1

 labels = np.asarray(map(int,labels))

 niter = 10
 auc_list = []
 mean_auc = 0.0; itr = 0
 if sub == 1:
  xtrain = traindata#[train]
  xtest = testdata#[test]

  ytrain = labels#[train]
  predsorig = np.asarray([0] * testdata.shape[0]) #np.copy(ytest)

  labelsP = []

  for i in range(len(labels)):
   if labels[i] > 0:
    labelsP.append(1)
   else:
    labelsP.append(0)

  labelsP = np.asarray(labelsP)
  ytrainP = labelsP

  lsvc = LinearSVC(C=0.01, penalty="l1", dual=False, verbose = 2)
  print xtrain.shape, ytrainP.shape
  lsvc.fit(xtrain, ytrainP)
  xtrainP = lsvc.transform(xtrain)
  xtestP =  lsvc.transform(xtest)
  print xtrain.shape, xtest.shape
  print xtrainP.shape, xtest.shape

  clf.fit(xtrainP,ytrainP)
  predsP = clf.predict(xtestP)
  preds_ = clf.predict(xtrainP)
  print sum(preds_), sum(ytrainP), sum(abs(preds_-ytrainP))

  nztrain = np.where(ytrainP > 0)[0]
  nztest = np.where(predsP == 1)[0]

  nztrain0 = np.where(ytrainP == 0)[0]
  nztest0 = np.where(predsP == 0)[0]

  xtrainP = xtrain[nztrain]
  xtestP = xtest[nztest]

  ytrain0 = ytrain[nztrain0]
  ytrain1 = ytrain[nztrain]

  clf.fit(xtrainP,ytrain1)
  preds = clf.predict(xtestP)

  predsorig[nztest] = preds
  predsorig[nztest0] = 0

  np.savetxt('predictions.csv',predsorig ,delimiter = ',', fmt = '%d')

#%%

if __name__ == '__main__':
 filename = 'train_v2_dev_10000.csv'
 X_test = testdata('test_v2_10000.csv')

 X, labels = data(filename)
 
 clf = lm.LogisticRegression(penalty='l2', dual=True, tol=0.0001, 
                             C=1.0, fit_intercept=True, intercept_scaling=1.0, 
                             class_weight=None, random_state=None)

 X = preprocessing.scale(X)
 X_test = preprocessing.scale(X_test)

 createSub(clf, X, labels, X_test)