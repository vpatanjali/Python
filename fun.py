# -*- coding: utf-8 -*-
"""
Created on Thu Apr 03 05:39:04 2014

@author: patanjali
"""

import pandas, matplotlib.pyplot, numpy

def nuniq(x): 
    return numpy.unique(x).shape[0]
def last2(x):
    return x[x.index[-2]]
def last2eq(x):
    return (x[x.index[-1]]==x[x.index[-2]])*1

def loadData(trainfile, testfile):
    train_data = pandas.read_csv(trainfile)
    test_data = pandas.read_csv(testfile)
    return train_data,test_data

def plotFrequencies(data):
    for colname in data.columns[1:]:
        print colname        
        df = data[colname].value_counts().sort_index()
        matplotlib.pyplot.figure()
        df.plot(kind='bar',title=colname)
        
def predictMostRecent(data):
    most_recent = data.groupby('customer_ID').last()[['A','B','C','D','E','F','G']]
    most_recent['customer_ID'] = most_recent.index
    return most_recent
    
def getIDVData(data):
    return data[data['record_type']==0]
    
def truncate_series(data,pct=30):
    data['truncation_rand'] = numpy.random.randint(0,100,data.shape[0])
    groups = data.groupby('customer_ID')
    trunc = data[:0]
    values = []
    for key,value in groups:
        size = value.shape[0]
        if size <= 2:
            values.append(value)
        else:
            for t in xrange(size-2):
                if value['truncation_rand'][value.index[0]+size-t-1] < pct:
                    value = value[:-1]
                    if t == size-2-1:
                        values.append(value)
                else:
                    values.append(value)
                    break
    trunc = pandas.concat(values,axis=0)
    return trunc

def truncate_series2(data,pct=60):
    groups = data.groupby('customer_ID')
    values = []
    for key,value in groups:
        size = value.shape[0]
        if size <= 2 or numpy.random.randint(0,100)>=pct:
            values.append(value)
        else:
            value = value[:int(numpy.ceil(numpy.random.rand()*(size-1)))+1]
            values.append(value)
    trunc = pandas.concat(values,axis=0)
    return trunc

def genIDVs_cust(data):
    
    #Binarizing categorical variables
    categorical = ['day','state','car_value','risk_factor','A','B','C','D','E','F','G']
    for var in categorical:
        print 'Binarizing ', var
        data = pandas.concat([data,pandas.get_dummies(data[var],prefix=var)],axis=1)
    
    #Extracting time
    data['hour'] = data['time'].apply(lambda x : x.split(':')[0]).astype('int64')
    data['minute'] = data['time'].apply(lambda x : x.split(':')[1]).astype('int64')    
    
    #Picking only the numeric variables
    collist = data.columns[((data.dtypes=='int64') + (data.dtypes=='float64'))]
    
    groups = data[collist].groupby('customer_ID')
    
    idvs = groups.aggregate(['first','last','min','max','mean','median','sum','std',])
    #                         nuniq,last2,last2eq])
    idvs.columns = [x[0]+'_'+x[1] for x in idvs.columns]
    return idvs
    
def genIDVs_profiles(data,key,dvs):
    orig_cols = data.shape[1]
    data = pandas.concat([data]+[pandas.get_dummies(data[dv],prefix=dv) for dv in dvs],axis=1)
    cols = data.columns[orig_cols:]

    groups = data.groupby(key)
    idvs = groups.mean()[cols]
    idvs.columns = [key+'_'+col for col in idvs.columns]
    idvs[key] = idvs.index
    return idvs
    
def getDVs(data):
    return data[data['record_type']==1][['customer_ID','A','B','C','D','E','F','G']]
    
def error(predicted,actual):
    #predicted = predicted.sort('customer_ID')
    #actual = actual.sort('customer_ID')
    actual[['A','B','C','D','E','F','G']] = actual[['A','B','C','D','E','F','G']].astype('str')
    predicted[['A','B','C','D','E','F','G']] = predicted[['A','B','C','D','E','F','G']].astype('str')
    actual['plan'] = plan(actual)
    predicted['plan'] = plan(predicted)
    return sum(predicted['plan'] == actual['plan'])/(predicted.shape[0]*1.0)

def plan(data):
    return data['A'].astype('str') + data['B'].astype('str') + data['C'].astype('str') + data['D'].astype('str') + data['E'].astype('str') + data['F'].astype('str') + data['G'].astype('str')    
    
def minMaxFirstLastStats(idvs,dvs):
    for col in ['A','B','C','D','E','F','G']:
        print col, sum(idvs.groupby('customer_ID')[col].first() == dvs[col]),\
                    sum(idvs.groupby('customer_ID')[col].min() == dvs[col]),\
                    sum(idvs.groupby('customer_ID')[col].max() == dvs[col]),\
                    sum(idvs.groupby('customer_ID')[col].last() == dvs[col])

def twoWayFreqs(dvs):
    l = ['A','B','C','D','E','F','G']
    for dv1 in xrange(len(l)):
        for dv2 in xrange(dv1+1,len(l)):
            print pandas.crosstab(dvs[l[dv1]], dvs[l[dv2]])
