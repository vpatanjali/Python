import numpy

from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

class _logger():
    def debug(self,msg):
        print(msg)

class RemoveDuplicateCols():
    
    def __init__(self,logger = _logger()):
        self.duplicates = []
        self.logger = logger
        
    def fit(self,X,y=None,**params):
        for col1 in xrange(X.shape[1]):
            self.logger.info("Checking duplicates for column %s" %(col1))
            diff = numpy.abs(X - X[:,col1].reshape((X.shape[0],1)))
            dupes = numpy.where(diff.sum(0) == 0)[0]
            dupes = [i for i in dupes if i > col1+1]
            if len(dupes) > 0:
                self.logger.debug('%s has duplicates %s' %(col1, dupes))
                self.duplicates.extend(dupes)
    
    def transform(self,X):
        keepcols = numpy.array([i for i in xrange(X.shape[1]) if i not in self.duplicates])
        return X[:,keepcols]
        
    def fit_transform(self,X,y=None,**params):
        self.fit(X)
        return self.transform(X)

class ReturnSame():
    def __init__(self,**params):
        return None
    
    def fit(self,X,y=None,**params):
        return None
    
    def transform(self,X):
        return X
        
    def fit_transform(self,X,y=None,**params):
        return self.transform(X)
                
class InteractionFeatures():

    def __init__(self, method = None, threshold = 0.1, subsample = 0.1, logger = _logger()):
        self.pairs = {}
        self.method = method
        self.logger = logger
        self.threshold = threshold
        self.subsample = subsample
        
    def fit(self,X,y, **params):
        """
        Feature extraction method that looks at the threshold correlation of 
        pairwise diff/sum/div/prod of columns of X with y and returns features
        with corrcoeff > threshold
        """
        subsample = numpy.unique(numpy.random.randint(0,X.shape[0],int(X.shape[0]*self.subsample)))
        XX = X[subsample,:]
        yy = y[subsample,:]
        for i in xrange(X.shape[1]):
            self.logger.info("Checking %s of %s" %(i,XX.shape[1]))
            features = self.method(XX, XX[:,i].reshape(XX.shape[0],1))
            corr_coeffs = numpy.array([numpy.abs(numpy.corrcoef(features[:,j].T, yy.T)[0,1] ) 
                                for j in xrange(i+1,XX.shape[1])])
            if (numpy.where(corr_coeffs > self.threshold)[0].shape[0] > 0):
                self.pairs[i] = numpy.where(corr_coeffs > self.threshold)[0] + i + 1
                self.logger.debug(numpy.where(corr_coeffs > self.threshold)[0])
                self.logger.debug(corr_coeffs[numpy.where(corr_coeffs > self.threshold)[0]])
        self.logger.debug(self.pairs)
    
    def transform(self,X):
        output = numpy.zeros((X.shape[0],0))
        for col1 in self.pairs:
            for col2 in self.pairs[col1]:
                output = numpy.hstack((output,
                                       self.method(X[:,col2],
                                                   X[:,col1]).reshape((X.shape[0],1))))
        return output

    def fit_transform(self,X,y,**params):
        self.fit(X,y,**params)
        return self.transform(X)

class Bounder():
    
    def __init__(self,upper,lower,copy=False):
        self.upper = upper
        self.lower = lower
        self.copy = copy
    
    def fit(self,X,y,**params):
        return None
        
    def transform(self,X,**params):
        if self.copy:
            Y = X.copy()
            Y[numpy.where(Y>self.upper)] = self.upper
            Y[numpy.where(Y<self.lower)] = self.lower
            Y[numpy.isnan(Y)] = 0
            return Y
        else:
            X[numpy.where(X>self.upper)] = self.upper
            X[numpy.where(X<self.lower)] = self.lower
            X[numpy.isnan(X)] = 0
            return X
            
    def fit_transform(self,X,y=None,**params):
        return self.transform(X,**params)
        
class Model():
    _methods = {'RFC' : RandomForestClassifier, 
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
    _params_dict = {'RFC' : {'n_estimators' : 10, 'max_depth': 5},
                   'RFR' : {'n_estimators' : 10, 'max_depth': 5},
                   'GBR' : {'subsample' : 0.7, 'n_estimators' : 1, 'loss': 'lad',
                           'max_depth': 3, 'verbose' : 0},
                   'GBC' : {'subsample' : 0.7, 'n_estimators' : 10,
                           'max_depth': 3, 'verbose' : 0},
                   'ABR' : {'n_estimators' : 10, 'loss': 'linear'},
                   'ABC' : {'n_estimators' : 10, 'random_state' : 1,
                            'algorithm' : 'SAMME'},
                   'Lasso' : {'alpha' : 1.0},
                   'LR' : {'penalty': 'l1',
                           'C' : 1, 'dual' : False},
                   'LSVC' : {'C' : 0.01, 'penalty': 'l1',
                             'dual' : False, 'verbose' : 0},
                   'SVC' : {'kernel' : 'linear', 'max_iter' : 1000, 
                            'probability' : True, 'verbose' : 2}
                   }
    def __init__(self, method, logger = _logger(), **params):
        self.method = method
        self.params = self._params_dict[method].update(params)
        self.model = self._methods[method](**params)
        self.logger = logger
        
    def fit(self,X,y,**params):
        return self.model.fit(X,y,**params)
        
    def transform(self,X,**params):
        return self.model.transform(X,**params)
        
    def predict(self,X,**params):
        return self.model.predict(X)
    
    def predict_proba(self,X,**params):
        return self.model.predict_proba(X)
    
    def _ks(self,predictions,actuals,scoring=True):
        sortorder = predictions.argsort(0).reshape((predictions.shape[0],))
        predictions_sorted = predictions[sortorder]
        actuals_sorted = actuals[sortorder]
        _ks_array = numpy.cumsum(1-actuals_sorted)/sum(1-actuals_sorted) - \
                        numpy.cumsum(actuals_sorted)/sum(actuals_sorted)
        if not scoring:
            self._ks_array = _ks_array
            self.predictions_sorted = predictions_sorted
            self.ks_threshold = predictions_sorted[numpy.where(_ks_array==_ks_array.max())[0]]
        return _ks_array, predictions_sorted
        
    def ks(self,predictions,actuals,threshold = None,scoring = True,):
        if threshold is None:
            return self._ks(predictions,actuals,scoring)[0].max()
        else:
            _ks_array, predictions_sorted = self._ks(predictions,actuals,scoring)
            return _ks_array[numpy.where(predictions_sorted < threshold)[0].max()]

    def _f1(self,predictions,actuals,scoring=True):
        sortorder = predictions.argsort(0).reshape((predictions.shape[0],))[::-1]
        predictions_sorted = predictions[sortorder]
        actuals_sorted = actuals[sortorder]
        precision = numpy.cumsum(actuals_sorted)/numpy.cumsum(numpy.ones(actuals_sorted.shape))
        recall = numpy.cumsum(actuals_sorted)/numpy.sum(actuals_sorted)
        _f1_array = 2*precision*recall/(precision+recall)
        if not scoring:
            self._f1_array = _f1_array
            self.predictions_sorted = predictions_sorted
            self.f1_threshold = predictions_sorted[numpy.where(_f1_array==_f1_array.max())[0]]
        return _f1_array, predictions_sorted

    def f1(self,predictions,actuals,threshold = None,scoring = True,):
        if threshold is None:
            return self._f1(predictions,actuals,scoring)[0].max()
        else:
            _f1_array, predictions_sorted = self._f1(predictions,actuals,scoring)
            return _f1_array[numpy.where(predictions_sorted > threshold)[0].max()]

    def summary(self,p1,y1,p2,y2):
        self.logger.debug("Method %s fitted using settings %s" %(self.method, self._params_dict[self.method]))
        
        _ks_array,ps = self._ks(p1,y1,scoring=False)
        ks1 = _ks_array.max()
        ks_threshold = ps[numpy.where(_ks_array == ks1)[0]]
        ks2 = self.ks(p2,y2)
        ks3 = self.ks(p2,y2,ks_threshold)
        _f1_array,rps = self._f1(p1,y1,scoring=False)
        f11 = _f1_array.max()
        f1_threshold = rps[numpy.where(_f1_array == f11)[0]]
        f12 = self.f1(p2,y2)
        f13 = self.f1(p2,y2,f1_threshold)
        self.logger.debug("Model has KS dev, val, val @ thresh, thresh %.4f %.4f %.4f %.5f" %(ks1, ks2, ks3,ks_threshold))
        self.logger.debug("Model has F1 dev, val, val @ thresh, thresh %.4f %.4f %.4f %.5f" %(f11, f12, f13,f1_threshold))
        
def findMedianSplit(self,predictions,actuals,left_median=0):
    sortorder = predictions.argsort()
    predictions_sorted = predictions[sortorder]
    actuals_sorted = actuals[sortorder]
    medians = numpy.array([numpy.median(actuals_sorted[i:]) for i in xrange(actuals_sorted.shape[0])])
    index = numpy.where(medians>left_median)[0][0]
    return predictions_sorted[index], medians[index], index

def _f1(predictions,actuals):
    sortorder = predictions.argsort(0).reshape((predictions.shape[0],))[::-1]
    predictions_sorted = predictions[sortorder]
    actuals_sorted = actuals[sortorder]
    precision = numpy.cumsum(actuals_sorted)/numpy.cumsum(numpy.ones(actuals_sorted.shape))
    recall = numpy.cumsum(actuals_sorted)/numpy.sum(actuals_sorted)
    _f1_array = 2*precision*recall/(precision+recall)
    return _f1_array, predictions_sorted

def f1(scorer,X,actuals,predictions = None):
    if predictions is None:
        predictions = scorer.predict_proba(X)[:,1]
    _f1_array = _f1(predictions,actuals)[0]
    return _f1_array[numpy.isfinite(_f1_array)].max()

def lad(scorer,X,actuals):
    return numpy.mean(numpy.abs(scorer.predict(X)-actuals))