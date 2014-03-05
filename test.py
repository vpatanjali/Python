from sklearn.pipeline import Pipeline

class Test1:
    def __init__(self,):
        pass
    def transform(self,data):
        print "Transform Test1"
        return data
    def fit(self,data,data2=None,**params):
        print "Fit Test1"
        return None
    def fit_transform(self,data,data2=None,**params):
        print "fit_transform Test1"
        self.fit(data)
        return self.transform(data)

class Test2:
    def __init__(self,):
        pass
    def transform(self,data):
        print "Transform Test2"
        return data
    def fit(self,data,data2=None,**params):
        print "Fit Test2"
        return None
    def fit_transform(self,data,data2=None,**params):
        print "fit_transform Test2"
        self.fit(data)
        return self.transform(data)


test1 = Test1()
test2 = Test2()
pipe = Pipeline([('t1', test1), ('t2', test2)])

#print test1.fit(0)
#print test1.transform(0)
#print test1.fit_transform(0)

pipe.fit(0)
pipe.transform(0)
pipe.fit_transform(0)