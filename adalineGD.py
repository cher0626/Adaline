import numpy as np

class AdalineGD:
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, x, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0, scale=0.1, size=1 + x.shape[1])
        #self.cost_ stores the cost of each iteration
        self.cost_ = []

        #update the weight once all data pass through once
        for i in range(self.n_iter):
            z = self.weighted_sum(x)
            errors = (y - z)
            #the shape of x is (#of sample, #of data of each sample)
            #the shape of errors is (#of sample, 1)
            #the shape of self.w_ is (#of data of each sample, 1)
            #so x cannot dot errors directly to output w_, x has to be in transpose
            #the shape of x.T is (#of data of each sample, #of sample)
            self.w_[1:] += self.eta * x.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum()/2
            self.cost_.append(cost)
        return self
    
    def weighted_sum(self, x):
        return np.dot(x, self.w_[1:]) + self.w_[0]

    def prediction(self, x):
        return np.where(self.weighted_sum(x) >= 0, 1, -1)