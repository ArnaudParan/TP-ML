#!/usr/bin/python
#-*-coding:utf-8-*

import numpy as np
from hinge import hinge, hinge_grad, ponderation, DATA, LABELS

class Perceptron(object):
    def __init__(self, max_iter=1000, eps=1e-3):
        self.max_iter = max_iter
        self.eps = eps
        self.weights = None
        self.hist_w = None
        self.hist_f = None

    def fit(self, data, labels, verbose=True):
        self.weights = np.random.random((1, data.shape[1]))
        self.hist_w = np.zeros((self.max_iter, data.shape[1]))
        self.hist_f = np.zeros((self.max_iter, 1))
        for train_id in range(self.max_iter):
            gradient = hinge_grad(self.weights, data, labels)
            self.weights = self.weights - self.eps * gradient
            self.hist_w[train_id] = self.weights
            self.hist_f[train_id] = hinge(self.weights, data, labels)
            if train_id % 100 == 0 and verbose:
                print train_id, self.hist_f[train_id]

    def predict(self, data):
        pondere = ponderation(self.weights, data)
        prediction = np.sign(pondere)
        return prediction

    def score(self, data, labels):
        prediction = self.predict(data)
        nb_succes = np.maximum(labels * prediction, 0)
        proportion_succes = np.mean(nb_succes)
        return proportion_succes

if __name__ == "__main__":
    print '\033[93m' + "\nPerceptron" + '\033[0m'
    PERCEP = Perceptron(max_iter=10000)
    PERCEP.fit(DATA, LABELS, verbose=True)
    print PERCEP.score(DATA, LABELS)
    print PERCEP.weights
