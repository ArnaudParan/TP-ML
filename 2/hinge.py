#!/usr/bin/python
#-*-coding:utf-8-*
"""This module implements the calculation of hinge loss and hinge gradient"""

import numpy as np

def hinge(weight_matrix, data, labels, alpha=0):
    cout = vecteur_cout(weight_matrix, data, labels, alpha)
    cout_moyen = np.mean(cout)
    return cout_moyen

def hinge_grad(weight_matrix, data, labels, alpha=0):
    indicatrice = indicatrice_yfw_neg(weight_matrix, data, labels, alpha)
    t_labels = labels.reshape(indicatrice.shape)
    gradients = - indicatrice * t_labels * data
    if gradients.ndim == 2:
        gradient_moyen = np.average(gradients, axis=0)
    elif gradients.ndim == 1:
        gradient_moyen = gradients
    return gradient_moyen

def vecteur_cout(weight_matrix, data, labels, alpha=0):
    pondere = ponderation(weight_matrix, data) # wx
    t_labels = labels.reshape(pondere.shape)
    cout_interm = alpha - t_labels * pondere # a - ywx
    cout = np.maximum(0, cout_interm) # max(0, a - ywx)
    return cout

def indicatrice_yfw_neg(weight_matrix, data, labels, alpha=0):
    pondere = ponderation(weight_matrix, data)
    t_labels = labels.reshape(pondere.shape)
    cout_interm = alpha - t_labels * pondere
    cout = np.maximum(0, cout_interm)
    cout = np.sign(cout)
    return cout

def ponderation(weight_matrix, data):
    if data.ndim == 1:
        nb_col = data.shape[0]
    elif data.ndim == 2:
        nb_col = data.shape[1]
    vect = weight_matrix.reshape((nb_col, 1))
    pondere = data.dot(vect) # WxV
    return pondere

WEIGHTS = np.random.random((1, 3))
DATA = np.random.random((100, 3))
LABELS = np.random.randint(0, 2, size=(100, 1)) *2 - 1

if __name__ == "__main__":
    print '\033[93m' + "\nerreurs" + '\033[0m'
    print hinge(WEIGHTS, DATA, LABELS)
    print hinge(WEIGHTS, DATA[0], LABELS[0])
    print hinge(WEIGHTS, DATA[0, :], LABELS[0])

    print '\033[93m' + "\ngradients" + '\033[0m'
    print hinge_grad(WEIGHTS, DATA, LABELS)
    print hinge_grad(WEIGHTS, DATA[0], LABELS[0])
    print hinge_grad(WEIGHTS, DATA[0, :], LABELS[0])
