#!/usr/bin/python
#-*-coding:utf-8-*

import unittest
import numpy as np
import hinge

class HingeTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def test_hinge_no_loss(self):
        data = np.array([[0., 10., -1.]])
        labels = np.array([[1]])
        weights = np.array([0.3, 0.3, 0.3])
        exp_loss = 0
        act_loss = hinge.hinge(weights, data, labels)
        self.assertEqual(act_loss, exp_loss, "Zero hinge loss")

    def test_hinge_loss(self):
        data = np.array([[0., 1., -1.]])
        labels = np.array([[1]])
        weights = np.array([0, 0, 1])
        exp_loss = 1
        act_loss = hinge.hinge(weights, data, labels)
        self.assertEqual(act_loss, exp_loss, "Negative hinge loss")

    def test_mean_hinge_loss(self):
        data = np.array([[0., 1., -1.], [0., -1., 1.]])
        labels = np.array([[1], [1]])
        weights = np.array([[0, 0, 1]])
        exp_loss = 0.5
        act_loss = hinge.hinge(weights, data, labels)
        self.assertEqual(act_loss, exp_loss, "Mean hinge loss")

    def test_hinge_grad_no_loss(self):
        data = np.array([[0., 10., -1.]])
        labels = np.array([[1]])
        weights = np.array([0.3, 0.3, 0.3])
        exp_loss = np.array([0, 0, 0])
        act_loss = hinge.hinge_grad(weights, data, labels)
        arr_equality = act_loss == exp_loss
        equality = min(arr_equality)
        self.assertTrue(equality, "Zero hinge loss grad")

    def test_hinge_grad_loss(self):
        data = np.array([[0., 1., -1.]])
        labels = np.array([1])
        weights = np.array([0, 0, 1])
        exp_grad = - data[0]
        act_grad = hinge.hinge_grad(weights, data, labels)
        arr_equality = act_grad == exp_grad
        equality = min(arr_equality)
        self.assertTrue(equality, "Negative hinge grad grad")

    def test_mean_hinge_grad_loss(self):
        data = np.array([[0., 1., -1.], [0., -1., 1.]])
        labels = np.array([1, 1])
        weights = np.array([[0, 0, 1]])
        exp_loss = - 0.5 * data[0]
        act_loss = hinge.hinge_grad(weights, data, labels)
        arr_equality = act_loss == exp_loss
        equality = min(arr_equality)
        self.assertTrue(equality, "Mean hinge loss grad")

if __name__ == "__main__":
    unittest.main()
