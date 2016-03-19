#!/usr/bin/python
#-*-coding:utf-8-*

from perceptron import Perceptron
from gen_data import gen_arti, plot_frontiere, plot_data
import matplotlib.pyplot as plt

### Generer et tracer des donnees
DATAX, DATAY = gen_arti(data_type=0, nbex=1000, eps=0.1)
PERCEP = Perceptron(eps=1e-1, max_iter=1000)
PERCEP.fit(DATAX, DATAY)
print PERCEP.score(DATAX, DATAY)
plot_frontiere(DATAX, PERCEP.predict, 50)
plot_data(DATAX, DATAY)
plt.show()
