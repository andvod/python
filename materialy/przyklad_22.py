
import neurolab as nl
import numpy as np
import pylab as pl

#Zbior uczacy
train = [[0, 0], [0, 1], [1, 0], [1, 1]]
train_target = [[0], [0], [0], [1]]

#Zbior testowy
test = [[2, 1], [2,2], [-1,0], [-1,1]]
test_targets=[[1], [1], [0], [0]]

#Tworzymy siec z 2 wejsciami i 1 neuronem
net = nl.net.newp([[-1, 2],[0, 2]], 1)

#Uczymy siec
error = net.train(train, train_target, epochs=100, show=10, lr=0.1)

#Symulujemy
out = net.sim(test)

#Obliczamy blad
f = nl.error.MSE()
test_error = f(test_targets, out)

print "Blad klasyfikacji: %f" %test_error