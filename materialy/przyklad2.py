
from sklearn import preprocessing
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

#liczba probek
N = 10

#proba losowa z rozkladu jednostajnego na przedziale [1,5]
x1 = 4*np.random.rand(N)+1

#proba losowa z rozkladu N(3,2)
x2 = 2*np.random.randn(N)+3

#umieszczany oba atrybuty w jednej macierzy
data=np.vstack((x1,x2))
#transpozycja - wartosci atrybutu reprezentujemy w kolumnie
data=data.conj().transpose()

#wartosci pierwszego atrybutu:
print data[:,0]

#wartosci pierwszej probki
print data[0,:]

#wykres ilustrujacy probki w przestrzeni cech
plt.scatter(data[:,0], data[:,1])
plt.show()

#wartosci srednie dla atrybutow
print 'wartosci srednie:'
print data.mean(0)
#odchylenia standardowe dla atrybutow
print 'odchylenia standardowe:'
print data.std(0)

#normalizacja
data_norm = preprocessing.scale(data)

print 'wartosci srednie i odchylenia po normalizacji:'
print data_norm.mean(0)
print data_norm.std(0)

#przedzial, na ktory dokonywane jest skalowanie
skalowanie = preprocessing.MinMaxScaler((0,1))
#operacja skalowania
data_skal = skalowanie.fit_transform(data)
print 'wartosci minimalne i maksymalne atrybutow po przeskalowaniu:'
print data_skal.min(0)
print data_skal.max(0)

#macierz odleglosci euklidesowych miedzy obiektami zbioru uczacego
odl_eukl=metrics.pairwise.pairwise_distances(data, metric='euclidean')
print 'Macierz odleglosci euklidesowych miedzy elementami zbioru uczacego'
print odl_eukl

#macierz odleglosci euklidesowych miedzy obiektami zbioru uczacego
print 'Macierz odleglosci Mahalanobisa miedzy elementami zbioru uczacego'
odl_mah=metrics.pairwise.pairwise_distances(data, metric='mahalanobis')
print odl_mah