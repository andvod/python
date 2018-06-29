
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.metrics import confusion_matrix
import sklearn

#wczytanie zbioru iris
iris = datasets.load_iris()
#macierz danych oraz znane etykiety klasowe
Data = iris.data
Classes = iris.target

#parametr algorytmu kMeans - podzial na 3 grupy
kmeans=KMeans(n_clusters=3)

#uczenie algorytmu kMeans
kmeans.fit(Data)

#uzyskany podzial
print 'Uzyskane etykiety podzialu dla zbioru iris:'
podzial=kmeans.labels_
print podzial

#ilustracja zbioru danych na wykresie 3d (trzy pierwsze atrybuty)
fig = plt.figure(1)
wykr = Axes3D(fig)
wykr.scatter(Data[:, 0], Data[:, 1], Data[:, 2], c=podzial.astype(np.float))
plt.show()

#Macierz konfuzji - porownanie uzyskanego podzialu z etykietami
print confusion_matrix(podzial,Classes)
