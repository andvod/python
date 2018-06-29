
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.datasets import make_classification
from numpy import array

def diam(data, classes):
    """
    najmniejsza odleglosc miedzy dwuma klasami
    """
    d = pairwise_distances(data)
    d_min = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if (classes[i] != classes[j]):
                d_min.append(d[i][j])
    return min(d_min)

def dunn(data, classes):
    """
    najwiksza srednica
    """
    d = pairwise_distances(data)
    d_max = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if (classes[i] == classes[j]):
                d_max.append(d[i][j])
    return diam(data, classes) / max(d_max)

#wczytanie zbioru digits
digits = datasets.load_digits()
#macierz danych oraz znane etykiety klasowe
Data = digits.data
Classes = digits.target

kmeans_grups = []
podzial = []
for i in range(2, 21):
    #parametr algorytmu kMeans - podzial na grupy od 2 do 20
    kmeans=(KMeans(n_clusters=i))

    #uczenie algorytmu kMeans
    kmeans_grups.append(kmeans.fit(Data))

    #uzyskany podzial
    podzial.append(kmeans.labels_)

#Ocena podzialu zbioru digits dla grup od 2 do 20 za pomoca indexa Dunn
indexes = []
for i in range(len(kmeans_grups)):
    index = dunn(Data, podzial[i])
    indexes.append(index)
    print 'index dunna dla ', i+2, 'grup', index

#ilustracja zbioru danych na wykresie 3d dla najlepszego indexa dunna(trzy pierwsze atrybuty)
fig = plt.figure(1)
wykr = Axes3D(fig)
index = indexes.index(max(indexes))
print 'najlepszy index dunn', indexes[index]
print 'optymalna ilosc grup', index+2
wykr.scatter(Data[:, 0], Data[:, 1], Data[:, 2], c=podzial[index].astype(np.float))
plt.show()

#Macierz konfuzji - porownanie uzyskanego podzialu z etykietami
#print confusion_matrix(podzial,Classes)
print confusion_matrix(podzial[index],Classes)

print 'Index Dunna dla danych losowych'

#Wygenerowanie zbioru danych na potrzeby klasyfikacji
mk_data=make_classification(n_samples=30, n_features=3, n_informative=3,
                         n_redundant=0, n_repeated=0, n_classes=4,
                         n_clusters_per_class=1, class_sep=3)
#dane uczace
data=mk_data[0]
#etykiety - na potrzeby wygenerowania macierzy konfuzji
labels=mk_data[1]

kmeans_grups_losowe = []
podzial_losowe = []
for i in range(2, 21):
    #parametr algorytmu kMeans - podzial na grupy od 2 do 20
    kmeans=(KMeans(n_clusters=i))

    #uczenie algorytmu kMeans
    kmeans_grups_losowe.append(kmeans.fit(data))

    #uzyskany podzial
    podzial_losowe.append(kmeans.labels_)

#Ocena podzialu zbioru losowego dla grup od 2 do 20
indexes = []
for i in range(len(kmeans_grups_losowe)):
    index = dunn(data, podzial_losowe[i])
    indexes.append(index)
    print 'index dunna dla ', i+2, 'grup', index

#ilustracja zbioru danych na wykresie 3d dla najlepszego indexa dunna(trzy pierwsze atrybuty)
fig = plt.figure(1)
wykr = Axes3D(fig)
index = indexes.index(max(indexes))
print 'najlepszy index dunn', indexes[index]
print 'optymalna ilosc grup', index+2
wykr.scatter(data[:, 0], data[:, 1], data[:, 2], c=podzial_losowe[index].astype(np.float))
plt.show()

#Macierz konfuzji - porownanie uzyskanego podzialu z etykietami
print confusion_matrix(podzial_losowe[index],labels)