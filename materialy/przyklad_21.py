
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.datasets import make_classification
from sklearn.metrics import adjusted_rand_score, jaccard_similarity_score

#Wygenerowanie zbioru danych na potrzeby klasyfikacji
mk_data=make_classification(n_samples=200, n_features=3, n_informative=3,
                         n_redundant=0, n_repeated=0, n_classes=5,
                         n_clusters_per_class=1, class_sep=2.5)
#dane uczace
Data=mk_data[0]
#etykiety - na potrzeby wygenerowania macierzy konfuzji
Classes=mk_data[1]

#ilustracja zbioru danych za pomoca wykresu 3D
fig = plt.figure(1)
wykr = Axes3D(fig)
plt.title('Wygenerowany zbior danych')
wykr.scatter(Data[:, 0], Data[:, 1], Data[:, 2])
plt.show()

rand_ind=[]
jacc_ind=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i)
    #uczenie algorytmu kMeans
    kmeans.fit(Data)

    #uzyskany podzial
    podzial=kmeans.labels_

    #Wartosci indeksu Randa oraz Jaccarda
    rand_ind.append(adjusted_rand_score(podzial,Classes))
    jacc_ind.append(jaccard_similarity_score(podzial, Classes))

plt.title('wartosci indeksu Randa (czerwony) oraz Jaccarda (niebieski)')
plt.plot(range(1,11),rand_ind,c='r')
plt.plot(range(1,11),jacc_ind,c='b')
plt.show()
