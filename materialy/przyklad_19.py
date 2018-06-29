
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_classification
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import confusion_matrix
from numpy import array
import numpy as np

#Wygenerowanie zbioru danych na potrzeby klasyfikacji
mk_data=make_classification(n_samples=20, n_features=2, n_informative=2,
                         n_redundant=0, n_repeated=0, n_classes=3,
                         n_clusters_per_class=1, class_sep=2)
#dane uczace
data=mk_data[0]
#etykiety - na potrzeby wygenerowania macierzy konfuzji
labels=mk_data[1]

#macierz ilustrujaca przebieg grupowania hierarchicznego
#single - miara najblizszego sasiada dmin
#complete - miara najdalszego sasiada dmax
linkage_matrix = linkage(data, 'complete')
print 'Macierz grupowania hierarchicznego'
print linkage_matrix

#wykres ilustrujacy dane w przestrzeni cech
plt.figure(1)
plt.title('Dane uczace')
plt.scatter(data[:,0],data[:,1], c=list(labels))
plt.show()

#Wygenerowanie dendrogramu
plt.figure(2)
plt.title("Dendrogram")
dendrogram(linkage_matrix)
plt.show()

#Odciecie podzialu na poziomie 3 grup
y = fcluster(linkage_matrix, 3, 'maxclust')
print 'Podzial na trzy grupy'
print y

#Macierz konfuzji - porownanie uzyskanego podzialu z etykietami
print confusion_matrix(labels,y-1)
