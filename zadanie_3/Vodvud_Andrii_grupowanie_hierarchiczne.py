
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_classification
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import confusion_matrix
from numpy import array
import numpy as np
from sklearn.metrics import adjusted_rand_score, jaccard_similarity_score


#Wygenerowanie zbioru danych na potrzeby klasyfikacji
mk_data=make_classification(n_samples=30, n_features=3, n_informative=3,
                         n_redundant=0, n_repeated=0, n_classes=4,
                         n_clusters_per_class=1, class_sep=3)
#dane uczace
data=mk_data[0]
#etykiety - na potrzeby wygenerowania macierzy konfuzji
labels=mk_data[1]

#single - miara najblizszego sasiada dmin
linkage_matrix_dmin = linkage(data, 'single')
print 'Macierz grupowania hierarchicznego, najblizszego sasiada'
print linkage_matrix_dmin

#complete - miara najdalszego sasiada dmax
linkage_matrix_dmax = linkage(data, 'complete')
print 'Macierz grupowania hierarchicznego, najdalszego sasiada'
print linkage_matrix_dmax

#wykres ilustrujacy dane w przestrzeni cech
plt.figure(1)
plt.title('Dane uczace')
plt.scatter(data[:,0],data[:,1], c=list(labels))
plt.show()

#Wygenerowanie dendrogramu dla najblizszego sasiada
plt.figure(2)
plt.title("Dendrogram_dmin")
dendrogram(linkage_matrix_dmin)
plt.show()

#Wygenerowanie dendrogramu dla najdalszego sasiada
plt.figure(2)
plt.title("Dendrogram_dmax")
dendrogram(linkage_matrix_dmax)
plt.show()

#Odciecie podzialu na poziomie 4 grup
y_dmin = fcluster(linkage_matrix_dmin, 4, 'maxclust')
print 'Podzial na cztery grupy dla najblizszego sasiada'
print y_dmin

#Macierz konfuzji - porownanie uzyskanego podzialu z etykietami
print confusion_matrix(labels,y_dmin-1)

#Odciecie podzialu na poziomie 4 grup
y_dmax = fcluster(linkage_matrix_dmax, 4, 'maxclust')
print 'Podzial na cztery grupy dla najdalszego sasiada'
print y_dmax

#Macierz konfuzji - porownanie uzyskanego podzialu z etykietami
print confusion_matrix(labels,y_dmax-1)

rand_ind=[]
jacc_ind=[]
#Wartosci indeksu Randa oraz Jaccarda
rand_ind.append(adjusted_rand_score(y_dmin,labels))
rand_ind.append(adjusted_rand_score(y_dmax,labels))
jacc_ind.append(jaccard_similarity_score(y_dmin, labels))
jacc_ind.append(jaccard_similarity_score(y_dmax, labels))

print 'Wartosci indeksu Randa oraz Jaccarda'

print rand_ind[0]
print jacc_ind[0]

print rand_ind[1]
print jacc_ind[1]

plt.title('wartosci indeksu Randa (czerwony) oraz Jaccarda (niebieski)')
plt.plot(rand_ind, c='r')
plt.plot(jacc_ind, c='b')
plt.show()

print 'najlepsza wartosc indeksu Randa'
print max(rand_ind)

print 'najlepsza wartosc indeksu Jaccarda'
print max(jacc_ind)