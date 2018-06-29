
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.lda import LDA
from sklearn.cross_validation import train_test_split
from sklearn import neighbors, datasets
import numpy as np

#importujemy zbior danych
breast_cancer = datasets.load_breast_cancer()

X = breast_cancer.data       #macierz obiektor
y = breast_cancer.target     #wektor ich poprawnej klasyfikacji
target_names = breast_cancer.target_names    #nazwa klasy obiektu

#Uruchamiamy algorytm PCA dla 5 komponentow
pca_parent = PCA(n_components=5)
X_r_parent = pca_parent.fit(X).transform(X)

#Uruchamiamy algorytm PCA dla 2 komponentow
pca = PCA(n_components=2)
X_r = pca.fit(X_r_parent).transform(X_r_parent)

#Procent wariancji wyjasnianej przez metode dla kazdego komponentu
print('Wspolczynnik wyjasnianej wariancji dla 5 komponentow:\n %s'
      % str(pca_parent.explained_variance_ratio_))

#Tworzymy wykres
plt.figure()
for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c,
                label=target_name)
plt.legend()
plt.show()

## dla 7NN
print "testujemy 7NN"
k = 7
for i in range(1, 6):
    pca = PCA(n_components=i) ##podajemy ilosc komponentow
    X_r = pca.fit(X).transform(X)

    #Dzielimy losowo zbior na dwie czesci
    train, test, train_targets, test_targets = train_test_split(X, y,
                                     test_size=0.30, random_state=42)

    #Tworzymy klasyfikator.
    #Jako metryke przyjmujemy metryke euklidesowa.
    clf = neighbors.KNeighborsClassifier(k, weights='uniform',
                                         metric='euclidean')

    #Uczymy klasyfikator
    clf.fit(train, train_targets)

    #Sprawdzamy sprawnosc klasyfikatora
    print "Sprawnosc klasyfikatora: %f" % clf.score(test,test_targets)

## dla 15NN
print "testujemy 15NN"
k = 15
for i in range(1, 6):
    pca = PCA(n_components=i) ##podajemy ilosc komponentow
    X_r = pca.fit(X).transform(X)

    #Dzielimy losowo zbior na dwie czesci
    train, test, train_targets, test_targets = train_test_split(X, y,
                                     test_size=0.30, random_state=42)

    #Tworzymy klasyfikator.
    #Jako metryke przyjmujemy metryke euklidesowa.
    clf = neighbors.KNeighborsClassifier(k, weights='uniform',
                                         metric='euclidean')

    #Uczymy klasyfikator
    clf.fit(train, train_targets)

    #Sprawdzamy sprawnosc klasyfikatora
    print "Sprawnosc klasyfikatora: %f" % clf.score(test,test_targets)

