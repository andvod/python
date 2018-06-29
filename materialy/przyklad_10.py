

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.lda import LDA

#importujemy zbior danych
iris = datasets.load_iris()

X = iris.data       #macierz obiektor
y = iris.target     #wektor ich poprawnej klasyfikacji
target_names = iris.target_names    #nazwa klasy obiektu
print y


#Uruchamiamy algorytm PCA dla 2 komponentow
pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)
print pca

#Procent wariancji wyjasnianej przez metode dla kazdego komponentu
print('Wspolczynnik wyjasnianej wariancji dla 2 komponentow: %s'
      % str(pca.explained_variance_ratio_))

#Tworzymy wykres
plt.figure()
for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c,
                label=target_name)
plt.legend()
plt.show()