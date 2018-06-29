
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.cross_validation import train_test_split

k = 9

#importujemy zbior danych
iris = datasets.load_iris()

#Zbior obiektow
X = iris.data[:, :2]
# W niniejszym przykladzie interesuja nas jedynie pierwsze
#dwie cechy obiektow

#Wektor poprawnej klasyfikacji obiektow
y = iris.target

#Dzielimy losowo zbior na dwie czesci
train, test, train_targets, test_targets = train_test_split(X, y,
                                 test_size=0.30, random_state=42)

h = .02

#Tworzymy kolorowa mape
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

#Tworzymy klasyfikator.
#Jako metryke przyjmujemy metryke euklidesowa.
clf = neighbors.KNeighborsClassifier(k, weights='uniform',
                                     metric='euclidean')

#Uczymy klasyfikator
clf.fit(train, train_targets)

#Ustalamy wielkosc wykresu
#Dla kazdego obszaru decyzyjnego przypisany zostanie inny kolor
x_min, x_max = test[:, 0].min() - 1, test[:, 0].max() + 1
y_min, y_max = test[:, 1].min() - 1, test[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

#Testujemy klasyfikator
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

#Sprawdzamy sprawnosc klasyfikatora
print "Sprawnosc klasyfikatora: %f" % clf.score(test,test_targets)

#Umieszczamy wyniki na rysunku
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

#Umieszczamy elementy zbioru testowego na rysunku
plt.scatter(test[:, 0], test[:, 1], c=test_targets, cmap=cmap_bold)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()
