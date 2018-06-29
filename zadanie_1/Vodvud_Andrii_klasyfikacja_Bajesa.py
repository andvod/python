
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split

class Banana_type(object):
    """
    tworzenie typu danych jak na przykladzie
    """
    def __init__(self, data, target):
        self.data = data
        self.target = target

def rozpoznanie(name, atribut):
    """
    podzial na dane i imie klasy
    """
    banana = open(name, atribut)
    _data_0 = []
    _data_1 = []
    _target = []
    for line in banana:
        line = line.split()
        line = line[0].split(';')
        _data_0.append(float(line[0]))
        _data_1.append(float(line[1]))
        _target.append(int(line[2]))
    #umieszczany oba atrybuty w jednej macierzy
    data=np.vstack((_data_0, _data_1))
    #transpozycja - wartosci atrybutu reprezentujemy w kolumnie
    data=data.conj().transpose()
    banana = Banana_type(data, _target)
    return banana

#importujemy zbior danych
banana = rozpoznanie('banana.txt', 'r')

gnb = GaussianNB()

#Zbior obiektow
X = banana.data

#Wektor poprawnej klasyfikacji obiektow
y = banana.target

y=np.array(y)

#proba losowa z rozkladu jednostajnego na przedziale [0,1]
proba = np.random.rand(1)

#Dzielimy losowo zbior na dwie czesci
train, test, train_targets, test_targets = train_test_split(X, y,
                                 test_size=proba, random_state=42)   ## test_size procent danych testowych

#Uczymy klasyfikator
clf = gnb.fit(train, train_targets)

#Testujemy
Z = clf.predict(test)

h = .02

#Tworzymy kolorowa mape
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA']) ## korzystamy z dwoch cech
cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

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


