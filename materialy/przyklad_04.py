
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split

#importujemy zbior danych
iris = datasets.load_iris()

gnb = GaussianNB()

#Zbior obiektow
X = iris.data

#Wektor poprawnej klasyfikacji obiektow
y = iris.target

y=np.array(y)

#Dzielimy losowo zbior na dwie czesci
train, test, train_targets, test_targets = train_test_split(X, y,
                                 test_size=0.30, random_state=42)
print train

#Uczymy klasyfikator
clf = gnb.fit(train, train_targets)

#Testujemy
Z = clf.predict(test)

