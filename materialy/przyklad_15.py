
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from matplotlib.colors import ListedColormap

xx, yy = np.meshgrid(np.linspace(-3, 3, 300),np.linspace(-3, 3, 300))
Data = np.random.randn(500, 2)
Classes = np.logical_xor(Data[:, 0] > 0, Data[:, 1] > 0)

TrainData, TestData, TrainClasses, TestClasses = train_test_split(Data, Classes, test_size=.5)

#Parametry przykladowych klasyfikatorow
klasyfikatory = [
    SVC(kernel="rbf", gamma=3, C=10),
    SVC(kernel="rbf", gamma=0.5, C=0.1),
    SVC(kernel="rbf", gamma=0.05, C=0.5),
    SVC(kernel="poly", degree=2, coef0=0, C=1),
    SVC(kernel="poly", degree=3, coef0=1, C=0.1)]

cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
#Iterujemy po klasyfikatorach
for svm in klasyfikatory:
    #Uczenie klasyfikatora
    svm.fit(TrainData, TrainClasses)
    #Sprawnosc
    score = svm.score(TestData, TestClasses)

    #Wykres powierzchni rozdzielajacych
    Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)

    #Wykres probek uczacych
    plt.scatter(TrainData[:, 0], TrainData[:, 1], c=TrainClasses, cmap=cm_bright)
    #Wykres probek testowych
    plt.scatter(TestData[:, 0], TestData[:, 1], c=TestClasses, cmap=cm_bright,alpha=0.6)
    #Tytul wykresu zawierajacy uzyskana sprawnosc i parametry klasyfikatora
    plt.title(str(svm) + ' Score='+str(score))
    plt.show()
