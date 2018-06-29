import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC


#generujemy dane dla problemu liniowo separowalnego
class1=np.random.multivariate_normal([3,3],[[1,0],[0,1]],75)
class2=np.random.multivariate_normal([-3,-3],[[1,0],[0,1]],75)
TrainData=np.vstack([class1[0:25,:],class2[0:25,:]])
TestData=np.vstack([class1[25:75,:],class2[25:75,:]])
TrainClasses=[0]*25+[1]*25
TestClasses=[0]*50+[1]*50

#wykres ilustrujacy zbior uczacy w przestrzeni cech
plt.scatter(TestData[0:50,0],TestData[0:50,1],c='b')
plt.scatter(TestData[50:100,0],TestData[50:100,1],c='r')
plt.show()

#Parametry liniowego klasyfikatora SVM
svm = SVC(kernel="linear", C=100)

#Uczenie klasyfikatora
svm.fit(TrainData,TrainClasses)

##przeprowadzenie ponownego procesu uczenia
data = []
for i in svm.support_:
    data.append(TrainClasses[i])
svm.fit(svm.support_vectors_, data)

#Obiekt nauczonego klasyfikatora przechowuje takie informacje jak:
print 'Numery probek bedace wektorami podpierajacymi'
print svm.support_
print 'Wektory zbioru uczacego bedace wektorami podpierajacymi'
print svm.support_vectors_
print 'Liczba wektrow podpierajacych w poszczegolnych klasach'
print svm.n_support_

#Optymalna hiperplaszczyzna rozdzialajaca
w = svm.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 7)
yy = a * xx - (svm.intercept_[0]) / w[1]

#Wyznaczamy proste rownolegle do OSH odlegle od niej o wartosc marginesu
#Na prostych tych znajduja sie wektory podpierajace
b = svm.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = svm.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])

#Narysowanie OSH i prostych rownoleglych
plt.plot(xx, yy, 'k-')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')
#Zaznaczenie na wykresie wektorow podpierajacych
plt.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
            s=80, facecolors='none')
plt.scatter(TrainData[:, 0], TrainData[:, 1], c=TrainClasses, cmap=plt.cm.Paired)

plt.axis('tight')
plt.show()

print 'Numery probek bedace wektorami podpierajacymi'
print svm.support_
print 'Wektory zbioru uczacego bedace wektorami podpierajacymi'
print svm.support_vectors_
print 'Liczba wektrow podpierajacych w poszczegolnych klasach'
print svm.n_support_

#Testowanie klasyfikatora na zbiorze uczacym
y = svm.predict(TestData)
print 'Etykiety klas dla zbioru testowego'
print y

#Sprawnosc klasyfikatora na zbiorze testowym
print 'Sprawnosc klasyfikatora:'
print svm.score(TestData,TestClasses)
