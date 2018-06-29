
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_moons, make_circles

def gauss_0():
    return SVC(kernel="rbf", gamma=1e-1, C=0.1)

def gauss_1():
    return SVC(kernel="rbf", gamma="auto", C=0.1)

def gauss_2():
    return SVC(kernel="rbf", gamma="auto", C=1)

def poly_0():
    return SVC(kernel="poly", degree=3, coef0=0.0, C=0.1)

def poly_1():
    return SVC(kernel="poly", degree=90, coef0=0.2, C=0.1)

def poly_2():
    return SVC(kernel="poly", degree=90, coef0=0.2, C=1)

data1=make_moons(noise=0.1, random_state=0)
data2=make_circles(noise=0.1, random_state=0)

print 'liczba wspolrzednych i etykiet dla moons'
print len(data2[0]), len(data2[1])
print 'liczba wspolrzednych i etykiet dla circles'
print len(data2[0]), len(data2[1])

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

plt.scatter(data1[0][:,0],data1[0][:,1],c=data1[1], cmap=cmap_light)
plt.scatter(data2[0][:,0],data2[0][:,1],c=data2[1], cmap=cmap_bold)
plt.show()

#Podzial na czesc uczaca i testowa
TrainData_1, TestData_1, TrainClasses_1, TestClasses_1 = train_test_split(data1[0], data1[1], test_size=.5)
TrainData_2, TestData_2, TrainClasses_2, TestClasses_2 = train_test_split(data2[0], data2[1], test_size=.5)

#wartosc kroku
h=0.025
x_min_1, x_max_1 = data1[0][:, 0].min() - .5, data1[0][:, 0].max() + .5
y_min_1, y_max_1 = data1[0][:, 1].min() - .5, data1[0][:, 1].max() + .5
xx_1, yy_1 = np.meshgrid(np.arange(x_min_1, x_max_1, h), np.arange(y_min_1, y_max_1, h))
cm_bright_1 = ListedColormap(['#FF0000', '#0000FF'])
#dane uczace
plt.scatter(TrainData_1[:, 0], TrainData_1[:, 1], c=TrainClasses_1, cmap=cm_bright_1)
#dane testowe
plt.scatter(TestData_1[:, 0], TestData_1[:, 1], c=TestClasses_1, cmap=cm_bright_1, alpha=0.6)

x_min_2, x_max_2 = data2[0][:, 0].min() - .5, data2[0][:, 0].max() + .5
y_min_2, y_max_2 = data2[0][:, 1].min() - .5, data2[0][:, 1].max() + .5
xx_2, yy_2 = np.meshgrid(np.arange(x_min_2, x_max_2, h), np.arange(y_min_2, y_max_2, h))
cm_bright_2 = ListedColormap(['#FFAAAA', '#AAFFAA'])
#dane uczace
plt.scatter(TrainData_2[:, 0], TrainData_2[:, 1], c=TrainClasses_2, cmap=cm_bright_2)
#dane testowe
plt.scatter(TestData_2[:, 0], TestData_2[:, 1], c=TestClasses_2, cmap=cm_bright_2, alpha=0.6)
#wyswietlenie wykresu
plt.show()


scores = []

data1_gauss=SVC(kernel="rbf", gamma=1e-1, C=0.1)
data1_gauss.fit(TrainData_1,TrainClasses_1)
score = data1_gauss.score(TestData_1, TestClasses_1)
print 'Sprawnosc na zbiorze testowym 1'
print score
scores.append(score)

data1_gauss=SVC(kernel="rbf", gamma='auto', C=0.1)
data1_gauss.fit(TrainData_1,TrainClasses_1)
score = data1_gauss.score(TestData_1, TestClasses_1)
print 'Sprawnosc na zbiorze testowym 2'
print score
scores.append(score)

data1_gauss=SVC(kernel="rbf", gamma='auto', C=1)
data1_gauss.fit(TrainData_1,TrainClasses_1)
score = data1_gauss.score(TestData_1, TestClasses_1)
print 'Sprawnosc na zbiorze testowym 3'
print score
scores.append(score)

index = scores.index(max(scores))
if (index == 0):
    gauss = gauss_0()
elif (index == 1):
    gauss = gauss_1()
elif (index == 2):
    gauss = gauss_2()

gauss.fit(TrainData_1,TrainClasses_1)

#Generujemy wykres powierzchni rozdzielajacej
Z = gauss.decision_function(np.c_[xx_1.ravel(), yy_1.ravel()])
Z = Z.reshape(xx_1.shape)
plt.contourf(xx_1, yy_1, Z, cmap=cm_bright_1, alpha=.8)
#Wyswietlenie elementow zbioru uczacego
plt.scatter(TrainData_1[:, 0], TrainData_1[:, 1], c=TrainClasses_1, cmap=cm_bright_1)
#Wyswietlenie elementow zbioru testowego
plt.scatter(TestData_1[:, 0], TestData_1[:, 1], c=TestClasses_1, cmap=cm_bright_1,alpha=0.6)
plt.show()


scores_2 = []

data2_poly=SVC(kernel="poly", degree=3, coef0=0.0, C=0.1)
data2_poly.fit(TrainData_2,TrainClasses_2)
score = data2_poly.score(TestData_2, TestClasses_2)
print 'Sprawnosc na zbiorze testowym 1'
print score
scores_2.append(score)

data2_poly=SVC(kernel="poly", degree=90, coef0=0.2, C=0.1)
data2_poly.fit(TrainData_2,TrainClasses_2)
score = data2_poly.score(TestData_2, TestClasses_2)
print 'Sprawnosc na zbiorze testowym 2'
print score
scores_2.append(score)

data2_poly=SVC(kernel="poly", degree=90, coef0=0.2, C=1)
data2_poly.fit(TrainData_2,TrainClasses_2)
score = data2_poly.score(TestData_2, TestClasses_2)
print 'Sprawnosc na zbiorze testowym 3'
print score
scores_2.append(score)

index = scores_2.index(max(scores_2))
if (index == 0):
    poly = poly_0()
elif (index == 1):
    poly = poly_1()
elif (index == 2):
    poly = poly_2()

poly.fit(TrainData_2,TrainClasses_2)

Z = poly.decision_function(np.c_[xx_2.ravel(), yy_2.ravel()])
Z = Z.reshape(xx_2.shape)
plt.contourf(xx_2, yy_2, Z, cmap=cm_bright_2, alpha=.8)
plt.scatter(TrainData_2[:, 0], TrainData_2[:, 1], c=TrainClasses_2, cmap=cm_bright_2)
plt.scatter(TestData_2[:, 0], TestData_2[:, 1], c=TestClasses_2, cmap=cm_bright_2,alpha=0.6)
plt.show()