
import numpy as np
from sklearn.datasets import load_iris

#wczytanie zbioru danych
iris_data = load_iris()

#wektory cech
print iris_data.data

#etykiety klasowe - 0,1,2
print iris_data.target

#etykieta i odpowiada i-temu elementowi ponizszej listy
labels=list(iris_data.target_names)
print labels