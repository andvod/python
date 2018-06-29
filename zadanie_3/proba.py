
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_classification
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import confusion_matrix
from numpy import array
import numpy as np

data = [[0, 0], [0, 1], [2, 2], [3, 3]]
classes = [1, 1, 2, 2]

d = pairwise_distances(data)
print d

def diam(data, classes):
    """
    najmniejsza odleglosc miedzy dwuma klasami
    """
    d = pairwise_distances(data)
    d_min = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if (classes[i] != classes[j]):
                d_min.append(d[i][j])
    return min(d_min)

def dunn(data, classes):
    """
    najwiksza srednica
    """
    d = pairwise_distances(data)
    d_max = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if (classes[i] == classes[j]):
                d_max.append(d[i][j])
    return diam(data, classes) / max(d_max)

print dunn(data, classes)

d = [2, 4, 5, 6, 9, 7]
index = d.index(max(d))
print d[index], index