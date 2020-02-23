import numpy as np 
import sklearn.mixture


def parse(file):
    f = open(file, 'r')
    clusterData = f.readlines()
    pointList = []
    for line in clusterData:
        pointList.append((float(line.strip().split(",")[0]), (line.strip().split(",")[1])))
    
    return pointList
    
data = parse('clusters.txt')
gmm = sklearn.mixture.GaussianMixture(n_components=3)
gmm.fit(data)
print("Means: ")
print(gmm.means_)
print(" ")
print("Covariances: ")
print(gmm.covariances_)
print(" ")
print("Weights: ")
print(gmm.weights_)
print(" ")