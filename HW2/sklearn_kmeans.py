import numpy as np 
import sklearn.cluster


def parse(file):
    f = open(file, 'r')
    clusterData = f.readlines()
    pointList = []
    for line in clusterData:
        pointList.append((float(line.strip().split(",")[0]), (line.strip().split(",")[1])))
    
    return pointList
    
data = parse('clusters.txt')
km = sklearn.cluster.KMeans(n_clusters=3)
km.fit(data)
print(km.cluster_centers_)