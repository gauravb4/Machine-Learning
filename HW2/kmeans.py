import numpy as npy
from random import sample

#https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-k-means-clustering/
#https://datatofish.com/k-means-clustering-python/

class Point:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return ("(X: " + str(self.x) + ", Y: " + str(self.y) + ")")


class kmeans:

    def __init__(self, filename, nums, maxium):
        self.points = self.parseDoc(filename)
        self.numClusters = nums
        self.maxIterations = maxium

        
        self.centroids = self.start()


        return

    def start(self):
        initCentroids = npy.random.choice(self.points, 3)
        
        ite = 0
        difference = 10000
        while difference != 0 and ite < self.maxIterations:
            currData = self.points
            ite += 1
            

        



    def parseDoc(self, filename):
        f = open(filename, 'r')
        clusterData = f.readlines()
        pointList = []
        for line in clusterData:
            pointList.append(Point(float(line.strip().split(",")[0]), float(line.strip().split(",")[1])))
        
        return npy.array(pointList)

km = kmeans("clusters.txt", 3, 100)