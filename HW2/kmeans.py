import numpy as npy
import pprint as pp
from random import sample

#https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-k-means-clustering/
#https://datatofish.com/k-means-clustering-python/

class Point:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def equal(self, other):
        return (self.x == other.x and self.y == other.y)
    
    def distance(self, other):
        return npy.sqrt((((self.x - other.x) * (self.x - other.x)) + ((self.y - other.y) * (self.y - other.y))))

    def __repr__(self):
        return ("(X: " + str(self.x) + ", Y: " + str(self.y) + ")")


class kmeans:

    def __init__(self, filename, nums, maxium):
        self.points = self.parseDoc(filename)
        self.numClusters = nums
        self.maxIterations = maxium

        
        self.centroids = self.start()


    def assignClusters(self, data, centroids):
        currMap = {c : [] for c in centroids}
        for point in data:
            minDist = 10000000
            cc = None
            for centroid in centroids:
                dist = point.distance(centroid)
                if dist < minDist:
                    minDist = dist
                    cc = centroid
            currMap[cc].append(point)

        return currMap


    def reCalcCentroids(self, clusterData):
        newPoints = []
        for key in clusterData:
            sumX = 0
            sumY = 0
            lenCluster = len(clusterData[key])
            for p in clusterData[key]:
                sumX += p.x
                sumY += p.y
            newPoints.append(Point(sumX / lenCluster, sumY / lenCluster))

        return newPoints


    def start(self):
        currCentroids = npy.random.choice(self.points, 3)
        mainmap = None
        ite = 0
        difference = 10000
        while difference != 0 and ite < self.maxIterations:
            currData = self.points
            clusters = self.assignClusters(currData, currCentroids)
            newCentroids = self.reCalcCentroids(clusters)
            found = False
            for c in newCentroids:
                
            ite += 1
            mainmap = clusters

        return mainmap

    def parseDoc(self, filename):
        f = open(filename, 'r')
        clusterData = f.readlines()
        pointList = []
        for line in clusterData:
            pointList.append(Point(float(line.strip().split(",")[0]), float(line.strip().split(",")[1])))
        
        return npy.array(pointList)

km = kmeans("clusters.txt", 3, 1)