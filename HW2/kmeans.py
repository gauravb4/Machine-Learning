import numpy as npy
import pprint as pp
from random import sample

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

        
        self.clusters, self.centroids = self.start()
        pp.pprint(self.clusters)
        pp.pprint(self.centroids)


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
            newPoints.append((key, Point(sumX / lenCluster, sumY / lenCluster)))

        return newPoints


    def start(self):
        currCentroids = npy.random.choice(self.points, 3)
        ite = 0
        difference = 10000
        clusters = self.assignClusters(self.points, currCentroids)
        while difference != 0 and ite < self.maxIterations:
            newCentroids = self.reCalcCentroids(clusters)
            found = True
            for x, y in newCentroids:
                if(x.equal(y) == False):
                    found = False
            if(found == False):
                break

            clusters = self.assignClusters(self.points, newCentroids)
                
            ite += 1
            currCentroids = newCentroids

        return clusters, currCentroids

    def parseDoc(self, filename):
        f = open(filename, 'r')
        clusterData = f.readlines()
        pointList = []
        for line in clusterData:
            pointList.append(Point(float(line.strip().split(",")[0]), float(line.strip().split(",")[1])))
        
        return npy.array(pointList)

km = kmeans("clusters.txt", 3, 1)