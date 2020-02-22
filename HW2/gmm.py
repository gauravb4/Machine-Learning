import numpy as npy
import pprint as pp
from random import sample
import kmeans

class Point:

    def __init__(self, x, y, ric=None):
        self.x = x
        self.y = y
        self.Ric = ric

    def equal(self, other):
        return (self.x == other.x and self.y == other.y)
    
    def distance(self, other):
        return npy.sqrt((((self.x - other.x) * (self.x - other.x)) + ((self.y - other.y) * (self.y - other.y))))

    def __repr__(self):
        return ("(X: " + str(self.x) + ", Y: " + str(self.y) + ")")

    
class gmm:

    def __init__(self, filename, nums, maxium, thresh):
        self.points = self.parseDoc(filename)
        self.numClusters = nums
        self.maxIterations = maxium
        self.threshold = thresh

    def management(self, currWeights):


    
    def start():
        ite = 0
        currWeights = None
        while ite < self.maxIterations:
            self.management(currWeights)



    def parseDoc(self, filename):
        f = open(filename, 'r')
        clusterData = f.readlines()
        pointList = []
        for line in clusterData:
            pointList.append(Point(float(line.strip().split(",")[0]), float(line.strip().split(",")[1])))
        
        return npy.array(pointList)

g = gmm("clusters.txt", 3, 1000, 0.01)