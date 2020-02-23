import numpy as np
import pprint as pp
import math
from random import sample
from kmeans import kmeans 

class Point:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.Ric = []

    def equal(self, other):
        return (self.x == other.x and self.y == other.y)
    
    def distance(self, other):
        return np.sqrt((((self.x - other.x) * (self.x - other.x)) + ((self.y - other.y) * (self.y - other.y))))

    def __repr__(self):
        return ("(X: " + str(self.x) + ", Y: " + str(self.y) + ")")


class Gaussian:
    def __init__(self, mean, covariance, pi):
        self.mean = mean
        self.covariance = covariance
        self.pi = pi

    def equal(self, other):
        return (np.array_equal(self.mean, other.mean) and np.array_equal(self.covariance, other.covariance) and self.pi == other.pi)

    def __repr__(self):
        return ("mean: " + str(self.mean) + "\n covariance " + str(self.covariance) + "\n pi " + str(self.pi))

class gmm:

    def __init__(self, filename, nums, maxium, thresh):
        self.points = self.parseDoc(filename)
        self.numClusters = nums
        self.maxIterations = maxium
        self.threshold = thresh
        self.assignRandomRic()
        km = kmeans(filename, nums, maxium)
        centroids = km.getClusters(km)
        self.answer = self.start(centroids)
        pp.pprint(self.answer)
        pp.pprint(centroids.keys())


    def assignRandomRic(self):
        for point in self.points:
            temp = np.random.randint(1,10,self.numClusters)
            for val in temp:
                point.Ric.append(val / np.sum(temp))
            point.Ric = np.array(point.Ric)


    def recomputeGaussians(self):
        gaussians = np.array([None] * self.numClusters)
        for c in range(self.numClusters):
            sumPoints = np.array([0.0, 0.0])
            covariance = np.zeros((2,2), dtype=np.float)
            clusterWeight = 0.0
            for point in self.points:
                xVal = float(point.x)
                yVal = float(point.y)
                ric  = float(point.Ric[c])
                xVal *= ric
                yVal *= ric
                sumPoints += ((xVal, yVal))
                clusterWeight += ric
            mean = sumPoints/float(clusterWeight)
            for point in self.points:
                p = np.array((float(point.x), float(point.y)))
                firstBit = ((p- mean)[np.newaxis]).T
                covariance += (point.Ric[c] / clusterWeight)* np.matmul(firstBit, firstBit.T)
            pi  = clusterWeight / len(self.points) 
            gaussians[c] = Gaussian(mean, covariance, pi)
        return gaussians

    
    def calcPVal(self, gaus, X, Y):
        p = np.array((float(X), float(Y)))
        firstBit = (p- gaus.mean)[np.newaxis]
        factor = np.matmul(np.matmul(firstBit, np.linalg.inv(gaus.covariance)), firstBit.T)
        constant = 1 / np.sqrt(((2 * math.pi)**((self.numClusters))) * np.linalg.det(gaus.covariance)) 
        return gaus.pi * constant * math.exp(-1 / 2 * (factor.item()))


    def recomputeRics(self, currGaussians):
        for point in self.points:
            temp = [None] * self.numClusters
            for c in range(self.numClusters):
                temp[c] = self.calcPVal(currGaussians[c], point.x, point.y)
            totalN = float(np.sum(temp))
            temp = np.array(temp) / totalN
            for c, vals in enumerate(temp):
                point.Ric[c] = vals


    def calcLog(self, gaussians):
        gamma = np.zeros((len(self.points), len(gaussians)))
        d = np.array([(point.x, point.y) for point in self.points])
        for c, g in enumerate(gaussians):
            top = g.pi * np.exp(-0.5 * np.sum(np.multiply(np.dot(d - g.mean, np.linalg.inv(g.covariance)), d - g.mean), axis = 1))
            bot = np.sqrt((2 * math.pi) ** (self.numClusters) * np.linalg.det(g.covariance))
            gamma[:, c] = top / bot
        return np.sum(np.log(np.sum(gamma, axis = 1)))


    def initGaussians(self, clusterData):
        gaussians = np.array([None] * self.numClusters)
        for c, key in enumerate(clusterData):
            mean = (key.x, key.y)
            covariance  = np.eye(2)
            pi = 1 / self.numClusters
            gaussians[c] = Gaussian(mean, covariance, pi)
        return gaussians

    def start(self, centroids):
        ite = 0
        currGaussians = self.initGaussians(centroids)
        
        while ite < self.maxIterations:
            #print(ite)
            self.recomputeRics(currGaussians)
            newGaussians =  self.recomputeGaussians()
            currVal = self.calcLog(currGaussians)
            newVal = self.calcLog(newGaussians)
            if np.abs(currVal - newVal) <= self.threshold:
                break

            currGaussians = newGaussians
            ite += 1

        return currGaussians            



    def parseDoc(self, filename):
        f = open(filename, 'r')
        clusterData = f.readlines()
        pointList = []
        for line in clusterData:
            pointList.append(Point(float(line.strip().split(",")[0]), float(line.strip().split(",")[1])))
        
        return np.array(pointList)

g = gmm("clusters.txt", 3, 1000, 0.001)
