import numpy as np
import pprint as pp
from random import sample
import matplotlib.pyplot as plt

class Point:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def equal(self, other):
        return (self.x == other.x and self.y == other.y)
    
    def distance(self, other):
        return np.sqrt((((self.x - other.x) * (self.x - other.x)) + ((self.y - other.y) * (self.y - other.y))))

    def __repr__(self):
        return ("(X: " + str(self.x) + ", Y: " + str(self.y) + ")")


class kmeans:

    def __init__(self, filename, nums, maxium):
        self.points = self.parseDoc(filename)
        self.numClusters = nums
        self.maxIterations = maxium

        
        self.clusters, self.centroids = self.start()
        pp.pprint(self.centroids)
        #self.display()
    
    @staticmethod
    def getClusters(self):
        return self.clusters

    @staticmethod
    def getCentroids(self):
        return self.centroids

    def display(self):
        data = []
        colors = ("red", "green", "blue")
        for key in self.clusters:
            g = [[], []]
            for point in self.clusters[key]:
                g[0].append(point.x)
                g[1].append(point.y)
            data.append(g)
        
        groups = ("coffee", "tea", "water")

        # Create plot
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        for data, color, group in zip(data, colors, groups):
            x, y = data
            ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)

        plt.title('Matplot scatter plot')
        plt.legend(loc=2)
        plt.show()
        


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
        currCentroids = np.random.choice(self.points, 3)
        ite = 0
        clusters = self.assignClusters(self.points, currCentroids)
        while ite < self.maxIterations:
            newCentroids = self.reCalcCentroids(clusters)
            #print(ite)
            found = True
            i = 0
            for x, y in newCentroids:
                if(x.equal(y) == False):
                    found = False
                    currCentroids[i] = y
                    i += 1  
            if(found == True):
                break
            clusters = self.assignClusters(self.points, currCentroids)
            ite += 1

        return clusters, currCentroids

    def parseDoc(self, filename):
        f = open(filename, 'r')
        clusterData = f.readlines()
        pointList = []
        for line in clusterData:
            pointList.append(Point(float(line.strip().split(",")[0]), float(line.strip().split(",")[1])))
        
        return np.array(pointList)

km = kmeans("clusters.txt", 3, 1000)