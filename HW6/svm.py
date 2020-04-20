import numpy as np 
import pprint as pp
import copy
import math

class Point:
    
    def __init__(self, x, y, l):
        self.x = x
        self.y = y
        self.label = l

    def equal(self, other):
        return (self.x == other.x and self.y == other.y)
    
    def distance(self, other):
        return np.sqrt((((self.x - other.x) * (self.x - other.x)) + ((self.y - other.y) * (self.y - other.y))))

    def __repr__(self):
        return ("(X: " + str(self.x) + ", Y: " + str(self.y) + ")\n\t" + "Label: " + str(self.label))

class SupportVectorMachine:

    def __init__(self, type):
        filename = "linsep.txt" if type == "linear" else "nonlinsep.txt"
        self.points = self.parseDoc(filename)
        self.type = type
        self.fit()

    def linKernelGen(self, pos, i, j):
        return np.dot(pos[i], pos[j])

    def nonLinGen(self, pos, i, j):
        exp = 0.0
        for k in range(pos[i]):
            exp += (pos[i][k] - pos[j][k])**2
        return math.exp(-0.005 * exp)

    def fit(self):
        pos = []
        ls = []
        for point in self.points:
            pos.append((point.x, point.y))
            ls.append(point.label)
        
        pos = np.array(pos)
        ls = np.array(ls)
        K = np.zeros(len(pos), len(pos))
        for i in range(len(pos)):
            for j in range(len(pos)):
                if self.type == "linear":
                    K[i, j] = self.linKernelGen(pos, i, j)
                else:
                    K[i, j] = self.nonLinGen(pos, i, j)
                
        return

    def parseDoc(self, filename):
        f = open(filename, 'r')
        clusterData = f.readlines()
        pointList = []
        for line in clusterData:
            pointList.append(Point(float(line.strip().split(",")[0]), \
                                   float(line.strip().split(",")[1]), \
                                   int(line.strip().split(",")[2])))
        
        return np.array(pointList)

linstep = 'linstep.txt'
nonlinstep = 'nonlinstep.txt'

svm = SupportVectorMachine("linear")

svm_n = SupportVectorMachine("nonlinear")
