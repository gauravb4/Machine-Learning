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

    def fit(self):
        pass

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
