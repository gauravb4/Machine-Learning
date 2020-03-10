import numpy as np
import pprint as pp
import random as rr
from collections import defaultdict

class FastMap():

    def __init__(self, filename):
        self.distances, self.objList = self.parse(filename)
        self.new_dimensions = defaultdict(list)
        self.current_dimension = 0
        self.currMax, self.maxDistance = self.initfurthestpoints()
        self.vals = {}
        self.initCords()
        self.leftVals = 2 - len(self.vals[self.currMax[0]])
        pp.pprint(self.vals)

        for i in range(self.leftVals):
            self.currMax, self.maxDistance = self.furthestPoints()
            self.updateCords()

        pp.pprint(self.vals)


    def updateCords(self):
        a = self.currMax[0]
        b = self.currMax[1]
        newVals = {}
        checkSet = []
        checkSet.append(a)
        checkSet.append(b)
        for obj in self.objList:
            if obj not in self.vals:
                self.vals[obj] = []
            checkSet.append(obj)
            checkSet.sort()
            one = checkSet[0]
            two = checkSet[1]
            three = checkSet[2]
            v1 = (self.distance(a, obj)) ** 2 if obj != a else 0
            v2 = (self.distance(b, obj)) ** 2 if obj != b else 0
            v3 = self.maxDistance ** 2
            v4 = 2 * self.maxDistance
            self.vals[obj].append((v1 + v3 - v2) / v4)
            checkSet.remove(obj)     


    def initDist(self, p1, p2):
        if p1 > p2:
            p1, p2 = p2, p1
        return self.distances[p1][p2]

    def initCords(self):
        a = self.currMax[0]
        b = self.currMax[1]
        checkSet = []
        checkSet.append(a)
        checkSet.append(b)
        for obj in self.objList:
            if obj not in self.vals:
                self.vals[obj] = []
            checkSet.append(obj)
            checkSet.sort()
            one = checkSet[0]
            two = checkSet[1]
            three = checkSet[2]

            v1 = (self.initDist(a, obj)) ** 2 if obj != a else 0
            v2 = (self.initDist(b, obj)) ** 2 if obj != b else 0
            v3 = self.maxDistance ** 2
            v4 = 2 * self.maxDistance
            self.vals[obj].append((v1 + v3 - v2) / v4)
            checkSet.remove(obj)        

    def parse(self, file):
        f = open(file, 'r')
        data = f.readlines()
        data = [[int(line.strip().split("\t")[0]), int(line.strip().split("\t")[1]), int(line.strip().split("\t")[2])] for line in data]

        distances = {}
        points = set()
        for line in data:
            if line[0] not in distances:
                distances[line[0]] = {}
            distances[line[0]][line[1]] = line[2]
            points.add(line[0])
            points.add(line[1])

        return distances, list(points)


    def initfurthestpoints(self):
        newDist  = {}
        maxVal = -1111111
        r = rr.choice(self.objList)
        currTuple = (-1,-1)
        for curr in range(3):
            for p in self.objList:
                if p != r:
                    dist = None
                    if r > p:
                        dist = self.distances[p][r]
                    else:
                        dist = self.distances[r][p]
                    if dist > maxVal:
                        currTuple = (r, p)
                        maxVal = dist        
            r = currTuple[1]
        if currTuple[0] > currTuple[1]:
            currTuple = (currTuple[1], currTuple[0])
        return currTuple, maxVal


    def distance(self, p1, p2):
        sub = 0
        for idx in range(self.leftVals):
            sub += ((self.vals[p1][idx] - self.vals[p2][idx]) ** 2)
        if p1 > p2:
            p1, p2 = p2, p1
        return np.sqrt(((self.distances[p1][p2]) ** 2) - sub)


    def furthestPoints(self):
        newDist  = {}
        maxVal = -1111111
        r = rr.choice(self.objList)
        for curr in range(6):
            for p in self.objList:
                if p != r:
                    dist = None
                    dist = self.distance(r, p)
                    if dist > maxVal:
                        currTuple = (r, p)
                        maxVal = dist        
            r = currTuple[1]
        if currTuple[0] > currTuple[1]:
            currTuple = (currTuple[1], currTuple[0])
        return currTuple, maxVal



filename = "fastmap-data.txt"
fm = FastMap(filename)
#distances, ids = parse(filename)
#fastmap(distances, ids)
