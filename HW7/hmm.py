import numpy as np
import pprint as pp
from collections import defaultdict
import copy


class PathNode:

    def __init__(self, point, parent, probability, time):
        self.point = point
        self.parent = parent
        self.probability = probability
        self.timestep = time

    def __repr__(self):
        return ("(X: " + str(self.point.x) + ", Y: " + str(self.point.y) + "), " \
                + "Parent: " + str(self.parent) + "\n"  + "Probability: " + str(self.probability) + \
                    ", "  + "Timestep: " + str(self.timestep))

    def equal(self, other):
        return (self.point == other)

class Grid:
    def __init__(self, gridSize):
        self.gridSize = gridSize
        self.grid = [[None for _ in range(gridSize)] for _ in range(gridSize)]

    def getNeighbors(self, point):
        l = point.getNeighbors()
        ret = [self.grid[p[0]][p[1]] for p in l]
        return tuple(ret)
        #x = point.x
        #y = point.y
        #l = []
        #n = None
        #if(x - 1 >= 0):
        #    n = self.grid[x - 1][y]
        #    l.append(n)
        #s = None
        #if(x + 1 < self.gridSize):
        #    s = self.grid[x + 1][y]
        #    l.append(s)
        #w = None
        #if(y - 1 >= 0):
        #    w = self.grid[x][y - 1]
        #    l.append(w)
        #e = None
        #if(y + 1 < self.gridSize):
        #    e = self.grid[x][y + 1]
        #    l.append(e)
        #return tuple(l)

class Point:

    def __init__(self, x, y, obstacle, gridSize):
        self.x = x
        self.y = y
        self.obstacle = True if obstacle == 0 else False
        self.gridSize = gridSize
        self.neighbors = self.findNeighbors()

    def equal(self, other):
        return (self.x == other.x and self.y == other.y)

    def distance(self, other):
        return np.sqrt((((self.x - other.x) * (self.x - other.x)) + ((self.y - other.y) * (self.y - other.y))))

    def __repr__(self):
        return ("(X: " + str(self.x) + ", Y: " + str(self.y) + ")")
    
    def isObstacle(self):
        return self.obstacle

    def getNeighbors(self):
        return self.neighbors

    def findNeighbors(self):
        n = []
        if(self.x - 1 >= 0):
            n.append((self.x - 1, self.y))
        if(self.x + 1 < self.gridSize):
            n.append((self.x + 1, self.y))
        if(self.y - 1 >= 0):
            n.append((self.x, self.y - 1))
        if(self.y + 1 < self.gridSize):
            n.append((self.x, self.y + 1))
        return n

class HMM:

    def __init__(self, filename):
        self.grid, towers, distances = self.parse(filename)
        self.states = []
        for row in self.grid.grid:
            for point in row:
                if not point.obstacle:
                    self.states.append((point))
        towerDistances  = self.calcDistances(towers)
        self.getNextStates(distances, towerDistances)
        self.tProb = self.transition()
        self.pruneStates()
        allPaths = self.viterbi()
        result = self.generatePath(allPaths)
        self.printResult(result)

    def printResult(self, result):
        res = []
        for node in result:
            res.append(node.point)
        pp.pp(res)

    def generatePath(self, allPaths):
        mProb = 0.0
        time = 10;
        curr = None
        final = []
        for node in allPaths[time]:
            if mProb < node.probability:
                mProb = node.probability
                curr = node
        final.append(curr)
        while curr:
            parent = curr.parent
            final.append(parent)
            curr = parent
        return reversed(final[:-1])

    def viterbi(self):
        time = 0
        paths = []
        nodes = []
        points = []
        for point in self.pointsListByTime[time]:
            prob = 1.0 / len(self.pointsListByTime[time])
            curr = PathNode(point, None, prob, time)
            nodes.append(curr)
        paths.append(nodes)
        for times in range(1, len(self.pointsListByTime)):
            nodes = paths[times - 1]
            currTimeNodes = []
            for node in nodes:
                for n in self.tProb[times - 1][node.point]:
                    curr = self.nContains(currTimeNodes, n)
                    if not curr:
                        prob = node.probability * 0.25
                        curr = PathNode(n, node, prob, times)
                        currTimeNodes.append(curr)
                    else:
                        prob = node.probability * 0.25
                        curr.probability += prob
            paths.append(currTimeNodes)            

        return paths

    def pruneStates(self):
        i = len(self.pointsListByTime) - 1
        
        for timestep in self.pointsListByTime[:0:-1]:
            
            prune = []
            for point in timestep:
                p = True
                neighbors = self.grid.getNeighbors(point)
                for n in neighbors:
                    if self.pContains(self.pointsListByTime[i - 1], n):
                        p = False
                
                if p:
                    prune.append(point)

            for point in prune:
                self.pointsListByTime[i].remove(point)
            i -= 1

        pStates = set()
        stateTimes = defaultdict(list)

        for timestep, points in enumerate(self.pointsListByTime):
            for point in points:
                pStates.add(point)
                stateTimes[point].append(timestep)

        stateTimes = dict(stateTimes)

        for timestep, points in enumerate(self.tProb):
            for point in points:
               
                p = set()
                for n in self.tProb[timestep][point]:
                    if not self.pContains(self.pointsListByTime[timestep + 1], n):
                        p.add(n)
                
                for n in p:
                    self.tProb[timestep][point].pop(n)     
        

    def transition(self):
        tProb = []
        temp = copy.deepcopy(self.pointsListByTime)
        for timestep, points in enumerate(self.pointsListByTime[:-1]):
            currTimestep = {}
            prune = []
            for point in points:
                p = True
                timeStepState = {}
                currTimestep[point] = timeStepState
                neighbors = self.grid.getNeighbors(point)
                for n in neighbors:
                    if self.pContains(self.pointsListByTime[timestep + 1],n):
                        p = False
                    timeStepState[n] = 0.25
                if p:
                    currTimestep.pop(point)
                    prune.append(point)
            
            for point in prune:
                self.pointsListByTime[timestep].remove(point)

            tProb.append(currTimestep)
        
        return tProb

    def pContains(self, pointlist, point):
        for p in pointlist:
            if(p.equal(point)):
                return True
        return False

    def nContains(self, nodeList, node):
        for n in nodeList:
            if(n.equal(node)):
                return n
        return None
        
    def getNextStates(self, distances, towerDistances):
        timesorted = []
        for timestep, distance in enumerate(distances):
            timesorted.append(set())
            for index, state_distances in enumerate(towerDistances):
                if self.Possible(state_distances, distance):
                    timesorted[timestep].add(self.states[index])
        self.pointsListByTime = timesorted

    def Possible(self, state, distance):
        for actual, measured in zip(state, distance):
            if actual*0.7 > measured or actual*1.3 < measured:
                return False
        return True

    def calcDistances(self, towers):
        distances = []
        for row in self.grid.grid:
            for point in row:
                if not point.obstacle:
                    t = []
                    for tower in towers:
                        t.append(point.distance(tower))
                    distances.append(tuple(t))
                #else:
                #    distances.append("Empty")
        return distances

    def parse(self, filename):
        f = open(filename, "r")
        
        d = f.readlines()
        d1 = d[2:12]
        d1 = [line.strip() for line in d1]
        grid = []
        gridSize = 10
        g = Grid(gridSize)
        for i in range(gridSize):
            l = d1[i].split(" ")
            for j in range(gridSize):
                g.grid[i][j] = Point(i, j, int(l[j]), gridSize)

        towers = []
        d2 = d[16:20]
        d2 = [line.strip() for line in d2]
        for t in d2:
            x = int(t.split(":")[1][1])
            y = int(t.split(":")[1][3])
            towers.append(Point(x, y, 1, gridSize))

        noisyDistances = []
        d3 = d[24:35]
        d3 = [line.strip() for line in d3]
        noisy = [line.split() for line in d3]
        for vals in noisy:
            step = []
            for val in vals:
                step.append(float(val))
            noisyDistances.append(step)
        
        return g, towers, noisyDistances

 
h = HMM('hmm-data.txt')