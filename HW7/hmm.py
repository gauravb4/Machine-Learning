import numpy as np
import pprint as pp
from collections import defaultdict
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.axes
import os

from PIL import Image
import glob


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
        self.n, self.s, self.w, self.e,self.neighbors = self.findNeighbors()
        
    def direction(self, other):
        if self.n == other:
            return "V"
        if self.s == other:
            return "^"
        if self.w == other:
            return ">"
        if self.e == other:
            return "<"

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
        n1 = None
        n2 = None
        n3 = None
        n4 = None
        if(self.x - 1 >= 0):
            n1 = (self.x - 1, self.y)
            n.append(n1)
        if(self.x + 1 < self.gridSize):
            n2 = (self.x + 1, self.y)
            n.append(n2)
        if(self.y - 1 >= 0):
            n3 = (self.x, self.y - 1)
            n.append(n3)
        if(self.y + 1 < self.gridSize):
            n4 = (self.x, self.y + 1)
            n.append(n4)
        return n1, n2, n3, n4, n

class HMM:

    def __init__(self, filename):
        self.grid, towers, distances = self.parse(filename)
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
        grid = copy.deepcopy(self.grid.grid)
        for i in range(10):
            for j in range(10):
                if(grid[i][j].isObstacle()):
                    grid[i][j] = "0"
                else:
                    grid[i][j] = "1"
        grid[res[0].x][res[0].y] = "S"
        last = (res[0].x, res[0].y)
        for p in res[1:-1]:
            grid[p.x][p.y] = p.direction(last)
            last = (p.x, p.y)
        grid[res[-1].x][res[-1].y] = "G"
        pp.pp(grid)
        self.drawResult(grid, res)

    def drawResult(self, grid, res):
        x = [i for i in range(-2, 12, 1)]
        y = [i for i in range(-2, 12, 1)]

        cmap = mpl.colors.ListedColormap(['black','white'])
        
        zvals = copy.deepcopy(grid)
        for i in range(10):
            for j in range(10):
                if zvals[i][j] == "0":
                    zvals[i][j] = float(-5) 
                else:
                    zvals[i][j] = float(5)

        bounds=[-10,0,10]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        fig = plt.figure()
        img = plt.imshow(zvals,interpolation='nearest', cmap = cmap,norm=norm)
        
        ax = plt.gca();
        ax.set_xticks(np.arange(-.5, 10, 1))
        ax.set_yticks(np.arange(-.5, 10, 1))
        ax.set_xticklabels(np.arange(0, 10, 1), )
        ax.set_yticklabels(np.arange(0, 10, 1))
        plt.grid()
        for label in ax.get_xticklabels():
            label.set_horizontalalignment('center')

        list1 = ax.get_xticklabels()
        plt.savefig('0')
        plt.savefig('1')

        frame = 0
        
        for frame in range(11):
            self.addMoreImages(grid, res, frame)
            
        plt.savefig('13')
      
        self.saveGif()    

    def addMoreImages(self, grid, res, maxR):

        cmap = mpl.colors.ListedColormap(['black','white', 'Green', 'Red'])
        
        zvals = copy.deepcopy(grid)
        for i in range(10):
            for j in range(10):
                if zvals[i][j] == "0":
                    zvals[i][j] = float(-9) 
                else:
                    zvals[i][j] = float(-2)

        for p in res[0:maxR]:
            zvals[p.x][p.y] = float(2)

        if maxR == 10:
            p = res[-1]
            zvals[p.x][p.y] = float(7)

        bounds=[-10,-5,0,5,10]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        img2 = plt.imshow(zvals,interpolation='None', cmap = cmap,norm=norm)
        
        ax = plt.gca();
        ax.set_xticks(np.arange(-.5, 10, 1))
        ax.set_yticks(np.arange(-.5, 10, 1))
        ax.set_xticklabels(np.arange(0, 10, 1), )
        ax.set_yticklabels(np.arange(0, 10, 1))

        for label in ax.get_xticklabels():
            label.set_horizontalalignment('center')

        list1 = ax.get_xticklabels()
        plt.savefig('{}'.format(maxR + 2))

    def saveGif(self):
        frames = []
        imgs = []
        for i in range(14):
            imgs.append(glob.glob("{}.png".format(i)))
        for i in imgs:
            new_frame = Image.open(i[0])
            frames.append(new_frame)
        
        # Save into a GIF file that loops forever
        frames[0].save('route.gif', format='GIF',
                    append_images=frames[1:],
                    save_all=True,
                    duration=100, loop=10000)
        os.startfile('route.gif')

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
                        #curr.parent = node
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
            for index, pointDistances in enumerate(towerDistances):
                if pointDistances != "Empty" and self.Possible(pointDistances, distance):
                    x  = int(index / 10)
                    y = int(index % 10)
                    timesorted[timestep].add(self.grid.grid[x][y])
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
                else:
                    distances.append("Empty")
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
                #print("x and y {} {} and value {}".format(i, j, l[j]))
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