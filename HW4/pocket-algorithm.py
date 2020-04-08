import numpy as np
import pprint as pp
import copy
import matplotlib.pyplot as plt

class PA:

    def __init__(self, filename, dimensions, numIterations):
        self.data = self.parse(filename)
        self.weights = np.zeros(dimensions + 1)
        self.pocket = np.zeros(dimensions + 1)
        self.accRate = None
        self.xpoints = []
        self.ypoints = []
        for i in range(numIterations):
            self.findPocket(i)
        pp.pp(self.checkAccuracy(self.pocket))
        pp.pp(self.pocket)
        plt.plot(self.xpoints, self.ypoints)
        plt.show()


    def parse(self, filename):
        f = open(filename, 'r')
        fl = f.readlines()
        data = []
        for line in fl:
            d = {}
            l = line.strip().split(",")
            d["point"] = np.array([float(1.0) ,float(l[0]), float(l[1]), float(l[2])])
            d["label"] = int(l[4])
            data.append(d)
        return data

    def findPocket(self, i):
        self.learn()
        currRate = self.checkAccuracy(self.weights)
        
        if self.accRate is None:
            self.accRate = currRate
            self.pocket = copy.deepcopy(self.weights)
        elif(currRate > self.accRate):
            self.accRate = currRate
            self.pocket = copy.deepcopy(self.weights)
        self.xpoints.append(i)
        self.ypoints.append(((1-self.accRate) * len(self.data)))
        

    def checkAccuracy(self, weights):
        length = len(self.data)
        correct = 0
        for row in self.data:
            dotP = np.dot(weights, row["point"])
            val = 1 if dotP > 0 else -1
            if val == row["label"]:
                correct += 1
        accuracy = float(correct)/length
        return accuracy
    
    def learn(self):
        for d in self.data:
            point = d["point"]
            label = d["label"]
            dotP = np.dot(self.weights, point)
            if dotP <= 0 and label == 1:
                self.weights = np.add(self.weights, point)
            elif dotP > 0 and label == -1:
                self.weights = np.subtract(self.weights, point)

pa = PA("classification.txt", 3, 7000)