import numpy as np
import pprint as pp

class LR:

    def __init__(self, filename, dimensions, numIterations):
        self.data = self.parse(filename)
        self.dim = dimensions
        self.weights = np.zeros(dimensions + 1)
        self.out = open("log.txt", "w+")
        for i in range(numIterations):
            self.train(i)
        pp.pp(self.weights)
        pp.pp(self.checkAccuracy(self.weights))

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

    def train(self, i):
        currProb = np.zeros(self.dim + 1)
        for d in self.data:
            point = d["point"]
            label = d["label"]
            dot = np.dot(self.weights, point) * label
            e = np.exp(dot)
            final = 1 / (1 + e)
            for i in range(self.dim + 1):
                currProb[i] += point[i] * label * final
        for i in range(self.dim + 1):
            self.weights[i] += (1 / len(self.data)) * currProb[i]

    def checkAccuracy(self, weights):
        length = len(self.data)
        correct = 0
        for d in self.data:
            point = d["point"]
            label = d["label"]
            dotP = np.dot(weights, point)
            e = np.exp(dotP)
            final = e / (1 + e)
            val = -1 if final < 0.5 else 1
            if val == label:
                correct += 1
        accuracy = float(correct)/length
        return accuracy

lr = LR("classification.txt", 3, 7000)