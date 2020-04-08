import pprint as pr
import numpy as np

class PLA:

    def __init__(self, filename, dimensions, numIterations):
        self.data = self.parse(filename)
        self.weights = np.zeros(dimensions + 1)
        for _ in range(numIterations):
            self.learn()
        
        print(self.checkAccuracy())
        pr.pp(self.weights)

    def checkAccuracy(self):
        length = len(self.data)
        correct = 0
        for row in self.data:
            dotP = np.dot(self.weights, row["point"])
            val = 1 if dotP > 0 else -1
            if val == row["label"]:
                correct += 1
        accuracy = float(correct)/length
        return accuracy

    def parse(self, filename):
        f = open(filename, 'r')
        fl = f.readlines()
        data = []
        for line in fl:
            d = {}
            l = line.strip().split(",")
            d["point"] = np.array([float(1.0) ,float(l[0]), float(l[1]), float(l[2])])
            d["label"] = int(l[3])
            data.append(d)
        return data

    def learn(self):
        for d in self.data:
            point = d["point"]
            label = d["label"]
            dotP = np.dot(self.weights, point)
            if dotP <= 0 and label == 1:
                self.weights = np.add(self.weights, point)
            elif dotP > 0 and label == -1:
                self.weights = np.subtract(self.weights, point)
        

pla = PLA("classification.txt", 3, 20)