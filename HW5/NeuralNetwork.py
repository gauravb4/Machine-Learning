import numpy as np 
import pprint as pp

class NeuralNetwork:

    def __init__(self, trainFile, testFile):
        self.trainData = self.parse(trainFile)
        self.testData = self.parse(testFile)
        self.numLayers = 100
        self.weights = {}    
    
    def parse(self, file):
        file_list = []
        with open(file, 'r') as f:
            for line in f:
                label = 1 if "down" in line else 0
                file_list.append([line.strip(), label])
        return np.array(file_list)

    def sig(self, s):
        return 1/(1+np.exp(-s))

    def sig_b(self, s):
        return sig(s) * (1 - sig(s))

test = "downgesture_test.list"
train = "downgesture_train.list"
nn = NeuralNetwork(train, test)    