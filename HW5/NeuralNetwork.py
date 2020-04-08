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
                file_list.append(line.strip())
        return file_list



test = "downgesture_test.list"
train = "downgesture_train.list"
nn = NeuralNetwork(train, test)    