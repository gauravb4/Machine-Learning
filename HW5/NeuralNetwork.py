import numpy as np 
import pprint as pp

class NeuralNetwork:

    def __init__(self, trainFile, testFile, epochs=1000, learningRate=0.1):
        self.epochs = epochs
        self.learningRate = learningRate
        tData = self.parse(trainFile)
        self.trainData = self.process(tData)
        tData = self.parse(testFile)
        self.testData = self.process(tData)
        self.hiddenLayer = (960,100)
        self.inputLayer = (100,1)
        self.numLayers = 2
        self.weights = {}
        self.weights[0] = np.random.uniform(-0.01, 0.01, (self.hiddenLayer[1], self.hiddenLayer[0]))
        self.weights[1] = np.random.uniform(-0.01, 0.01, (self.inputLayer[1], self.inputLayer[0]))
        self.biases = {}
        self.biases[0] = np.random.uniform(-0.01, 0.01, (self.hiddenLayer[1], 1))
        self.biases[1] = np.random.uniform(-0.01, 0.01, (self.inputLayer[1], 1))
        self.train(self.trainData)    
    
    def fullForward(self, values):
        results = {}
        currSet = values

        for i in range(self.numLayers):
            prevSet = currSet
            currentWeight = self.weights[i]
            currentBias = self.biases[i]
            weightedSum = np.dot(currentWeight, prevSet) + currentBias
            currSet = self.sig(weightedSum)
            results['A' + str(i)] = prevSet
            results['Z' + str(i)] = weightedSum
        
        return currSet, results
    
    def getAccuracy(self, new, old):
        temp = np.zeros(new.shape)
        for i in range(len(new[0])):
            if new[0, i] > 0.5:
                temp[0, i] = 1
        new = temp
        return (new == old).all(axis=0).mean()
    
    def fullBackward(self, new, old, results):
        gradient = []
        old = old.reshape(new.shape)
        return gradient
    
    def train(self, trainingData):
        history = {}
        vals = trainingData[0].T
        labels = np.array([(trainingData[1].T)])

        for curr in range(self.epochs):
            newLabels, results = self.fullForward(vals)
            c = np.sum((labels-newLabels) ** 2)
            ac = self.getAccuracy(newLabels, labels)
            history[curr] = (c, ac)
            gradient = self.fullBackward(newLabels, labels, results)
            for i in range(self.numLayers):
                self.weights[i] = self.weights[i] - (self.learningRate * gradient[i][0])
                self.biases[i] = self.biases[i] - (self.learningRate * gradient[i][1])
            continue


    def process(self, data):
       listOne = []
       listTwo = []
       for i in range(len(data)):
           listOne.append(data[i][0])
           listTwo.append(data[i][1])
       return (np.array(listOne), np.array(listTwo))
    
    def parse(self, file):
        file_list = []
        with open(file, 'r') as f:
            for line in f:
                label = 1 if "down" in line else 0
                img = self.parse_img('gestures/' + line.strip())
                file_list.append([img, label])
        # print(file_list[1])
        return file_list

    def parse_img(self, fn):
        f = open(fn, 'rb')
        #assert f.readline() == 'P5\n'
        f.readline()
        f.readline()
        dim = [int(i) for i in f.readline().split()]
        assert int(f.readline()) == 255
        # print(dim)

        img = np.zeros(dim[0] * dim[1])
        for i in range(dim[1]):
            for j in range(dim[0]):
                # print(i, j)
                img[i * dim[0] + j] = ord(f.read(1))/255.0
        return img

    def sig(self, s):
        return 1/(1+np.exp(-s))

    def sig_b(self, s):
        return self.sig(s) * (1 - self.sig(s))

test = "downgesture_test.list"
train = "downgesture_train.list"
nn = NeuralNetwork(train, test)    