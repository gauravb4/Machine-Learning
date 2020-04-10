import numpy as np 
import pprint as pp

class NeuralNetwork:

    def __init__(self, trainFile, testFile, epochs=1000, learningRate=0.1):
        self.epochs = epochs
        self.learningRate = learningRate
        tData, _ = self.parse(trainFile)
        self.trainData = self.process(tData)
        tData, tDataName = self.parse(testFile)
        self.testData = self.process(tData)
        self.testFiles = tDataName
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
        self.test(self.testData)       

    def fullForward(self, values):
        results_A = {}
        results_Z = {}
        currSet = values

        for i in range(self.numLayers):
            prevSet = currSet
            currentWeight = self.weights[i]
            currentBias = self.biases[i]
            weightedSum = np.dot(currentWeight, prevSet) + currentBias
            currSet = self.sig(weightedSum)
            results_A[i] = prevSet
            results_Z[i] = weightedSum
        return currSet, results_A, results_Z
    
    def getAccuracy(self, new, old):
        temp = np.zeros(new.shape)
        for i in range(len(new[0])):
            if new[0, i] > 0.5:
                temp[0, i] = 1
        new = temp
        return (new == old).all(axis=0).mean(), new
    
    def fullBackward(self, new, old, results_A, results_Z):
        gradient = {}
        old = old.reshape(new.shape)

        prev_dA = (np.divide(old, new) - np.divide(1-old, 1-new)) * -1
        for i in reversed(range(self.numLayers)):
            curr_dA = prev_dA
            prev_A = results_A[i]
            curr_Z = results_Z[i]
            curr_weight = self.weights[i]

            dZ = self.sig_b(curr_Z) * curr_dA
            dW = np.dot(dZ, prev_A.T) / prev_A.shape[1]
            dB = np.sum(dZ, axis=1, keepdims=True) / prev_A.shape[1]
            prev_dA = np.dot(curr_weight.T, dZ)

            gradient[i] = [dW, dB]
        return gradient
    
    def train(self, trainingData):
        history = {}
        vals = trainingData[0].T
        labels = np.array([(trainingData[1].T)])

        for curr in range(self.epochs):
            newLabels, results_A, results_Z = self.fullForward(vals)
            c = np.sum((labels-newLabels) ** 2)
            ac, _ = self.getAccuracy(newLabels, labels)

            if ((curr+1)%100 == 0):
                print("Epoch " +  str(curr) +  " Accuracy: " +  str(ac))

            history[curr] = (c, ac)
            gradient = self.fullBackward(newLabels, labels, results_A, results_Z)
            for i in range(self.numLayers):
                self.weights[i] = self.weights[i] - (self.learningRate * gradient[i][0])
                self.biases[i] = self.biases[i] - (self.learningRate * gradient[i][1])
            continue

    def test(self, testingData):
        results, _, _ =  self.fullForward(testingData[0].T)
        a, c = self.getAccuracy(results, np.array([(testingData[1].T)])) 
        print("*******************")
        print("Final accuracy: " + str(a))
        c = c.T
        test_labels = (np.array([(testingData[1].T)])).T
        matches = []
        row = []
        for i in range(len(c)):
            if(c[i] == test_labels[i]):
                row.append(str(i) + "-Match")
            else:
                row.append(str(i) + "-NoMatch")
            if (i+1)%5 == 0:
                matches.append(row)
                row = []
        pp.pprint(matches)
        print("*******************")

    def process(self, data):
       listOne = []
       listTwo = []
       for i in range(len(data)):
           listOne.append(data[i][0])
           listTwo.append(data[i][1])
       return (np.array(listOne), np.array(listTwo))
    
    def parse(self, file):
        file_list = []
        file_names = []
        with open(file, 'r') as f:
            for line in f:
                label = 1 if "down" in line else 0
                img = self.parse_img('gestures/' + line.strip())
                file_list.append([img, label])
                file_names.append(line.strip())
        return file_list, file_names

    def parse_img(self, fn):
        f = open(fn, 'rb')
        f.readline()
        f.readline()
        dim = [int(i) for i in f.readline().split()]
        assert int(f.readline()) == 255

        img = np.zeros(dim[0] * dim[1])
        for i in range(dim[1]):
            for j in range(dim[0]):
                img[i * dim[0] + j] = ord(f.read(1))/255.0
        return img

    def sig(self, s):
        return 1/(1+np.exp(-s))

    def sig_b(self, s):
        return self.sig(s) * (1 - self.sig(s))

test = "downgesture_test.list"
train = "downgesture_train.list"
nn = NeuralNetwork(train, test)    