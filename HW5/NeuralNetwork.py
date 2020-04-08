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
                img = self.parse_img('gestures/' + line.strip())
                file_list.append([img, label])
        # print(file_list[1])
        return np.array(file_list)

    def parse_img(self, fn):
        f = open(fn, 'rb')
        assert f.readline() == 'P5\n'
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
        return sig(s) * (1 - sig(s))

test = "downgesture_test.list"
train = "downgesture_train.list"
nn = NeuralNetwork(train, test)    