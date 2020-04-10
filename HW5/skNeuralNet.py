from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import pprint as pp


def process(data):
    listOne = []
    listTwo = []
    for i in range(len(data)):
        listOne.append(data[i][0])
        listTwo.append(data[i][1])
    return (np.array(listOne), np.array(listTwo))

def parse(file):
    file_list = []
    file_names = []
    with open(file, 'r') as f:
        for line in f:
            label = 1 if "down" in line else 0
            img = parse_img('gestures/' + line.strip())
            file_list.append([img, label])
            file_names.append(line.strip())
    return file_list, file_names

def parse_img(fn):
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


train = 'downgesture_train.list'
test = 'downgesture_test.list'



tData, _ = parse(train)
trainData = process(tData)
tData, tDataName = parse(test)
testData = process(tData)

val = trainData[0]
labels = trainData[1]

val_test = testData[0]
label_test = testData[1]

nn = MLPClassifier(hidden_layer_sizes=100, activation='logistic', learning_rate_init=0.1, max_iter=1000)
nn.fit(val, labels)

predictedLabels = nn.predict(val_test)
pp.pp(classification_report(label_test, predictedLabels))
trainLabel = nn.predict(val)
pp.pp(classification_report(labels, trainLabel))

