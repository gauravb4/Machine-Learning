import numpy as np
import sys
import time
# import matplotlib.pyplot as plt
import random

# Layer class that represent each layer in the neural network
class Layer:
    def __init__(self, input_size, output_size, rate, relu):
        self.relu = relu
        if self.relu == True:
            pass
        else:
            # Xavier initialization for each layer
            self.rate = rate
            self.bias = np.array(np.zeros(output_size))

            scale = np.sqrt(2/(input_size + output_size))
            self.weight = np.zeros((input_size, output_size))
            for i in range(input_size):
                for j in range(output_size):
                    self.weight[i][j] = np.random.normal(0.0, scale)    

    def forward_single(self, input):
        if self.relu == True:
            return np.maximum(0, input)

        # Single forward pass with affine transformation
        weightedSum = np.dot(input, self.weight)
        weightedSum += self.bias
        return weightedSum

    def backward_single(self, input, grad_out):
        if self.relu == True:
            return grad_out * (input > 0)

        # Single backward propogation, compute weight and bias gradient
        grad_in = np.dot(grad_out, self.weight.T)

        self.bias -= self.rate * grad_out.mean(axis=0) * input.shape[1]
        self.weight -= self.rate * np.dot(input.T, grad_out)  

        return grad_in

class NN:
    def __init__(self, files, testMode=False):     
        self.testMode = testMode
        train_image, train_label, test_image, test_label = self.parse(files)
        self.train_image = train_image
        self.train_label = train_label
        self.test_image = test_image
        self.test_label = test_label
        self.network = []
        
    # Initialize the hyperparameters (model, learning rate, batch size, epochs)
    def hyper_param_init(self, layers, rate, b_size, epochs):
        self.network = []
        for i in layers:
            if i[2] == True:
                self.network.append(Layer(i[0], i[1], rate, True))
            else:
                self.network.append(Layer(i[0], i[1], rate, False))
        self.b_size = b_size
        self.epochs = epochs
        self.rate = rate

    # Parse all the files specified by command line
    def parse(self, files, t_label_file = "test_label.csv"):
        if self.testMode == True:
            print("Parsing start...")
            start = time.time()

        train_image, train_label, test_image, test_label = ([] for i in range(4))
        with open(files[0]) as fp:
            for l in fp.readlines():
                data = l.strip().split(',')
                train_image.append([int(i)/255.0 for i in data])
        with open(files[1]) as fp:
            for l in fp.readlines():
                data = int(l.strip())
                train_label.append(data)
        with open(files[2]) as fp:
            for l in fp.readlines():
                data = l.strip().split(',')
                test_image.append([int(i)/255.0 for i in data])

        # Read the test labels if doing testing, skip for grading
        if self.testMode == True:
            with open(t_label_file) as fp:
                for l in fp.readlines():
                    data = int(l.strip())
                    test_label.append(data)

        train_image = np.array(train_image)
        train_label = np.array(train_label)
        test_image = np.array(test_image)
        test_label = np.array(test_label)

        if self.testMode == True:
            print("Parsing Complete, time elapsed: " + str(time.time() - start))

        return train_image, train_label, test_image, test_label

    # Loss function based on log-softmax
    def loss_func(self, output, labels):
        loss = output[np.arange(len(output)), labels]
        loss = np.log(np.sum(np.exp(output), axis=-1)) - loss

        grad_loss = np.zeros(output.shape)
        grad_loss[np.arange(len(output)),labels] = 1       
        exp = np.exp(output)
        softmax = exp/exp.sum(keepdims=True, axis=-1)
        grad_loss = (softmax - grad_loss) / len(output)

        return loss, grad_loss

    def train_single(self, data, labels):
        # Calculate forward pass
        forward_pass_result = []
        input = data
        for i in range(len(self.network)):
            currLayer = self.network[i]
            weightedSum = currLayer.forward_single(input)
            forward_pass_result.append(weightedSum)
            input = forward_pass_result[-1]
        
        # Compute Loss function
        loss, grad_loss = self.loss_func(forward_pass_result[-1], labels)

        # Perform backward propogation
        layer_inputs = [data]
        for i in range(len(forward_pass_result)):
            layer_inputs.append(forward_pass_result[i])
        for i in reversed(range(len(self.network))):
            grad_loss = self.network[i].backward_single(layer_inputs[i], grad_loss)

        return np.mean(loss)

    def make_prediction(self, output_file):
        # Perform forward pass with the test images
        forward_pass_result = []
        input = self.test_image
        for i in range(len(self.network)):
            currLayer = self.network[i]
            weightedSum = currLayer.forward_single(input)
            forward_pass_result.append(weightedSum)
            input = forward_pass_result[-1]
        
        # Make prediction using argmax and otuput to file
        prediction = forward_pass_result[-1].argmax(axis=-1)
        with open(output_file, "w") as fp:
            for i in range(len(prediction)):
                fp.write(str(prediction[i]) + "\n")
        
    # split the data into chunks of b_size
    def batch(self, data):
        return np.array_split(data, self.b_size)

    def train(self, data, labels, test_data, test_label):
        # Convert data into batches
        data_batches = self.batch(data)
        label_batches = self.batch(labels)
        log = []

        # Go through each epoch
        for e in range(self.epochs):
            # Randomly "shuffle" the train data
            rand = list(range(len(data_batches)))
            random.shuffle(rand)

            # Train each of the shuffled batches
            for i in range(len(data_batches)):
                self.train_single(data_batches[rand[i]], label_batches[rand[i]])

            # Logging for testing purposes only
            if self.testMode == True:
                forward_pass_result = []
                input = test_data
                for i in range(len(self.network)):
                    currLayer = self.network[i]
                    weightedSum = currLayer.forward_single(input)
                    forward_pass_result.append(weightedSum)
                    input = forward_pass_result[-1]
                test_prediction = forward_pass_result[-1].argmax(axis=-1)
                log.append(np.mean(test_prediction == test_label))

        # Output for testing purposes only
        if self.testMode == True:
            print("Training done, Val accuracy=%f" % (log[-1]))
        return log

    # Initialize the hyperparameters then start training
    def start_train(self, epochs, layers, b_size, rate):
        self.hyper_param_init(layers, rate, b_size, epochs)
        if self.testMode == True:
            print("Training with Rate=%f B_size=%d Epochs=%d" % (self.rate, self.b_size, self.epochs))
        return self.train(self.train_image, self.train_label, self.test_image, self.test_label)

def main(argv):
    # Read all the input files, have default value just in case
    testMode = False
    files = ["train_image.csv", "train_label.csv", "test_image.csv"]
    if len(argv) == 3 :
        files = argv

    # The neural network layers
    layers = [
        [784, 200, False],
        [-1, -1, True],
        [200, 10, False]
    ]

    # Testing purposes only
    if testMode == True:
        all_logs = []
        MLP = NN(files, testMode)

        for i in range(1):
            start = time.time()
            print("********************************")

            log = MLP.start_train(75, layers, 25, 0.2)

            print("Training Complete, time elapsed: " + str(time.time() - start))
            print("********************************\n")

            all_logs.append(log)

        # for i in range(len(all_logs)):
        #     plt.plot(all_logs[i], label=str(i))
        # plt.legend(loc='best')
        # plt.grid()
        # plt.show()
    # For HW submission, train the NN then output to test_predictions.csv
    else :
        MLP = NN(files, testMode)
        MLP.start_train(75, layers, 25, 0.21)
        MLP.make_prediction("test_predictions.csv")

if __name__ == '__main__':
    main(sys.argv[1:])