import numpy as np 
import pprint as pp
import copy
import math
# import cvxopt
from cvxopt import matrix, solvers

class Point:
    
    def __init__(self, x, y, l):
        self.x = x
        self.y = y
        self.label = l

    def equal(self, other):
        return (self.x == other.x and self.y == other.y)
    
    def distance(self, other):
        return np.sqrt((((self.x - other.x) * (self.x - other.x)) + ((self.y - other.y) * (self.y - other.y))))

    def __repr__(self):
        return ("(X: " + str(self.x) + ", Y: " + str(self.y) + ")\n\t" + "Label: " + str(self.label))

class SupportVectorMachine:

    def __init__(self, type):
        filename = "linsep.txt" if type == "linear" else "nonlinsep.txt"
        self.points = self.parseDoc(filename)
        self.type = type
        self.fit()

    def linKernelGen(self, pos, i, j):
        return np.dot(pos[i], pos[j])

    def nonLinGen(self, pos, i, j):
        exp = 0.0
        for k in range(len(pos[i])):
            exp += (pos[i][k] - pos[j][k])**2
        return math.exp(-0.005 * exp)

    def calcWeight(self, cols):
        self.weights = np.zeros(cols)
        for i in range(len(self.alphas)):
            self.weights += self.sv[i] * self.sv_ls[i] * self.alphas[i]

    def calcIntercept(self):
        self.intercept = self.sv_ls[0] - np.dot(self.weights, self.sv[0])
        # intercept = 0
        # for i in range(len(self.alphas)):
        #     intercept = intercept + self.sv_ls[i].astype(float)
        #     intercept -= np.sum(self.alphas * self.sv_ls.astype(float) * K[np.arange(len(a))[i], sv])
        # self.intercept = intercept / len(self.alphas)

    def fit(self):
        pos = []
        ls = []
        for point in self.points:
            pos.append((point.x, point.y))
            ls.append(point.label)
        
        pos = np.array(pos)
        ls = np.array(ls)
        K = np.zeros((len(pos), len(pos)))
        for i in range(len(pos)):
            for j in range(len(pos)):
                if self.type == "linear":
                    K[i, j] = self.linKernelGen(pos, i, j)
                else:
                    K[i, j] = self.nonLinGen(pos, i, j)

        P = matrix(np.outer(ls, ls) * K)
        q = matrix(np.ones(len(pos)) * -1)
        A = matrix(ls, (1, len(pos)), 'd')
        b = matrix(0.0)

        G = matrix(np.diag(np.ones(len(pos)) * -1))
        h = matrix(np.zeros(len(pos)))

        alphas = np.array(solvers.qp(P, q, G, h, A, b)['x'])
        alphas = alphas.reshape(1,len(pos))[0]

        support_vectors = np.where(alphas > 1e-5)[0]

        self.alphas = alphas[support_vectors]
        self.sv = pos[support_vectors]
        self.sv_ls = ls[support_vectors]

        self.calcWeight(len(pos[0]))
        self.calcIntercept(K)
                
        return

    def parseDoc(self, filename):
        f = open(filename, 'r')
        clusterData = f.readlines()
        pointList = []
        for line in clusterData:
            pointList.append(Point(float(line.strip().split(",")[0]), \
                                   float(line.strip().split(",")[1]), \
                                   int(line.strip().split(",")[2])))
        
        return np.array(pointList)

linstep = 'linstep.txt'
nonlinstep = 'nonlinstep.txt'

print("************* Linear *************")
svm = SupportVectorMachine("linear")
print("W:")
print(svm.weights)
print("B:")
print(svm.intercept)
print("Support vectors")
print(svm.sv)
print("**********************************")
print("")

print("*********** Non-Linear ***********")
svm_n = SupportVectorMachine("nonlinear")
print("W:")
print(svm_n.weights)
print("B:")
print(svm_n.intercept)
print("Support vectors")
print(svm_n.sv)
print("**********************************")
