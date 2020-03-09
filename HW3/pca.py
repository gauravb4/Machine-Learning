import numpy as np
import pprint as pp

def parseData(filename):
    f = open(filename, 'r')
    d = f.readlines()
    data = []
    for line in d:
        ls = line.strip().split("\t")
        data.append((float(ls[0]), float(ls[1]), float(ls[2])))
    npd = np.array(data)
    mean = np.mean(npd, axis=0)  
    return npd-mean

file = "pca-data.txt"
data = parseData(file)
cov = np.cov(data, rowvar = False)
eigenvalue, eigenvector = np.linalg.eig(cov)
principal_components = eigenvector[:, :2]
pp.pprint(principal_components)