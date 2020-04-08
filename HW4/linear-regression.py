import numpy as np 
import pprint as pp

f = open("linear-regression.txt")
fl = f.readlines()
points = []
intercepts = []
for line in fl:
    l = line.strip().split(",")
    points.append(np.array([float(1.0) ,float(l[0]), float(l[1])]))
    intercepts.append(float(l[2]))
points = np.array(points)
intercepts = np.array(intercepts)
transpose = points.T
inverse = np.linalg.inv(np.matmul(transpose, points))
weights = np.matmul(np.matmul(inverse, transpose), intercepts)
pp.pp(weights)


