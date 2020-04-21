import numpy as np 
import sklearn.svm
import pprint

def parseDoc(filename):
        f = open(filename, 'r')
        clusterData = f.readlines()
        points = []
        label = []
        for line in clusterData:
            points.append((float(line.strip().split(",")[0]),float(line.strip().split(",")[1])))
            label.append(int(line.strip().split(",")[2]))
        
        return np.array(points), np.array(label)

linstep = 'linsep.txt'
nonlinstep = 'nonlinsep.txt'

points, labels = parseDoc(linstep)
q = open("SkleanResults.txt", "w+")

q.write("Linear\n")
t = sklearn.svm.SVC(kernel="linear")
t.fit(points, labels) 
q.write("Support Vectors\n")
pprint.pprint(t.support_vectors_)
pprint.pprint(t.support_vectors_, stream=q)
q.write("Coefficient\n")
pprint.pprint(t.coef_)
pprint.pprint(t.coef_, stream=q)
q.write("Intercept\n")
pprint.pprint(t.intercept_)
pprint.pprint(t.intercept_, stream=q)

q.write("\n\n")
points, labels = parseDoc(nonlinstep)

q.write("Nonlinear\n")
t2 = sklearn.svm.SVC(kernel="rbf")
t2.fit(points, labels)
q.write("Support Vectors\n")
pprint.pprint(t2.support_vectors_)
pprint.pprint(t2.support_vectors_, stream=q)
q.write("Intercept\n")
pprint.pprint(t2.intercept_)
pprint.pprint(t2.intercept_, stream=q)