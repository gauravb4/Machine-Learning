import numpy as np
import pprint as pp

def parse(file):
    f = open(file, 'r')
    data = f.readlines()
    data = [[int(line.strip().split("\t")[0]), int(line.strip().split("\t")[1]), int(line.strip().split("\t")[2])] for line in data]

    distances = {}
    points = set()
    for line in data:
        if line[0] not in distances:
            distances[line[0]] = {}
        distances[line[0]][line[1]] = line[2]
        points.add(line[0])
        points.add(line[1])

    return distances, list(points)

def fastmap(distances, points):
    mapping = np.zeros((len(points), 3))
    mapping[:, 0] = points
    pp.pprint(mapping)




filename = "fastmap-data.txt"
distances, ids = parse(filename)
fastmap(distances, ids)