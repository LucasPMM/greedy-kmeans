from math import *
import numpy as np

def p_root(value, root):
    root = 1 / float(root)
    return round ((value) ** (root), 3)

def minkowski_distance(x, y, p=1):
    return (p_root(sum(pow(abs(a-b), p) for a, b in zip(x, y)), p))


def matrix_of_distances(p, n_samples, rows):
    distances = np.zeros((n_samples,n_samples))

    for diag in range(0, n_samples):
        for row in range(0, n_samples-diag):
            col = row + diag
            if row == col:
                distances[row][col] = 0.0
            else:
                sample = rows[row]
                target = rows[col]
                distances[row][col] = minkowski_distance(sample, target, p)

    return distances
