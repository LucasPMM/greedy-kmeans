import distance
from math import *
import numpy as np
import time
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score


def calc_sklearn_radius(labels, centers, rows, p):
    candidates = []
    for center_idx, center in enumerate(centers):
        options = []
        for idx, cluster in enumerate(labels):
            if center_idx == cluster:
                options.append(distance.minkowski_distance(center, rows[idx], p))
        candidates.append(max(options))
        
    return max(candidates)


def sklearn_kmeans(K, data, optimal_labels, rows, p):
    start_time = time.time()

    kmeans = KMeans(n_clusters=K).fit(data)
    sklearn_labels = kmeans.labels_
    sklearn_centers = kmeans.cluster_centers_

    sklearn_radius = calc_sklearn_radius(sklearn_labels, sklearn_centers, rows, p)

    sklearn_silhouette = round(silhouette_score(data, kmeans.labels_), 5)
    rand_score = round(adjusted_rand_score(optimal_labels, sklearn_labels), 5)

    sklearn_end_time = round(time.time() - start_time, 5)

    print('sklearn radius', sklearn_radius)
    # print('sklearn centers', sklearn_centers)
    # print('sklearn labels', sklearn_labels)
    print('sklearn silhouette', sklearn_silhouette)
    print('sklearn rand index', rand_score)
    print('sklearn time', sklearn_end_time)
