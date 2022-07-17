import time
from operator import itemgetter
import numpy as np
from sklearn.metrics import silhouette_score, adjusted_rand_score

def select_index(x, y):
    row = min(x, y)
    column = max(x, y)
    return {'row': row, 'column': column}

def K_Means(distances, K, n_samples, data):
    # If we have more centers than elements, return the dataset itself
    if K > n_samples:
        return data
    
    # Starting with a random center 
    centroids = np.random.choice(n_samples, 1).tolist()
    # print('First center:', centroids[0])
    
    # Select s that maximize dist(s, C)
    while len(centroids) < K:
        candidates = []
        for idx_row in range(n_samples):
            s_is_centroid = [idx_row == centroid for centroid in centroids]
            if not any(s_is_centroid):
                options = []
                for centroid in centroids:
                    indexes = select_index(idx_row, centroid)
                    options.append(distances[indexes['row']][indexes['column']])
                most_distant_centroid = min(options)
                candidates.append({'distance': most_distant_centroid, 'index': idx_row})

        if len(candidates) > 0:
            max_item = max(candidates, key=itemgetter('distance'))
            s = max_item['index']
            centroids.append(s)
        
    return centroids

# Calcule clusters
def calc_clusters(centers, distances, n_samples, K):
    # Initialize clusters with empty array
    clusters = [[] for _ in range(K)]
    labels = []
    for idx_row in range(n_samples):
        candidates = []
        for idx, center in enumerate(centers):
            indexes = select_index(idx_row, center)
            candidates.append({'distance': distances[indexes['row']][indexes['column']], 'center_index': idx})
        closest_centroid = min(candidates, key=itemgetter('distance'))
        clusters[closest_centroid['center_index']].append(idx_row)
        labels.append(closest_centroid['center_index'])

    return {'clusters': clusters, 'labels': labels}

def calc_radius(labels, distances, centers):
    candidates = []
    for center_idx, center in enumerate(centers):
        options = []
        for idx, cluster in enumerate(labels):
            if center_idx == cluster:
                indexes = select_index(center, idx)
                options.append(distances[indexes['row']][indexes['column']])
        if len(options) > 0:
            candidates.append(max(options))
        
    return max(candidates)
 
def kmeans_calc(matrix, n_samples, K, data):
    C = K_Means(matrix, K, n_samples, data)
    print('Centers', C)

    result = calc_clusters(C, matrix, n_samples, K)
    radius = calc_radius(result['labels'], matrix, C)

    return {'radius': radius, 'labels': result['labels']}
    
    
def execute_calculations(max_iterations, data, optimal_labels, distances, n_samples, K): 
    all_radius = []
    all_silhouettes = []
    all_rand_scores = []
    all_times = []
    
    for _ in range(max_iterations):
        start_time = time.time()
        result = kmeans_calc(distances, n_samples, K, data)
        end_time = round(time.time() - start_time, 5)
        silhouette = round(silhouette_score(data, result['labels']), 5)
        rand_score = round(adjusted_rand_score(optimal_labels, result['labels']), 5)
        
        all_radius.append(result['radius'])
        all_silhouettes.append(silhouette)
        all_rand_scores.append(rand_score)
        all_times.append(end_time)
        
        print('Radius', result['radius'])
        # print('Labels', result['labels'])
        print('Silhouette', silhouette)
        print('Adjusted Rand', rand_score)
        print('Time', end_time)
        print('----------------')
    
    print('Average radius of ', max_iterations, ' executions: ', round(sum(all_radius)/len(all_radius), 5))
    print('Average silhouette of ', max_iterations, ' executions: ', round(sum(all_silhouettes)/len(all_silhouettes), 5))
    print('Average rand of ', max_iterations, ' executions: ', round(sum(all_rand_scores)/len(all_rand_scores), 5))
    print('Average time of ', max_iterations, ' executions: ', round(sum(all_times)/len(all_times), 5))
    print('Standard deviation', round(np.std(all_radius), 5))
    print('----------------')
    