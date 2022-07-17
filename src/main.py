#!/usr/bin/env python3
from heapq import nsmallest
import sys
import utils
import distance
import sklearnKmeans
import kmeans

import pandas as pd
from collections import Counter

# Mount output file name
def output_name(input, extension):
    if len(input) == 5:
        return input[4] + extension
    return input[2].split(".")[0] + extension

if __name__ == "__main__":
    input = sys.argv
    if utils.is_input_invalid(input):
        print('Invalid input')
        raise Exception("Ex: ./main.py file.data  P [p value for distance]")

    dataset = input[1]
    p = 1
    if len(input) == 3:
        p = int(input[2])


    # Load data and define K:
    df = pd.read_csv(dataset)
    df = df.fillna(0)
    number_of_columns = len(df.columns)
    list_of_classes = df['class'].to_list()

    # Number of unique classes:
    classes = Counter(list_of_classes).keys()
    K = len(classes)
    print('K=', K)

    optimal_labels = []
    for item in list_of_classes:
        optimal_labels.append(list(classes).index(item))

    # print('optimal_labels', optimal_labels)

    # Remove class from dataframe
    data = df.drop("class", axis=1)
    # data = data.select_dtypes(include='number').apply(zscore)
    # data.head()

    rows = []
    for index, row in data.iterrows():
        rows.append(row.values)
    n_samples, n_features = data.shape

    # Matrix of distances
    matrix = distance.matrix_of_distances(p, n_samples, rows)

    # Calc Kmeans using sklearn 
    sklearnKmeans.sklearn_kmeans(K, data, optimal_labels, rows, p)

    # Greedy kmeans
    kmeans.execute_calculations(30, data, optimal_labels, matrix, n_samples, K)