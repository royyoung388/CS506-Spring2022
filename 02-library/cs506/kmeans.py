from collections import defaultdict
from math import inf
import random
import csv
import numpy as np
from .sim import euclidean_dist


def point_avg(points):
    """
    Accepts a list of points, each with the same number of dimensions.
    (points can have more dimensions than 2)
    
    Returns a new point which is the center of all the points.
    """
    return np.mean(points, axis=0)


def update_centers(dataset, assignments):
    """
    Accepts a dataset and a list of assignments; the indexes 
    of both lists correspond to each other.
    Compute the center for each of the assigned groups.
    Return `k` centers in a list
    """
    cluster = np.unique(assignments)
    center = []
    for i in cluster:
        data = []
        for j in range(len(dataset)):
            if assignments[j] == cluster:
                data.append(dataset[j])
        data = np.array(data)
        center.append(point_avg(data))
    return center


def assign_points(data_points, centers):
    """
    """
    assignments = []
    for point in data_points:
        shortest = inf  # positive infinity
        shortest_index = 0
        for i in range(len(centers)):
            val = distance(point, centers[i])
            if val < shortest:
                shortest = val
                shortest_index = i
        assignments.append(shortest_index)
    return assignments


def distance(a, b):
    """
    Returns the Euclidean distance between a and b
    """
    return euclidean_dist(a, b)


def distance_squared(a, b):
    return distance(a, b) ** 2


def generate_k(dataset, k):
    """
    Given `data_set`, which is an array of arrays,
    return a random set of k points from the data_set
    """
    return dataset[np.random.choice(range(len(dataset)), k, replace=False)]


def cost_function(clustering):
    cost = 0
    for dataset in clustering:
        dist = 0
        for d1 in dataset:
            for d2 in dataset:
                dist = distance(d1, d2)
        cost += dist
    return cost


def generate_k_pp(dataset, k):
    """
    Given `data_set`, which is an array of arrays,
    return a random set of k points from the data_set
    where points are picked with a probability proportional
    to their distance as per kmeans pp
    """
    centriod = generate_k(dataset, k)
    probs = []
    for d in dataset:
        p = 0
        for c in centriod:
            p += distance_squared(c, d)
        probs.append(p)

    centroid_pp = []
    for i in range(k):
        index = np.argmax(probs)
        centroid_pp.append(dataset[index])
        probs[index] = np.min(probs)
    return np.array(centroid_pp)


def _do_lloyds_algo(dataset, k_points):
    assignments = assign_points(dataset, k_points)
    old_assignments = None
    while assignments != old_assignments:
        new_centers = update_centers(dataset, assignments)
        old_assignments = assignments
        assignments = assign_points(dataset, new_centers)
    clustering = defaultdict(list)
    for assignment, point in zip(assignments, dataset):
        clustering[assignment].append(point)
    return clustering


def k_means(dataset, k):
    if k not in range(1, len(dataset) + 1):
        raise ValueError("lengths must be in [1, len(dataset)]")

    k_points = generate_k(dataset, k)
    return _do_lloyds_algo(dataset, k_points)


def k_means_pp(dataset, k):
    if k not in range(1, len(dataset) + 1):
        raise ValueError("lengths must be in [1, len(dataset)]")

    k_points = generate_k_pp(dataset, k)
    return _do_lloyds_algo(dataset, k_points)
