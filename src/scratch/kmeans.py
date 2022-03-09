import random
from statistics import mean

class Cluster:
    
    def __init__(self, centroid: int) -> None:
        self.centroid = centroid
        self.points = []
    
    def __str__(self):
        return f"Centroid: {self.centroid}\nPoints: {self.points}"
    
    def __eq__(self, other):
        return self.centroid == other.centroid
    
    @property
    def mean(self):
        if len(self.points) == 0:
            return None
        return mean(self.points)
    
    def distance_from(self, x: int) -> int:
        return abs(self.centroid - x)
    
    @staticmethod
    def get_nearest(x: int, clusters: list):
        distances = [cluster.distance_from(x) for cluster in clusters]
        return clusters[distances.index(min(distances))]


def get_clusters(centroids: list, values: list, k: int) -> list:
    
    clusters = []
    for value in sorted(centroids):
        clusters.append(Cluster(value))
    
    for value in values:
        cluster = Cluster.get_nearest(value, clusters)
        cluster.points.append(value)
    
    return clusters


def k_means_clustering(values: list, k: int, seed: int = 0) -> list:
    
    random.seed(seed)
    
    random_values = random.choices(values, k=k)
    clusters = get_clusters(random_values, values, k)
    
    while True:
        cluster_means = [cluster.mean for cluster in clusters]
        new_clusters = get_clusters(cluster_means, values, k)
        if all([current == new for current, new in zip(clusters, new_clusters)]):
            break
        clusters = new_clusters
    
    return clusters
