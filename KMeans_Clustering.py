# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 15:31:43 2020

@author: daniel arnold
"""

#Imports
import numpy as np
import numpy.linalg as LA
from sklearn.cluster import KMeans
from sklearn import datasets

class KMeans_scratch:
    
    def __init__(self, data, c0, num_centroids, epsilon=1e-4):
        self.c0 = c0
        self.data = data
        self.num_centroids = num_centroids
        self.epsilon = 1e-4
        
        self.centroids = []
        self.centroids.append(c0)
        self.assignments = []
        
    def fit(self):
        
        print("Running KMeans")
        
        delta = 100
        i = 0
    
        while delta > self.epsilon:
    
            self.assignments.append(self.assign_observations_to_cluster(self.centroids[i]))
            self.centroids.append(self.compute_centroids(self.assignments[i]))
    
            delta = LA.norm(self.centroids[i+1] - self.centroids[i], 2)
            i += 1
        
        return self.assignments, self.centroids
    
        #KMeans - Step 1
    def assign_observations_to_cluster(self, centers):
        assigned = []
        for i in range(0,len(self.data)):
            dist = []
            for j in range(0,self.num_centroids):
                #compute euclidean distance
                dist.append(LA.norm(self.data[i] - centers[j], 2))
            assigned.append(np.argmin(dist))
        return np.asarray(assigned)
    
    #KMeans - Step 2
    def compute_centroids(self, assignments):
        centers = []
        for i in range(0,self.num_centroids):
            idxs = np.where(np.asarray(assignments) == i)
            cluster_data = self.data[idxs]
            centers.append(np.mean(cluster_data, axis=0))
        return np.asarray(centers)
    
##############################################################################    

if __name__ == "__main__":
    
    #Load the data
    iris = datasets.load_iris()
    y = iris.target
    x = iris.data[:,2:]
    
    #run KMeans from scratch
    c0 = np.asarray([[1, 0], [2,2], [4,4]]) #initial centroids
    num_clusters = 3
    kmeans = KMeans_scratch(x, c0, num_clusters)
    assignments, centroids = kmeans.fit()
    print("Cluster centers from scratch:")
    print(centroids[len(centroids)-1])
    
    #Compare to sklearn kmeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(x)
    print("Cluster Centers (sklearn):")
    print(kmeans.cluster_centers_)
