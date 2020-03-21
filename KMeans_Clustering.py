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
import matplotlib.pyplot as plt

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
def plotResults(assignments):
    i = len(assignments)
    idx_0 = np.asarray(np.where(iris.target == 0)).reshape(50)
    idx_1 = np.asarray(np.where(iris.target == 1)).reshape(50)
    idx_2 = np.asarray(np.where(iris.target == 2)).reshape(50)
    
    data_0 = iris.data[idx_0,:]
    
    data_1 = iris.data[idx_1,:]
    
    data_2 = iris.data[idx_2,:]
    
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.plot(x[np.where(assignments[i-1] == 0),0], x[np.where(assignments[i-1] == 0),1],'ko')
    ax1.plot(x[np.where(assignments[i-1] == 1),0], x[np.where(assignments[i-1] == 1),1],'r+')
    ax1.plot(x[np.where(assignments[i-1] == 2),0], x[np.where(assignments[i-1] == 2),1],'bo')
    ax1.set_xlabel('Petal length')
    ax1.set_ylabel('Petal width')
    ax1.set_title('KMeans')
    ax2.plot(data_0[:,2],data_0[:,3],'ko',label="Versicolour")
    ax2.plot(data_1[:,2],data_1[:,3],'r+',label="Versicolour")
    ax2.plot(data_2[:,2],data_2[:,3],'bo',label="Virginica")
    ax2.set_xlabel('Petal length')
    ax2.set_ylabel('Petal width')
    ax2.set_title('True Classification')
    plt.show()

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
    
    #Plot KMeans against true assignments
    plotResults(assignments)
    
    print("Cluster centers from scratch:")
    print(centroids[len(centroids)-1])
    
    #Compare to sklearn kmeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(x)
    print("Cluster Centers (sklearn):")
    print(kmeans.cluster_centers_)
