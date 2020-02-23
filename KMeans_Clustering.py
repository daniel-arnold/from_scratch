# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 15:31:43 2020

@author: daniel arnold, daniel.brian.arnold@gmail.com
"""

#Imports
import numpy as np
import numpy.linalg as LA
from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#Load the data
iris = datasets.load_iris()
y = iris.target
x = iris.data[:,2:]

#KMeans - Step 1
def assign_observations_to_cluster(data, centers, num_centroids):
    assigned = []
    for i in range(0,len(data)):
        dist = []
        for j in range(0,num_centroids):
            #compute euclidean distance
            dist.append(LA.norm(data[i] - centers[j], 2))
        assigned.append(np.argmin(dist))
    return np.asarray(assigned)

#KMeans - Step 2
def compute_centroids(assignments, data, num_centroids):
    centers = []
    for i in range(0,num_centroids):
        idxs = np.where(np.asarray(assignments) == i)
        cluster_data = data[idxs]
        centers.append(np.mean(cluster_data, axis=0))
    return np.asarray(centers)

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

def runKMeans():
    
    print("Running KMeans Clustering for Iris dataset")
    #compute initial centroids
    num_clusters = 3
    
    centroids = []
    #centroids.append(x[np.random.choice(len(x),num_centroids),:])
    centroids.append(np.asarray([[1, 0], [2,2], [4,4]]))
    
    assignments = []
    
    epsilon = 1e-4
    delta = 100
    
    i = 0
    
    while delta > epsilon:
        
        assignments.append(assign_observations_to_cluster(x, centroids[i], num_clusters))
        centroids.append(compute_centroids(assignments[i], x, num_clusters))
        
        delta = LA.norm(centroids[i+1] - centroids[i], 2)
        i += 1
    
    print("number of iterations:", i-1)
    print("Cluster Centers:")
    print(centroids[i-1])
    
    #Compare to sklearn kmeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(x)
    print("Cluster Centers (KMeans - sklearn):")
    print(kmeans.cluster_centers_)
    
    return assignments

if __name__ == "__main__":
    assignments = runKMeans()
    plotResults(assignments) #assumes there are 3 centroids