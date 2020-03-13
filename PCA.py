# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 16:01:03 2020

@author: daniel arnold
"""

#Imports
import numpy as np
import numpy.linalg as LA
from sklearn import datasets
from sklearn import decomposition
import matplotlib.pyplot as plt

class PCA_scratch:
    
    def __init__(self, data, epsilon):
        
        self.X = data
        self.n, self.p = np.shape(data)
        self.u = np.mean(self.X,axis=0).reshape(self.p,1) #column mean of data
        self.epsilon = epsilon
        self.pcs = []
        self.eigenvecs = []
        
    def fit(self):
        
        print("Running PCA from scratch")
        
        #subtract mean from dataset X
        B = self.X - np.dot(np.ones((self.n,1)), np.transpose(self.u))
        
        #compute singular values and eigenvectors 
        for i in range(0,self.p):
            #power iteration for largest principal component
            sing_val, ev = self.power_iteration(B)
            self.pcs.append(sing_val)
            self.eigenvecs.append(ev)
            
            #subtract principal component
            B = B - np.dot(B, np.dot(ev, np.transpose(ev)))
    
    def getSingularValues(self):
        return self.pcs
    
    def getEigenVectors(self):
        return self.eigenvecs
        
    def project(self, num_dimensions):
        #Project data into lower dimensions
        Wp = np.array(self.eigenvecs).reshape(self.p,self.p).transpose()
        
        Wl = Wp[:,:num_dimensions]
        
        B = self.X - np.dot(np.ones((self.n,1)), np.transpose(self.u))
        return B.dot(Wl)
                
    def power_iteration(self, X):
    
        r = np.random.rand(self.p,1)
        r = r/LA.norm(r,2)
        
        delta = 100
        
        num_iter = 0
        while(delta > self.epsilon):
            s = np.dot(np.transpose(X), np.dot(X,r))
            
            e_val = np.dot(np.transpose(r),s)
            delta = LA.norm(e_val*r - s, 1)
            r = s/LA.norm(s,2)
            num_iter += 1
            
        return np.sqrt(e_val), r

##############################################################################    

if __name__ == "__main__":
    
    #load the Iris dataset
    iris = datasets.load_iris()
    y = iris.target
    X = iris.data
    
    #PCA from scratch (first 2 principal components):
    pca_scratch = PCA_scratch(X, 1e-5)
    pca_scratch.fit()
    #project the data onto the first 2 principle components
    X_transformed_scratch = pca_scratch.project(2)
    
    #Compare to sklearn
    pca = decomposition.PCA(n_components=2)
    pca.fit(X)
    X_transformed_pca = pca.transform(X)
    
    #compare transformed data from both methods
    delta = LA.norm(X_transformed_pca - X_transformed_scratch,2)
    print("norm of difference of PCA and scratch transforms:", delta)
    
    plt.plot(X_transformed_scratch[:,0],X_transformed_scratch[:,1],'bo')
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.title('Data projected onto first two PCs (scratch)')
    plt.show()
    
    plt.plot(X_transformed_pca[:,0],X_transformed_pca[:,1],'bo')
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.title('Data projected onto first two PCs (sklearn)')
    plt.show()