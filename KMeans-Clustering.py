# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 01:02:27 2019

@author: PERSONALISE NOTEBOOK
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 21:52:08 2019

@author: PERSONALISE NOTEBOOK
"""

# KMeans
#%reset -f
# Import library 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[: , [3,4]].values # The columns who take effect with goals

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range (1,11) :
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, random_state = 6)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
# at n_cluters equals 5 is the optimum values before the curve climb up

# Apllying k-means to the mall dataset
kmeans = KMeans(n_clusters= 5, init = 'k-means++', max_iter = 300, random_state = 6) # clusters 5 'cause after point 5 the curve climb up
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Careful')
# X[y_kmeans == 0, 0] data X who have rows in y_kmeans = 0 and column in data X is column 0 (first column)
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Standard')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Target')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Careless')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Sensible')
# Plot of the centroid
# cluster center atribute who give you coordinate of the centroid
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of Clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1 - 100)')
plt.legend()
plt.show()











