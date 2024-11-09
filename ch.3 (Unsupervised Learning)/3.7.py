#The healthy_lifestyle dataset contains information on lifestyle measures such as amount of sunshine, pollution, and happiness levels for 44 major cities around the world. Apply k-means clustering to the cities' number of hours of sunshine and happiness levels.
#Import the needed packages for clustering.
#Initialize and fit a k-means clustering model using sklearn's Kmeans() function. Use the user-defined number of clusters, init='random', n_init=10, random_state=123, and algorithm='elkan'.
#Find the cluster centroids and inertia.

# Import needed packages
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

healthy = pd.read_csv('healthy_lifestyle.csv')

# Input the number of clusters
number = int(input())

# Define input features
X = healthy[['sunshine_hours', 'happiness_levels']]

# Use StandardScaler() to standardize input features
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns=['sunshine_hours', 'happiness_levels'])
X = X.dropna()

# Initialize a k-means clustering algorithm with a user-defined number of clusters, init='random', n_init=10, 
# random_state=123, and algorithm='elkan'
# Your code here
kmModel = KMeans(n_clusters=number, init ='random', n_init =10, random_state =123)

# Fit the algorithm to the input features
kmModel = kmModel.fit(X)

# Find and print the cluster centroids
centroid = kmModel.cluster_centers_
print("Centroids:", np.round(centroid,4))

# Find and print the cluster inertia
inertia = kmModel.inertia_
print("Inertia:", np.round(inertia,4))