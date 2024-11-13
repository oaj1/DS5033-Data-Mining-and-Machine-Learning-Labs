
#The diamonds dataset contains the price, cut, color, and other characteristics of a sample of nearly 54,000 diamonds. This data can be used to predict the price of a diamond based on its characteristics. Use sklearn's KNeighborsRegressor() function to predict the price of a diamond from the diamond's carat and table values.

#Import needed packages for regression.
#Initialize and fit a k-nearest neighbor regression model using a Euclidean distance metric and k=12.
#Predict the price of a diamond with the user-input carat and table values.
#Find the distances and indices of the 12 nearest neighbors for the user-input instance

# Import needed packages for regression
import pandas as pd #for dataframe
import numpy as np #for math
from sklearn.neighbors import KNeighborsRegressor   # For regression tasks

# Silence warning from sklearn
import warnings
warnings.filterwarnings('ignore')

# Input feature values for a sample instance
carat = float(input())
table = float(input())

diamonds = pd.read_csv('diamonds.csv')

# Define input and output features
X = diamonds[['carat', 'table']]
y = diamonds['price']

# Initialize a k-nearest neighbors regression model using a Euclidean distance and k=12 
kNNModel = KNeighborsRegressor(n_neighbors=12, metric='euclidean')

# Fit the kNN regression model to the input and output features
kNNModel = kNNModel.fit(X,y)

# Create array with new carat and table values
Xnew = [[carat, table]]

# Predict the price of a diamond with the user-input carat and table values
prediction = kNNModel.predict(Xnew)
print('Predicted price is', np.round(prediction, 2))

# Find the distances and indices of the 12 nearest neighbors for the new instance
neighbors = kNNModel.kneighbors(Xnew)
print('Distances and indices of the 12 nearest neighbors are', neighbors)