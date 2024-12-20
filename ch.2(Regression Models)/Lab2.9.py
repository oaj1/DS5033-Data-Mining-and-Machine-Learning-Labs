#The diamonds dataset contains the price, cut, color, and other characteristics of a sample of nearly 54,000 diamonds. This data can be used to predict the price of a diamond based on its characteristics. Use sklearn's LinearRegression() function to predict the price of a diamond from the diamond's carat and table values.

#Import needed packages for regression.
#Initialize and fit a multiple linear regression model.
#Get the estimated intercept weight.
#Get the estimated weights of the carat and table features.
#Predict the price of a diamond with the user-input carat and table values.

# Import needed packages for regression
# Your code here
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


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

# Initialize a multiple linear regression model
linear_model = LinearRegression()

# Fit the multiple linear regression model to the input and output features
linear_model = linear_model.fit(X,y)

# Get estimated intercept weight
intercept = linear_model.intercept_
print('Intercept is', round(intercept, 3))

# Get estimated weights for carat and table features
coefficients = linear_model.coef_
print('Weights for carat and table features are', np.round(coefficients, 3))

# Predict the price of a diamond with the user-input carat and table values
X_new = np.array([[carat, table]])  # Reshape the input as a 2D array
prediction = linear_model.predict(X_new)
print('Predicted price is',np.array([round(prediction[0], 2)]))
