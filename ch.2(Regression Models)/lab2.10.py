# Import needed packages for regression
# Your code here
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler


# Input standardized feature values for a sample instance
carat = float(input())
table = float(input())

diamonds = pd.read_csv('diamonds.csv')

# Define input and output features
X = diamonds[['carat', 'table']]
y = diamonds[['price']]

# Scale the input features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Initialize a model using elastic net regression with a regularization strength of 6, and l1_ratio=0.4
elastic_net_model = ElasticNet(alpha = 6.0, l1_ratio = 0.4)

# Fit the elastic net model to the input and output features
elastic_net_model = elastic_net_model.fit(X,y)

# Get estimated intercept weight
intercept = elastic_net_model.intercept_
print('Intercept is', np.round(intercept, 3))

# Get estimated weights for carat and table features
coefficients = elastic_net_model.coef_
print('Weights for carat and table features are', np.round(coefficients, 3))

# Predict the price of a diamond with the user-input carat and table values
X_new = np.array([[carat, table]])  # Reshape the input as a 2D array
prediction = elastic_net_model.predict(X_new)
print('Predicted price is', np.array([round(prediction[0], 2)]))
