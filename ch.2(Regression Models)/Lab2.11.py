# Import the necessary modules
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Read in nbaallelo_slr.csv
nba = pd.read_csv('nbaallelo_slr.csv')

# Create a new column in the data frame that is the difference between pts and opp_pts
nba['y'] = nba['pts'] - nba['opp_pts']

# Store relevant columns as variables
X = nba[['elo_i']]
y = nba['y']

# Initialize the linear regression model
SLRModel = LinearRegression()

# Fit the model on X and y
SLRModel.fit(X, y)

# Print the intercept
intercept = SLRModel.intercept_
print(f'The intercept of the linear regression line is {intercept:.3f}. ')

# Print the slope (coefficients for each feature in X)
slope = SLRModel.coef_
print(f'The slope of the linear regression line is {slope[0]:.3f}. ')

# Compute the proportion of variation explained by the linear regression using the score method
score = SLRModel.score(X, y)
print(f'The proportion of variation explained by the linear regression model is {score:.3f}. ')