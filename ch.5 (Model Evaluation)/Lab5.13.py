#The nbaallelo_slr dataset contains information on 126315 NBA games between 1947 and 2015. The columns report the points made by one team, the Elo rating of that team coming into the game, the Elo rating of the team after the game, and the points made by the opposing team. The Elo rating measures the relative skill of teams in a league.

#The code creates a new column y in the data frame that is the difference between pts and opp_pts.
#Split the data into 70 percent training set and 30 percent testing set using sklearn's train_test_split function. Set random_state=0.
#Store elo_i and y from the training data as the variables X and y.
#The code performs a simple linear regression on X and y.
#Perform 10-fold cross-validation with the default scorer using scikit-learn's cross_val_score function.

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score

nba = pd.read_csv("nbaallelo_slr.csv")

# Create a new column in the data frame that is the difference between pts and opp_pts
nba['y'] = nba['pts'] - nba['opp_pts']

# Split the data into training and test sets
train, test = train_test_split(nba, test_size=0.3,random_state=0)

# Store relevant columns as variables
X = train[['elo_i']]
y = train['y']

# Initialize the linear regression model
SLRModel = LinearRegression()
# Fit the model on X and y
SLRModel.fit(X,y)

# Perform 10-fold cross-validation with the default scorer
tenFoldScores = cross_val_score(SLRModel, X, y, cv=10)
print('The cross-validation scores are', tenFoldScores)