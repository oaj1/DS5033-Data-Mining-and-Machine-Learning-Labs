#The dataset fg_attempts.csv contains data on field goal distance and outcome (whether or not the kick was successful) during the 2012 NFL seasons.

#Create a dataframe X with input features Distance and ScoreDiffPreKick.
#Create a dataframe y with output feature Outcome.
#Flatten y into an array called yArray using np.ravel().
#Initialize a logistic regression model using LogisticRegression().
#Fit the model to the input and flattened output features in X and yArray.
#Create a new dataframe XNew with user-input values for Distance and ScoreDiffPreKick.
#Use the fitted logistic regression model to predict the outcome from the new data.
#Determine the accuracy of a fitted logistic regression classification model.

# Import the necessary modules
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Read in the file fg_attempt.csv
field_goal = pd.read_csv('fg_attempt.csv')

# Create a dataframe X containing 'Distance' and 'ScoreDiffPreKick'
X = field_goal[['Distance','ScoreDiffPreKick']]
# Create a dataframe y containing 'Outcome'
y = field_goal['Outcome']

# Flatten y into an array
yArray =np.ravel(y)

# Initialize a LogisticRegression() model
logisticModel = LogisticRegression()

# Fit the model
logisticModel.fit(X,yArray)

# Input feature values for a sample instance
Distance = float(input())
ScoreDiffPreKick = float(input())

# Create a new dataframe with user-input Distance and ScoreDiffPreKick
XNew = field_goal[['Distance','ScoreDiffPreKick']]

# Predict the outcome from the new data
pred = logisticModel.predict(XNew)
print(np.array([pred[0]]))

# Determine the accuracy of the model logisticModel
score = logisticModel.score(X,y)
print(score)
