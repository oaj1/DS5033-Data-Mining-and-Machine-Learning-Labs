#The heart dataset contains 13 health-related attributes from 303 patients 
#and one attribute denoting whether or not the patient has heart disease. 
#Using the file heart.csv and scikit-learn's LinearSVC() function, fit a support vector classifier 
#to predict whether a patient has heart disease based on other health attributes.
#Import the correct packages and functions.
#Split the data into 75% training data and 25% testing data. Set random_state=123.
#Initialize and fit a support vector classifier with C=0.2, a maximum of 500 iterations, and random_state=123.
#Print the model weights.



# Import the necessary packages
# Your code here
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

heart = pd.read_csv('heart.csv')

# Input features: thalach and age
X = heart[['thalach', 'age']]

# Output feature: target
y = heart['target']

# Create training and testing data with 75% training data and 25% testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

# Scale the input features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize a support vector classifier with C=0.2 and a maximum of 500 iterations
SVC_Model = LinearSVC(C=0.2, max_iter=500)
# Fit the support vector classifier according to the training data
SVC_Model = SVC_Model.fit(X_train,y_train)
# Evaluate model on testing data
score = SVC_Model.score(X_test, np.ravel(y_test))
print(np.round(score, 3))

# Print the model weights
# w0
# Print the model weights in the desired format
print('w0:', np.round(SVC_Model.intercept_, 3))  # Display intercept as a list
print('w1 and w2:', np.round(SVC_Model.coef_, 3))  # Display coefficients as a matrix

"""
output:
0.671
w0: [0.125]
w1 and w2: [[ 0.39  -0.084]]

Interpreation
Accuracy (0.671): The model correctly classifies 67.1% of the test data.
Intercept (0.125): This is the model's bias term, helping adjust the decision boundary.
Coefficients:
w1 (0.39): For thalach, higher maximum heart rate slightly increases the likelihood of heart disease.
w2 (-0.084): For age, older patients have a slightly decreased likelihood of heart disease, assuming all other factors remain constant.
These results suggest that, for this model, both age and maximum heart rate (thalach) contribute to the decision-making process, with a slightly stronger influence from thalach.

"""