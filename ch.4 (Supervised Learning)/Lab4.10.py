# Import needed packages for classification
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
# Import packages for evaluation
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load the dataset
skySurvey = pd.read_csv('SDSS.csv')

# Create a new feature from u - g
skySurvey['u_g'] = skySurvey['u'] - skySurvey['g']

# Create dataframe X with features redshift and u_g
X = skySurvey[['redshift','u_g']]

# Create dataframe y with feature class
y = skySurvey[['class']]

np.random.seed(42)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Initialize model with k=3
skySurveyKnn = KNeighborsClassifier(n_neighbors=3)

# Fit model using X_train and y_train
skySurveyKnn = skySurveyKnn.fit(X_train,y_train.values.ravel())
#skySurveyKnn = skySurveyKnn.fit(X_train,y_train)

# Find the predicted classes for X_test
y_pred = skySurveyKnn.predict(X_test)

# Calculate accuracy score
score = metrics.accuracy_score(y_test, y_pred)

# Print accuracy score
print('Accuracy score is ', end="")
print('%.3f' % score)

# Print confusion matrix 
#print(metrics.confusion_matrix(y_test, y_pred))
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
transposed_conf_matrix = conf_matrix.T  # Correct way to transpose the matrix
print(transposed_conf_matrix)

"""

output:
Accuracy score is 0.984
[[1463   12    3]
 [   5  274    0]
 [  29    0 1214]]

 Interpretation:

Each row represents the true class (actual labels), and each column represents the predicted class (predicted labels).

Row 1 (True Class: 0):

1463 samples were correctly classified as class 0 (True Positives for class 0).
12 samples were incorrectly classified as class 1 (False Positives for class 0).
3 samples were incorrectly classified as class 2 (False Positives for class 0).
Row 2 (True Class: 1):

5 samples were incorrectly classified as class 0 (False Negatives for class 1).
274 samples were correctly classified as class 1 (True Positives for class 1).
0 samples were incorrectly classified as class 2 (False Positives for class 1).
Row 3 (True Class: 2):

29 samples were incorrectly classified as class 0 (False Negatives for class 2).
0 samples were incorrectly classified as class 1 (False Negatives for class 2).
1214 samples were correctly classified as class 2 (True Positives for class 2).

"""