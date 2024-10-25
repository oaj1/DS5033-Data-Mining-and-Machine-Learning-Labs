import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('msleep.csv')

# Create subsets for each imputer
mammalSimple = df[['sleep_rem', 'sleep_cycle', 'awake', 'brainwt']]
mammalKnn = df[['sleep_rem', 'sleep_cycle', 'awake', 'brainwt']]
mammalIter = df[['sleep_rem', 'sleep_cycle', 'awake', 'brainwt']]

# Scale the mammalKnn dataframe using MinMaxScaler()
scaler = MinMaxScaler()
scaler.fit(mammalKnn)
mammalScaled = scaler.transform(mammalKnn)

print('Dataframe with missing data: \n', mammalSimple.sample(n=5, random_state=123))

# Initialize and fit a simple imputer to the mammalSimple dataframe using the most frequent value of each feature
simpleImputer = SimpleImputer(strategy='most_frequent')


# Fill the missing value for each feature in mammalSimple
mammalSimple = pd.DataFrame(simpleImputer.fit_transform(mammalSimple),columns=mammalSimple.columns)
print('Dataframe filled with SimpleImputer: \n', mammalSimple.sample(n=5, random_state=123))

# Initialize a KNN imputer with 4 neighbors and weights assigned by distance
knnImputer = KNNImputer(n_neighbors=4, weights='distance')
mammalScaled_filled = knnImputer.fit_transform(mammalScaled)

# Fit the KNN imputer to the mammalScaled dataframe and fill the missing value for each feature
#mammalScaled = pd.DataFrame(mammalScaled_filled, columns=mammalKnn.columns)
# Define the new column names
new_column_names =['scaled_sleep_rem', 'scaled_sleep_cycle', 'scaled_awake', 'scaled_brainwt']
# Create a new DataFrame with the renamed columns
mammalScaled = pd.DataFrame(mammalScaled_filled, columns=new_column_names)
print('Dataframe filled with KNNImputer: \n', mammalScaled.sample(n=5, random_state=123))

# Initialize and fit an iterative imputer with 100 imputation iterations to the mammalIter dataframe
iterImputer = IterativeImputer(max_iter =100,random_state=0)
mammalIter_filled = iterImputer.fit_transform(mammalIter)

# Fill the missing value for each feature in mammalIter
mammalIter = pd.DataFrame(mammalIter_filled, columns=mammalIter.columns)
print('Dataframe filled with IterativeImputer: \n', mammalIter.sample(n=5, random_state=123))