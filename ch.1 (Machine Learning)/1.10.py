import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import KBinsDiscretizer

homes = pd.read_csv('homes.csv')
priceFloor = homes[['Price', 'Floor']]
school = homes[['School']]

# Define a standardization scaler to transform values
# Your code here
standard_scale = StandardScaler ()

# Apply scaler to the priceFloor data
scaled = standard_scale.fit_transform(priceFloor)

homes_standardized = pd.DataFrame(scaled, columns=['Price','Floor'])
print('Standardized data: \n', homes_standardized)

# Define a normalization scaler to transform values
normalization_scale = MinMaxScaler()

# Apply scaler to the priceFloor data
normalized = normalization_scale.fit_transform(priceFloor)

homes_normalized = pd.DataFrame(normalized, columns=['Price','Floor'])
print('Normalized data: \n', homes_normalized)

# Define the OrdinalEncoder() function
ordinal_encoder = OrdinalEncoder()
# Create a dataframe of the ordinal encoder function fit to the school data, with the column labeled encoding
# Join the new column to the school data
school_encoded = ordinal_encoder.fit_transform(school)
school_encoded_df = pd.DataFrame(school_encoded, columns=['encoding'])
school_with_encoding = pd.concat([school, school_encoded_df], axis=1)


print('Encoded data: \n', school_with_encoding)

# Create a discretizer with equal weights and 3 bins
discretizer_eqwidth = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')

# Fit the discretizer to the Floor feature from the priceFloor data
binned_floor = discretizer_eqwidth.fit_transform(priceFloor[['Floor']])

# Reshape binned_floor to (76, 1) - automatically handled by the fit_transform output
# No need to reshape as fit_transform already returns the correct shape
# Just ensure you use it directly or reshape if needed
binned_floor_reshaped = binned_floor.reshape(-1, 1)

# Access bin edges
bin_edges = discretizer_eqwidth.bin_edges_[0]

# Output the binned floor values and the bin edges
#print('Binned Floor Reshaped to (76, 1): \n', binned_floor_reshaped)
print('Bin widths: \n', bin_edges)
