# The attributes most important to us from a descriptive perspective are wine name, 
# producer, vintage, subregion/region/country, color, 
# wine type, grape, and classification if it exists.

import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('./data_export.csv')

# Print the DataFrame
print(df)
