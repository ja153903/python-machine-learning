import pandas as pd 
from io import StringIO

csv_data = \
    '''A,B,C,D
    1.0,2.0,3.0,4.0
    5.0,6.0,,8.0
    10.0,11.0,12.0,
    '''

df = pd.read_csv(StringIO(csv_data))

print(df.head())

# checks the number of nulls per column
# in this case we find that we have 1 null in col C
# and 1 null in col D
print(df.isnull().sum())

# we can drop NaN values

# axis = 0 allows us to drop the row 
df.dropna(axis=0)

# axis = 1 allows us to drop the cols
df.dropna(axis=1)

# only drop the rows where all cols are NaN
df.dropna(how='all')

# drop rows that have less than 4 real values
df.dropna(thresh=4)

# Instead of removing rows/cols which could compromise our
# study, we could instead include values
from sklearn.preprocessing import Imputer

# we find the mean among the columns
imr = Imputer(missing_values='NaN', strategy='mean', axis=1)

imr = imr.fit(df.values)

imputed_data = imr.transform(df.values)
print(imputed_data)

