# Import pandas and numpy
import numpy as np # liniear algebra
import pandas as pd # data processing
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier

import os

# This is only to check where the data is
"""
for dirname, _, filenames in os.walk('.\Data'):
    for filename in filenames:
        print(os.path.join(dirname,filename))
"""

# Load the data
df = pd.read_csv('Data/train.csv') # In case you have a vairable you can use pd.DataFrame() to load the data

# print(df.loc[0]) # Prints first row
# print(df.tail()) # Prints header and last row
#print(df)
print(df.info())
# print(df.duplicated()) # Check for duplicates in the DataFrame

print("###############################")

def sex(x):
    if x == 'male':
        return 1
    else:
        return 0

df["Sex"].dropna(inplace=True)
df["Sex"] = df["Sex"].apply(sex)
df.astype({'Sex' : int})
print(df.info())
df["Age"].dropna(inplace=True)
df["Age"].plot(kind="hist")
#plt.show()

print(df.corr())

df.select_dtypes(include=np.number).plot(kind="barh")
plt.figure(figsize=(12,10))
sbn.heatmap(df.corr(),annot=True,cmap="magma",fmt='.2f')
plt.show()

#Scale 
#X, y = df.iloc[:,]
model = ExtraTreesClassifier()
#model.fit(X,y)
#print(df.head(10))