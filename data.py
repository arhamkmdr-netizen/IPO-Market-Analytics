import pandas as pd

df=pd.read_excel('EXPORT (2).xlsx')
print(df.head())
print(df.info())
print(df.describe())