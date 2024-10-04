import pandas as pd


df = pd.read_csv('house_class.csv')
print(df.shape[0])
print(df.shape[1])
print(df.isna().any().any())
print(df.Room.max())
print(round(df.Area.mean(),1))
print(df['Zip_loc'].nunique())