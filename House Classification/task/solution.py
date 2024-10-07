import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('house_class.csv')

X = df[['Area', 'Room', 'Lon', 'Lat', 'Zip_area', 'Zip_loc']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=X['Zip_loc'].values)

zip_loc_counts = X_train['Zip_loc'].value_counts().to_dict()

print(zip_loc_counts)
