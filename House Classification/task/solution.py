import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from category_encoders import TargetEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('house_class.csv')

X = df[['Area', 'Room', 'Lon', 'Lat', 'Zip_area', 'Zip_loc']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=X['Zip_loc'].values)

encoder = OneHotEncoder(drop='first', sparse_output=False)
encoder.fit(X_train[['Zip_area', 'Zip_loc', 'Room']])

X_train_transformed = pd.DataFrame(encoder.transform(X_train[['Zip_area', 'Zip_loc', 'Room']]), index=X_train.index)
X_test_transformed = pd.DataFrame(encoder.transform(X_test[['Zip_area', 'Zip_loc', 'Room']]), index=X_test.index)

X_train_final = X_train[['Area', 'Lon', 'Lat']].join(X_train_transformed)
X_test_final = X_test[['Area', 'Lon', 'Lat']].join(X_test_transformed)

X_train_final.columns = X_train_final.columns.astype(str)
X_test_final.columns = X_test_final.columns.astype(str)


ordinal_encoder = OrdinalEncoder()
ordinal_encoder.fit(X_train[['Zip_area', 'Zip_loc', 'Room']])

X_train_encoded = pd.DataFrame(ordinal_encoder.transform(X_train[['Zip_area', 'Zip_loc', 'Room']]),
                               columns=['Zip_area_encoded', 'Zip_loc_encoded', 'Room_encoded'],
                               index=X_train.index)

X_test_encoded = pd.DataFrame(ordinal_encoder.transform(X_test[['Zip_area', 'Zip_loc', 'Room']]),
                              columns=['Zip_area_encoded', 'Zip_loc_encoded', 'Room_encoded'],
                              index=X_test.index)

X_train_final = X_train[['Area', 'Lon', 'Lat']].join(X_train_encoded)
X_test_final = X_test[['Area', 'Lon', 'Lat']].join(X_test_encoded)

target_encoder = TargetEncoder(cols=['Zip_area', 'Room', 'Zip_loc'])
target_encoder.fit(X_train, y_train)

X_train_encoded = target_encoder.transform(X_train)
X_test_encoded = target_encoder.transform(X_test)




model = DecisionTreeClassifier(
    criterion='entropy',
    max_features=3,
    splitter='best',
    max_depth=6,
    min_samples_split=4,
    random_state=3
)

model.fit(X_train_encoded, y_train)

y_pred = model.predict(X_test_encoded)

accuracy = accuracy_score(y_test, y_pred)

print(round(accuracy, 4))