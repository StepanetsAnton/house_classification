import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from category_encoders import TargetEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

df = pd.read_csv('house_class.csv')

X = df[['Area', 'Room', 'Lon', 'Lat', 'Zip_area', 'Zip_loc']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=X['Zip_loc'].values)

model_params = {
    'criterion': 'entropy',
    'max_features': 3,
    'splitter': 'best',
    'max_depth': 6,
    'min_samples_split': 4,
    'random_state': 3
}

onehot_encoder = OneHotEncoder(drop='first', sparse_output=False)
onehot_encoder.fit(X_train[['Zip_area', 'Zip_loc', 'Room']])

X_train_onehot = pd.DataFrame(onehot_encoder.transform(X_train[['Zip_area', 'Zip_loc', 'Room']]), index=X_train.index)
X_test_onehot = pd.DataFrame(onehot_encoder.transform(X_test[['Zip_area', 'Zip_loc', 'Room']]), index=X_test.index)

X_train_onehot.columns = X_train_onehot.columns.astype(str)
X_test_onehot.columns = X_test_onehot.columns.astype(str)

X_train_onehot_final = X_train[['Area', 'Lon', 'Lat']].join(X_train_onehot)
X_test_onehot_final = X_test[['Area', 'Lon', 'Lat']].join(X_test_onehot)

model_onehot = DecisionTreeClassifier(**model_params)
model_onehot.fit(X_train_onehot_final, y_train)
y_pred_onehot = model_onehot.predict(X_test_onehot_final)

report_onehot = classification_report(y_test, y_pred_onehot, output_dict=True)
f1_onehot = report_onehot['macro avg']['f1-score']
print(f"OneHotEncoder:{round(f1_onehot, 2)}")


ordinal_encoder = OrdinalEncoder()
ordinal_encoder.fit(X_train[['Zip_area', 'Zip_loc', 'Room']])

X_train_ordinal = pd.DataFrame(ordinal_encoder.transform(X_train[['Zip_area', 'Zip_loc', 'Room']]),
                               columns=['Zip_area_encoded', 'Zip_loc_encoded', 'Room_encoded'],
                               index=X_train.index)

X_test_ordinal = pd.DataFrame(ordinal_encoder.transform(X_test[['Zip_area', 'Zip_loc', 'Room']]),
                              columns=['Zip_area_encoded', 'Zip_loc_encoded', 'Room_encoded'],
                              index=X_test.index)

X_train_ordinal.columns = X_train_ordinal.columns.astype(str)
X_test_ordinal.columns = X_test_ordinal.columns.astype(str)

X_train_ordinal_final = X_train[['Area', 'Lon', 'Lat']].join(X_train_ordinal)
X_test_ordinal_final = X_test[['Area', 'Lon', 'Lat']].join(X_test_ordinal)

model_ordinal = DecisionTreeClassifier(**model_params)
model_ordinal.fit(X_train_ordinal_final, y_train)
y_pred_ordinal = model_ordinal.predict(X_test_ordinal_final)

report_ordinal = classification_report(y_test, y_pred_ordinal, output_dict=True)
f1_ordinal = report_ordinal['macro avg']['f1-score']
print(f"OrdinalEncoder:{round(f1_ordinal, 2)}")


target_encoder = TargetEncoder(cols=['Zip_area', 'Room', 'Zip_loc'])
target_encoder.fit(X_train, y_train)

X_train_target = target_encoder.transform(X_train)
X_test_target = target_encoder.transform(X_test)

X_train_target.columns = X_train_target.columns.astype(str)
X_test_target.columns = X_test_target.columns.astype(str)

model_target = DecisionTreeClassifier(**model_params)
model_target.fit(X_train_target, y_train)
y_pred_target = model_target.predict(X_test_target)

report_target = classification_report(y_test, y_pred_target, output_dict=True)
f1_target = report_target['macro avg']['f1-score']
print(f"TargetEncoder:{round(f1_target, 2)}")