import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from xgboost import XGBRegressor

test_data = pd.read_csv('test.csv')
X_test_final = test_data.drop(['Id'], axis=1)

train_data = pd.read_csv('train.csv')
y_train_final = train_data.SalePrice
X_train_final = train_data.drop(['Id', 'SalePrice'], axis=1)


X_train_final = pd.get_dummies(X_train_final)
X_test_final = pd.get_dummies(X_test_final)

X_train_final, X_test_final = X_train_final.align(X_test_final, join='left', axis=1)

cols_with_missing_values = [col for col in X_train_final.columns
                            if X_train_final[col].isnull().any()]

for col in cols_with_missing_values:
    X_train_final[col + '_was_missing'] = X_train_final[col].isnull()
    X_test_final[col + '_was_missing'] = X_test_final[col].isnull()

pipeline = make_pipeline(SimpleImputer(), XGBRegressor(n_estimators=1000, learning_rate=0.065))


pipeline.fit(X_train_final, y_train_final)

predictions = pipeline.predict(X_test_final)
output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': predictions})

output.to_csv('submission_1.csv', index=False)