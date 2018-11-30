import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

test_data = pd.read_csv('test.csv')
test_X = test_data.select_dtypes(exclude=['object'])
test_X = test_X.drop(['Id'], axis=1)

train_data = pd.read_csv('train.csv')
train_y = train_data.SalePrice
train_X= train_data.drop(['SalePrice'], axis=1)
train_X = train_X.select_dtypes(exclude=['object'])
train_X = train_X.drop(['Id'], axis=1)

my_imputer = SimpleImputer()

train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.transform(test_X)

model = XGBRegressor(n_estimators=7950, learning_rate=0.0005, n_jobs=4)
model.fit(train_X, train_y)

predictions = model.predict(test_X)

output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': predictions})

output.to_csv('submission_1.csv', index=False)