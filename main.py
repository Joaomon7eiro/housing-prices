import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

train_data = pd.read_csv('train.csv', sep=',')
test_data = pd.read_csv('test.csv', sep=',')

y = train_data.SalePrice

X = train_data.drop(["SalePrice"], axis=1)
numeric_X = X.select_dtypes(exclude=['object'])

train_X, test_X, train_y, test_y = train_test_split(numeric_X, y, random_state=1)

my_imputer = SimpleImputer()

imputed_train_X = my_imputer.fit_transform(train_X)
imputed_test_X = my_imputer.transform(test_X)

model = RandomForestRegressor(random_state=1)
model.fit(imputed_train_X, train_y)

predictions = model.predict(imputed_test_X)
mae = mean_absolute_error(test_y, predictions)

print(mae)