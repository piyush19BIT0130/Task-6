import pandas as pd
housing = pd.read_csv("data.csv")
# housing.head()
# housing.info()
# housing['CHAS'].value_counts()
# housing.describe()
import numpy as np
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state = 42) # so the same data is splited always 
print(f"Rows in train set: {len(train_set)}\nRows in test set:{len(test_set)}")
from sklearn.model_selection import StratifiedShuffleSplit # so that the data is similarly distributed between the test data and train data
split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
# strat_test_set['CHAS'].value_counts()
# strat_train_set['CHAS'].value_counts()
housing = strat_train_set.copy()
corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False) # correlation matrix with MEDV
# for obtaining graphs
# from pandas.plotting import scatter_matrix
# attributes = ['MEDV','RM','ZN','LSTAT']
# scatter_matrix(housing[attributes], figsize = (12,8))
# housing.plot(kind = 'scatter', x="RM", y='MEDV', alpha = 0.8)
# housing['TAXRM'] = housing['TAX']/housing['RM'] # creating a new variable
# housing.shape
# corr_matrix = housing.corr()
# corr_matrix['MEDV'].sort_values(ascending=False)
# housing.plot(kind = 'scatter', x="TAXRM", y='MEDV', alpha = 0.8)
housing = strat_train_set.drop('MEDV', axis = 1)
housing_labels = strat_train_set["MEDV"].copy()

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = 'median')
imputer.fit(housing)
imputer.statistics_
imputer.statistics_.shape
X = imputer.transform(housing)


housing_tr = pd.DataFrame(X, columns=housing.columns)
housing_tr.describe()
# pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy = "median")),
    # .............. we can add as many as we want .........
    ("std_scalar", StandardScaler())
])
housing_num_tr = my_pipeline.fit_transform(housing)
# housing_num_tr.shape

# selecting desired model 

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#model = LinearRegression()
#model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
prepared_data = my_pipeline.transform(some_data)
model.predict(prepared_data)
# list(some_labels)
from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels,housing_predictions)
rmse = np.sqrt(mse)
#print("rmse:",rmse)
#using the better evaluation tachnique - cross-validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring = "neg_mean_squared_error", cv = 10)
rmse_scores = np.sqrt(-scores)
# print(rmse_scores)
def print_scores(scores):
    print("Score (RMSE): ", scores)
    print("Mean of RMSE: ", scores.mean())
    print("Standard devation of RMSE values: ", scores.std())
print_scores(rmse_scores)

#saving the model
from joblib import dump,load
dump(model, "Dragon.joblib")

#testing the data on test data
X_test = strat_test_set.drop("MEDV",axis =1)
Y_test = strat_test_set['MEDV'].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
# print(final_predictions,list(Y_test))
print("final_rmse on test data:", final_rmse)
prepared_data[0]

#using the model
from joblib import dump, load
import numpy as np
model = load('Dragon.joblib')
features = np.array([[-0.43942006,  3.12628155, -1.12165014, -0.27288841, -1.42262747, -0.24141041, -1.31238772,  2.61111401, -1.0016859 , -0.5778192 , -0.97491834,  0.41164221, -0.86091034]])
print("prediction on single array:", model.predict(features))
