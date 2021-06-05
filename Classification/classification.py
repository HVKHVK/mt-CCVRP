import pandas as pd

data = pd.DataFrame()
for i in range(1, 6):
    for j in [9, 16, 25, 36, 49, 64, 81, 100]:
        data_tmp = pd.read_csv('../Output_{}/output_{}.csv'.format(i, j))
        data = pd.concat([data, data_tmp])
data.drop('Unnamed: 0', axis='columns', inplace=True)
data.reset_index(inplace=True)
data.drop('index', axis='columns', inplace=True)
data.to_csv("combined.csv")
data_sorted = data.sort_values(["node_number","drone_number","capacity_of_drones","arc_weight"], ascending=True)
data_sorted.to_csv("combined_sorted.csv")

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.neural_network import MLPClassifier as MLP

from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.metrics import mean_squared_error, accuracy_score

models = 'MLP'

if models == 'KNR':

    data = data.replace(-1, 0)

    data_values = data[['node_number', 'drone_number', 'capacity_of_drones', 'arc_weight']]
    data_results = data['result']

    x_train, x_test, y_train, y_test = train_test_split(data_values, data_results, test_size=.25, random_state=10)

    search_params = [
        {'n_neighbors': [3, 4, 5, 6, 7], 'leaf_size': [10, 15, 20, 25, 30, 35], 'p': [1, 2]},
    ]

    model = GridSearchCV(
        KNR(),
        search_params,
        cv=5,
        n_jobs=-1,
        scoring='neg_root_mean_squared_error',
        verbose=4)

    model.fit(x_train, y_train)
    print(model.best_estimator_)

    predictions = model.predict(x_test)
    tt = y_test.to_numpy()
    acc = mean_squared_error(tt, predictions)
    print(acc)

if models == 'KNC':

    data = data.replace(-1, 0)

    data_values = data[['node_number', 'drone_number', 'capacity_of_drones', 'arc_weight']]
    data_results = data['result']

    x_train, x_test, y_train, y_test = train_test_split(data_values, data_results, test_size=.25, random_state=10)

    search_params = [
        {'n_neighbors': [4, 5, 6, 7]},
    ]

    model = GridSearchCV(
        KNC(),
        search_params,
        cv=5,
        n_jobs=-1,
        scoring='accuracy',
        verbose=4)

    model.fit(x_train, y_train)
    print(model.best_estimator_)

    predictions = model.predict(x_test)
    tt = y_test.to_numpy()
    acc = accuracy_score(tt, predictions)
    print(acc)


if models == 'RFC':

    data = data.replace(-1, 0)

    data_values = data[['node_number', 'drone_number', 'capacity_of_drones', 'arc_weight']]
    data_results = data['result']

    x_train, x_test, y_train, y_test = train_test_split(data_values, data_results, test_size=.25, random_state=10)

    search_params = [
        {'n_estimators': [4, 5]},
    ]

    model = GridSearchCV(
        RFC(),
        search_params,
        cv=5,
        n_jobs=-1,
        scoring='accuracy',
        verbose=4)

    model.fit(x_train, y_train)
    print(model.best_estimator_)

    predictions = model.predict(x_test)
    tt = y_test.to_numpy()
    acc = accuracy_score(tt, predictions)
    print(acc)

if models == 'MLP':

    data = data.replace(-1, 0)

    data_values = data[['node_number', 'drone_number', 'capacity_of_drones', 'arc_weight']]
    data_results = data['result']

    x_train, x_test, y_train, y_test = train_test_split(data_values, data_results, test_size=.25, random_state=10)

    search_params = [
        {'hidden_layer_sizes': [3, 4, 5], 'learning_rate': ['constant', 'adaptive'], 'solver': ['sgd', 'adam']},
    ]

    model = GridSearchCV(
        MLP(),
        search_params,
        cv=5,
        n_jobs=-1,
        scoring='accuracy',
        verbose=4)

    model.fit(x_train, y_train)
    print(model.best_estimator_)

    predictions = model.predict(x_test)
    tt = y_test.to_numpy()
    acc = accuracy_score(tt, predictions)
    print(acc)

import joblib

joblib_file = models+".pkl"
joblib.dump(model.best_estimator_, "models/"+joblib_file)
