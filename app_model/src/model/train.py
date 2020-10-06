import numpy as np
import pandas as pd  

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier

import mlflow
import mlflow.sklearn

try:
    import cPickle as pickle
except ImportError:
    import pickle

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


output = "/model_data/train_output"

df = pd.read_csv('/data/googleplaystore.csv')  # load data set
df.info()

df = df.drop([
        "App",
        "Reviews",
        "Size",
        "Installs",
        "Type",
        "Content Rating",
        "Genres",
        "Last Updated",
        "Current Ver",
        "Android Ver",
        "Price"
    ], axis=1)

df.info()

labels = df[["Category"]]
labels["Category"] = labels["Category"].astype("category")
labels.info()

df = df.drop("Category", axis=1)
df = df.fillna(0)
df.info()


# Build training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    df, labels["Category"].cat.codes, test_size=0.33
)

print("Labels")
print(pd.get_dummies(labels))

print(y_train)

# Set hyperparameters
estimators = 100
jobs = 3

with mlflow.start_run():
    clf = RandomForestClassifier(n_estimators=estimators, 
                                 n_jobs=jobs)
    clf.fit(X_train, y_train)

    predicted = clf.predict(X_test)

    (rmse, mae, r2) = eval_metrics(y_test, predicted)

    mlflow.log_param("estimators", estimators)
    mlflow.log_param("jobs", jobs)

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)

    mlflow.sklearn.log_model(clf, "clf")


with open(output, 'wb') as fd:
    pickle.dump(clf, fd)


# X = data.iloc[:, 0].values.reshape(-1, 1)  # values converts it into a numpy array
# Y = data.iloc[:, 1].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
# linear_regressor = LinearRegression()  # create object for the class
# linear_regressor.fit(X, Y)  # perform linear regression
# Y_pred = linear_regressor.predict(X)  # make predictions