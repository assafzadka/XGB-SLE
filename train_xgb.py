# Load modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from datetime import datetime
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from statistics import mean
from itertools import islice
from Consts import *
from utils import *

# Start timer
start_time = timer(None)

# Load data
filename_vtime_orig = 'datasets\data_orig'
data_orig_unfiltered = pd.read_pickle(filename_vtime_orig)
# Filter data to remove walking during obstacles
data_orig = data_orig_unfiltered[data_orig_unfiltered[walk_col] < obstacle_num]
reg_columns_names = columns_names[:walk_col+1] + [columns_names[label_col]]
data_orig.columns = reg_columns_names
# Select best feature names
BestFeatNames = data_orig.columns[best_feat_num]

# Initialize the XGBoost regressor
xgb = xgb.XGBRegressor(n_estimators=500, booster='gbtree')
# Initialize Group K-Fold cross-validator
kf = GroupKFold(n_splits=folds)

# Initialize lists to store training and testing data, predictions, and metrics
X_train_list = []
y_train_list = []
X_test_list = []
y_test_list = []
y_pred_list = []
Train_mse_list = []
Test_mse_list = []
best_params = []
prop_list = []
PropCols = ['ID', 'Visit', 'Walk']

#%% Iterate over the specified number of splits
for split in range(iterations):
    print(f"Training model {split + 1}/{iterations}")
    # Split the data into training and testing sets
    X_train, y_train, X_test, y_test, groups = train_test_split(data_orig, seed=seed_list[split])
    prop = X_test[PropCols]
    # Select only the best features
    X_train = X_train.loc[:, BestFeatNames]
    X_test = X_test.loc[:, BestFeatNames]
    X_train_list.append(X_train)
    y_train_list.append(y_train)
    X_test_list.append(X_test)
    y_test_list.append(np.array(y_test))
    prop_list.append(prop)
    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Perform randomized search for hyperparameter tuning
    random_search = RandomizedSearchCV(
        xgb, param_distributions=params, n_iter=param_comb, 
        scoring='neg_mean_squared_error', n_jobs=4, 
        cv=kf.split(X_train, y_train, groups=groups), 
        verbose=3, random_state=1001
    )
    random_search.fit(X_train, y_train, groups=groups)

    # Print the best score and estimator from the search
    print('\n Best score for %d-fold search with %d parameter combinations:' % (folds, param_comb))
    print(random_search.best_score_ * (-1))
    print('\n Best estimator:')
    print(random_search.best_estimator_)

    # Save the best model
    model = random_search.best_estimator_
    model.save_model("models\model_orig" + str(seed_list[split]) + '.json')
    best_params.append(random_search.best_estimator_)

    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    y_pred_list.append(y_pred)

    RMSE = mean_squared_error(y_pred, y_test, squared=False)
    RMSE_train = mean_squared_error(y_pred_train, y_train, squared=False)
    Test_mse_list.append(RMSE)
    Train_mse_list.append(RMSE_train)
