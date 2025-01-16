# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 18:51:57 2024

@author: assafz
"""


# Load modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from statistics import mean
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
import scipy.stats as stats
import matplotlib as mpl
from Consts import *
from utils import *

start_time = timer(None) 

filename_vtime = 'datasets\data'
data_unfiltered = pd.read_pickle(filename_vtime)
# Data is filtered to remove gait segment with obstacles in gait mat
data = data_unfiltered[data_unfiltered[walk_col] < obstacle_num] 
data.columns = columns_names

filename_vtime_orig = 'datasets\data_orig'
data_orig_unfiltered = pd.read_pickle(filename_vtime_orig)
# Filter data to remove walking during obstacles
data_orig = data_orig_unfiltered[data_orig_unfiltered[walk_col] < obstacle_num]
reg_columns_names = columns_names[:walk_col+1] + [columns_names[label_col]]
data_orig.columns = reg_columns_names
# Select best feature names
BestFeatNames = data_orig.columns[best_feat_num]

BestFeatNames = data.columns[best_feat_num]

X_train_list = []
y_train_list=[]
X_test_list=[]
y_test_list=[]
y_pred_list=[]
Train_mse_list=[]
Test_mse_list=[]
pearson_coef_list = []
ICC_list = []
prop_list =[]
#%%
# Iterate through each split
for split in range(iterations):
    print(f"Training models {split + 1}/{iterations}")
    # Split data into training and testing sets
    X_train, y_train, X_test, y_test, groups = train_test_split(data, seed=seed_list[split])
    X_train_orig, y_train_orig, X_test_orig, y_test_orig, groups_orig = train_test_split(data_orig, seed=seed_list[split])

    # Save the properties of the test group
    prop = X_test[PropCols]
    
    # Select best features for training and testing sets
    X_train = X_train.loc[:, BestFeatNames]
    X_test = X_test.loc[:, BestFeatNames]
    X_train_orig = X_train_orig.loc[:, BestFeatNames]
    X_test_orig = X_test_orig.loc[:, BestFeatNames]
    
    # Append split data to respective lists
    X_train_list.append(X_train)
    y_train_list.append(y_train)
    X_test_list.append(X_test)
    y_test_list.append(np.array(y_test))
    prop_list.append(prop)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_orig = scaler.fit_transform(X_train_orig)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Load pre-trained XGBoost model
    model_xgb = xgb.Booster()
    model_path = r'models\model_orig' + str(seed_list[split]) + ".json"
    model_xgb.load_model(model_path)
    
    # Initialize lists to store model results for the current split
    models = []
    names = []
    Test_mse_vec = []
    pearson_coef_vec = []
    ICC_vec = []
    
    # Add models to the list
    models.append(('LR', LinearRegression()))
    models.append(('RT', DecisionTreeRegressor(max_depth=max_depth)))
    models.append(('SVR', SVR(C=0.5)))
    models.append(('KNN', KNeighborsRegressor(n_neighbors=num_neighbors)))
    models.append(('Gradboost', GradientBoostingRegressor(n_estimators=num_estimators, max_depth=max_depth)))
    models.append(('XGB', model_xgb))
    models.append(('IP', None))
    
    # Evaluate each model
    for name, model in models:
        if name == 'IP':
            IP_StepLength = prop['IP_StepLength']
            y_pred = np.array(IP_StepLength)
            remove_IP_idx = np.where(y_pred == -999)[0]
            y_pred = np.delete(y_pred, remove_IP_idx)
            y_test_IP = np.delete(np.array(y_test), remove_IP_idx)
            test_mse = mean_squared_error(y_test_IP, y_pred, squared=False)
            pearson_coefficient, p_value = stats.pearsonr(y_test_IP, y_pred)
            ICC = Calc_ICC(y_test_IP, y_pred)
            ICC.set_index('Type', inplace=True)
        elif name != 'XGB':
            regressor = model
            regressor.fit(X_train, y_train)
            y_pred = regressor.predict(X_test)
            test_mse = mean_squared_error(y_test, y_pred, squared=False)
            pearson_coefficient, p_value = stats.pearsonr(y_test, y_pred)
            ICC = Calc_ICC(y_test, y_pred)
            ICC.set_index('Type', inplace=True)
        else:
            X_test = xgb.DMatrix(pd.DataFrame(X_test))
            X_train = xgb.DMatrix(pd.DataFrame(X_train))
            regressor = model
            y_pred = model.predict(X_test)
            test_mse = mean_squared_error(y_test, y_pred, squared=False)
            pearson_coefficient, p_value = stats.pearsonr(y_test, y_pred)
            ICC = Calc_ICC(y_test, y_pred)
            ICC.set_index('Type', inplace=True)
        
        # Append model evaluation metrics to respective lists
        Test_mse_vec.append(test_mse)
        pearson_coef_vec.append(pearson_coefficient)
        ICC_vec.append(ICC.loc['ICC2k', 'ICC'])
        names.append(name)
        
        # Print model evaluation metrics
        msg = "%s: %f" % (name, test_mse)
        print(msg)
    
    # Append results of the current split to main lists
    Test_mse_list.append(Test_mse_vec)
    pearson_coef_list.append(pearson_coef_vec)
    ICC_list.append(ICC_vec)
    
    # Print progress message
    iter_msg = "%i %s" % (split + 1, '/5')
    print(iter_msg)
#%%
LR_mse, RT_mse, SVR_mse, KNN_mse, GB_mse, XGB_mse, IP_mse = ([] for i in range(7))
LR_p_coef, RT_p_coef, SVR_p_coef, KNN_p_coef, GB_p_coef, XGB_p_coef, IP_p_coef = ([] for i in range(7))
LR_ICC, RT_ICC, SVR_ICC, KNN_ICC, GB_ICC, XGB_ICC, IP_ICC= ([] for i in range(7))

for i in range(len(Test_mse_list)):
    LR_mse.append(Test_mse_list[i][LR_order])
    RT_mse.append(Test_mse_list[i][RT_order])
    SVR_mse.append(Test_mse_list[i][SVR_order])
    KNN_mse.append(Test_mse_list[i][KNN_order])
    GB_mse.append(Test_mse_list[i][GB_order])
    XGB_mse.append(Test_mse_list[i][XGB_order])
    IP_mse.append(Test_mse_list[i][IP_order])

    LR_p_coef.append(pearson_coef_list[i][LR_order])
    RT_p_coef.append(pearson_coef_list[i][RT_order])
    SVR_p_coef.append(pearson_coef_list[i][SVR_order])
    KNN_p_coef.append(pearson_coef_list[i][KNN_order])
    GB_p_coef.append(pearson_coef_list[i][GB_order])
    XGB_p_coef.append(pearson_coef_list[i][XGB_order])
    IP_p_coef.append(pearson_coef_list[i][IP_order])
    
    LR_ICC.append(ICC_list[i][LR_order])
    RT_ICC.append(ICC_list[i][RT_order])
    SVR_ICC.append(ICC_list[i][SVR_order])
    KNN_ICC.append(ICC_list[i][KNN_order])
    GB_ICC.append(ICC_list[i][GB_order])
    XGB_ICC.append(ICC_list[i][XGB_order])
    IP_ICC.append(ICC_list[i][IP_order])
    
#%%
mse_vec = [LR_mse, RT_mse, SVR_mse, KNN_mse, GB_mse, XGB_mse]

# Calculate means and standard deviations using list comprehensions
mean_vec = [np.mean(data) for data in mse_vec]
std_vec = [np.std(data) for data in mse_vec]

# set up plot
fig, ax = plt.subplots()
# add axis labels and title
ax.set_ylabel('RMSE [cm]', fontsize=18, fontweight='bold')
ax.tick_params(axis='both', labelsize=18)
ax.grid(which="major", axis='x', color='#DAD8D7', alpha=0.5, zorder=1)
ax.grid(which="major", axis='y', color='#DAD8D7', alpha=0.5, zorder=1)
plt.subplots_adjust(left=None, bottom=0.2, right=None, top=0.85, wspace=None, hspace=None)
fig.patch.set_facecolor('white')
# Colours - Choose the extreme colours of the colour map
colours = ["#2196f3", "#bbdefb"]

# Colormap - Build the colour maps
cmap = mpl.colors.LinearSegmentedColormap.from_list("colour_map", colours, N=256)
norm = mpl.colors.Normalize(np.min(mean_vec), np.max(mean_vec)) # linearly normalizes data into the [0.0, 1.0] interval
# create bar plot
ax.bar(['LR','RT', 'SVR', 'KNN', 'GradientBoosting', 'XGB'], mean_vec,color=cmap(norm(mean_vec)), yerr=std_vec, capsize=10)

for i, mean in enumerate(mean_vec):
    ax.text(i, mean+std_vec[i]+0.1, f"{mean:.2f}", ha='center', fontsize=16)
# show plot
plt.show()

