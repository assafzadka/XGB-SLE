# Load modules
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from statistics import mean
from itertools import islice
from Consts import *
from utils import *


start_time = timer(None) 

avg_num_steps = 3

filename_vtime = 'datasets\data'
filename_vtime_orig = 'datasets\data_orig' # Used to normalize features - data was added during the work so the normalization is used to keep the results consistent
filename_ONPAR = 'datasets\data_ONPAR'
filename_MS = 'datasets\data_MS'

data_unfiltered = pd.read_pickle(filename_vtime)
data = data_unfiltered[data_unfiltered[walk_col] < obstacle_num]
data_orig_unfiltered = pd.read_pickle(filename_vtime_orig)
data_orig = data_orig_unfiltered[data_orig_unfiltered[walk_col] < obstacle_num]
data_ONPAR = pd.read_pickle(filename_ONPAR)
data_MS = pd.read_pickle(filename_MS)

reg_columns_names = columns_names[:walk_col+1] + [columns_names[label_col]]
data_orig.columns = reg_columns_names
data.columns = columns_names
data_MS.columns = columns_names
data_ONPAR.columns = columns_names

ONPAR_ID_filename = 'Patients_data\Onpar_patients.xlsx'
ONPAR_ID_data = pd.read_excel(ONPAR_ID_filename, sheet_name='Data')
ONPAR_ID_data['SubNumber'] = ONPAR_ID_data['SubNumber'].replace('OP', '', regex=True)
ONPAR_ID_data['SubNumber'] = ONPAR_ID_data['SubNumber'].replace(' :\)', '', regex=True)
ONPAR_ID_data['SubNumber'] = pd.to_numeric(ONPAR_ID_data['SubNumber'])
ONPAR_HC = ONPAR_ID_data[ONPAR_ID_data['Group'] == 0]
ONPAR_PD = ONPAR_ID_data[ONPAR_ID_data['Group'] == 1]

ONPAR_ID_HC = np.array(ONPAR_HC['SubNumber'])
ONPAR_ID_PD = np.array(ONPAR_PD['SubNumber'])

MS_ID_filename = 'Patients_data\MS_patients.xlsx'
MS_HC_data = pd.read_excel(MS_ID_filename, sheet_name='HC (2)')
MS_ID_HC = MS_HC_data['Subject_ID'].unique()[2:]  # starts with test001 and nan so I start with the second index
MS_ID_HC = pd.to_numeric(MS_ID_HC)

MS_MS_data = pd.read_excel(MS_ID_filename, sheet_name='MS CMCMvisit 1')
MS_ID_MS = MS_MS_data['Subject_ID'].unique()[2:]  # starts with test001 and nan so I start with the second index
MS_ID_MS = pd.to_numeric(MS_ID_MS)

BestFeatNames = data.columns[best_feat_num]
kf = GroupKFold(n_splits=folds)

# Initialize multiple lists
(Test_mse_list, Test_RA_list, Test_speed_list, RMSE_avg_list, RA_avg_list,
 speed_mse_list_avg, mse_MS_HC_list, RA_MS_HC_list, speed_MS_HC_list,
 RMSE_MS_HC_avg_list, RA_MS_HC_avg_list, mse_MS_MS_list, RA_MS_MS_list, speed_MS_MS_list,
 RMSE_MS_MS_avg_list, RA_MS_MS_avg_list, mse_ONPAR_HC_list, RA_ONPAR_HC_list, speed_ONPAR_HC_list,
 RMSE_ONPAR_HC_avg_list, RA_ONPAR_HC_avg_list, mse_ONPAR_PD_list, RA_ONPAR_PD_list, speed_ONPAR_PD_list,
 RMSE_ONPAR_PD_avg_list, RA_ONPAR_PD_avg_list, ICC_list_test, ICC_list_MS_HC, ICC_list_MS_MS,
 ICC_list_ONPAR_HC, ICC_list_ONPAR_PD) = ([] for _ in range(31))


#%%
for split in range(iterations):
    print(f"Testing {split + 1}/{iterations}")
    # Split the data
    X_train, y_train, X_test, y_test, groups = train_test_split(data, seed=seed_list[split])
    X_train_orig, y_train_orig, X_test_orig, y_test_orig, groups_orig = train_test_split(data_orig, seed=seed_list[split])
    X_MS_HC, y_MS_HC = load_data(data_MS, MS_ID_HC)
    X_MS_MS, y_MS_MS = load_data(data_MS, MS_ID_MS)
    X_ONPAR_HC, y_ONPAR_HC = load_data(data_ONPAR, ONPAR_ID_HC)
    X_ONPAR_PD, y_ONPAR_PD = load_data(data_ONPAR, ONPAR_ID_PD)

    # Extract step time
    time_test = X_test['Step time']
    time_ONPAR_HC = X_ONPAR_HC['Step time']
    time_ONPAR_PD = X_ONPAR_PD['Step time']
    time_MS_HC = X_MS_HC['Step time']
    time_MS_MS = X_MS_MS['Step time']

    # Calculate speed
    speed_test = y_test / time_test / speed_factor  # from cm/sec to m/sec
    speed_ONPAR_HC = y_ONPAR_HC / time_ONPAR_HC / speed_factor 
    speed_ONPAR_PD = y_ONPAR_PD / time_ONPAR_PD / speed_factor 
    speed_MS_HC = y_MS_HC / time_MS_HC / speed_factor 
    speed_MS_MS = y_MS_MS / time_MS_MS / speed_factor 

    # Select the best features
    X_train_orig = X_train_orig.loc[:, BestFeatNames]
    X_train = X_train.loc[:, BestFeatNames]
    X_test = X_test.loc[:, BestFeatNames]
    X_MS_HC = X_MS_HC.loc[:, BestFeatNames]
    X_MS_MS = X_MS_MS.loc[:, BestFeatNames]
    X_ONPAR_HC = X_ONPAR_HC.loc[:, BestFeatNames]
    X_ONPAR_PD = X_ONPAR_PD.loc[:, BestFeatNames]

    # Scale the data
    scaler = StandardScaler()
    X_train_orig = scaler.fit_transform(X_train_orig)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X_MS_HC = scaler.transform(X_MS_HC)
    X_MS_MS = scaler.transform(X_MS_MS)
    X_ONPAR_HC = scaler.transform(X_ONPAR_HC)
    X_ONPAR_PD = scaler.transform(X_ONPAR_PD)

    # Convert to DMatrix for XGBoost
    X_test = xgb.DMatrix(pd.DataFrame(X_test))
    X_train = xgb.DMatrix(pd.DataFrame(X_train))
    X_MS_HC = xgb.DMatrix(pd.DataFrame(X_MS_HC))
    X_MS_MS = xgb.DMatrix(pd.DataFrame(X_MS_MS))
    X_ONPAR_HC = xgb.DMatrix(pd.DataFrame(X_ONPAR_HC))
    X_ONPAR_PD = xgb.DMatrix(pd.DataFrame(X_ONPAR_PD))

    # Load the pre-trained model
    model_xgb = xgb.Booster()
    model_path = 'models\model_orig' + str(seed_list[split]) + ".json"
    model_xgb.load_model(model_path)
    
    # Make predictions
    y_pred = model_xgb.predict(X_test)
    speed_pred = y_pred / time_test / speed_factor
    y_pred_MS_HC = model_xgb.predict(X_MS_HC)
    y_pred_MS_MS = model_xgb.predict(X_MS_MS)
    y_pred_ONPAR_HC = model_xgb.predict(X_ONPAR_HC)
    y_pred_ONPAR_PD = model_xgb.predict(X_ONPAR_PD)

    # Calculate errors and metrics
    RMSE = mean_squared_error(y_pred, y_test, squared=False)
    RA = np.mean(abs(y_pred - y_test) / y_test) * percentage_factor
    speed_RMSE = mean_squared_error(speed_pred, speed_test, squared=False)
    Test_mse_list.append(RMSE)
    Test_RA_list.append(RA)
    Test_speed_list.append(speed_RMSE)

    RMSE_avg, pearson_coefficient_avg, y_pred_avg, y_test_avg = Calc_Avg_Error(y_test, y_pred, avg_num_steps)
    RA_avg = np.mean(abs(y_pred_avg - y_test_avg) / y_test_avg) * percentage_factor
    RMSE_avg_list.append(RMSE_avg)
    RA_avg_list.append(RA_avg)
    
    zip_list_pred = zip(*(islice(y_pred, i, None) for i in range(avg_num_steps)))
    y_pred_avg = np.array(list(map(mean, zip_list_pred)))
    zip_list = zip(*(islice(y_test, i, None) for i in range(avg_num_steps)))
    y_test_avg = np.array(list(map(mean, zip_list)))
    speed_RMSE_avg = mean_squared_error(y_test_avg, y_pred_avg, squared=False)
    speed_mse_list_avg.append(speed_RMSE_avg)

    # Process MS HC data
    RMSE_MS_HC = mean_squared_error(y_pred_MS_HC, y_MS_HC, squared=False)
    RA_MS_HC = np.mean(abs(y_pred_MS_HC - y_MS_HC) / y_MS_HC) * percentage_factor
    speed_pred_MS_HC = y_pred_MS_HC / time_MS_HC / speed_factor
    speed_MS_RMSE_HC = mean_squared_error(speed_pred_MS_HC, speed_MS_HC, squared=False)
    mse_MS_HC_list.append(RMSE_MS_HC)
    RA_MS_HC_list.append(RA_MS_HC)
    speed_MS_HC_list.append(speed_MS_RMSE_HC)
    RMSE_MS_HC_avg, pearson_coefficient_MS_HC, y_pred_MS_HC_avg, y_MS_HC_avg = Calc_Avg_Error(y_MS_HC, y_pred_MS_HC, avg_num_steps)
    RA_MS_HC_avg = np.mean(abs(y_pred_MS_HC_avg - y_MS_HC_avg) / y_MS_HC_avg) * percentage_factor
    RMSE_MS_HC_avg_list.append(RMSE_MS_HC_avg)
    RA_MS_HC_avg_list.append(RA_MS_HC_avg)
    # Process MS MS data
    RMSE_MS_MS = mean_squared_error(y_pred_MS_MS, y_MS_MS, squared=False)
    RA_MS_MS = np.mean(abs(y_pred_MS_MS - y_MS_MS) / y_MS_MS) * percentage_factor
    speed_pred_MS_MS = y_pred_MS_MS / time_MS_MS / speed_factor
    speed_MS_RMSE_MS = mean_squared_error(speed_pred_MS_MS, speed_MS_MS, squared=False)
    mse_MS_MS_list.append(RMSE_MS_MS)
    RA_MS_MS_list.append(RA_MS_MS)
    speed_MS_MS_list.append(speed_MS_RMSE_MS)
    RMSE_MS_MS_avg, pearson_coefficient_MS_MS, y_pred_MS_MS_avg, y_MS_MS_avg = Calc_Avg_Error(y_MS_MS, y_pred_MS_MS, avg_num_steps)
    RA_MS_MS_avg = np.mean(abs(y_pred_MS_MS_avg - y_MS_MS_avg) / y_MS_MS_avg) * percentage_factor
    RMSE_MS_MS_avg_list.append(RMSE_MS_MS_avg)
    RA_MS_MS_avg_list.append(RA_MS_MS_avg)
    
    # Process ONPAR HC data
    RMSE_ONPAR_HC = mean_squared_error(y_pred_ONPAR_HC, y_ONPAR_HC, squared=False)
    RA_ONPAR_HC = np.mean(abs(y_pred_ONPAR_HC - y_ONPAR_HC) / y_ONPAR_HC) * percentage_factor
    speed_pred_ONPAR_HC = y_pred_ONPAR_HC / time_ONPAR_HC / speed_factor
    speed_ONPAR_RMSE_HC = mean_squared_error(speed_pred_ONPAR_HC, speed_ONPAR_HC, squared=False)
    mse_ONPAR_HC_list.append(RMSE_ONPAR_HC)
    RA_ONPAR_HC_list.append(RA_ONPAR_HC)
    speed_ONPAR_HC_list.append(speed_ONPAR_RMSE_HC)
    RMSE_ONPAR_HC_avg, pearson_coefficient_ONPAR_HC, y_pred_ONPAR_HC_avg, y_ONPAR_HC_avg = Calc_Avg_Error(y_ONPAR_HC, y_pred_ONPAR_HC, avg_num_steps)
    RA_ONPAR_HC_avg = np.mean(abs(y_pred_ONPAR_HC_avg - y_ONPAR_HC_avg) / y_ONPAR_HC_avg) * percentage_factor
    RMSE_ONPAR_HC_avg_list.append(RMSE_ONPAR_HC_avg)
    RA_ONPAR_HC_avg_list.append(RA_ONPAR_HC_avg)
    
    # Process ONPAR PD data
    RMSE_ONPAR_PD = mean_squared_error(y_pred_ONPAR_PD, y_ONPAR_PD, squared=False)
    RA_ONPAR_PD = np.mean(abs(y_pred_ONPAR_PD - y_ONPAR_PD) / y_ONPAR_PD) * percentage_factor
    speed_pred_ONPAR_PD = y_pred_ONPAR_PD / time_ONPAR_PD / speed_factor
    speed_ONPAR_RMSE_PD = mean_squared_error(speed_pred_ONPAR_PD, speed_ONPAR_PD, squared=False)
    mse_ONPAR_PD_list.append(RMSE_ONPAR_PD)
    RA_ONPAR_PD_list.append(RA_ONPAR_PD)
    speed_ONPAR_PD_list.append(speed_ONPAR_RMSE_PD)
    RMSE_ONPAR_PD_avg, pearson_coefficient_ONPAR_PD, y_pred_ONPAR_PD_avg, y_ONPAR_PD_avg = Calc_Avg_Error(y_ONPAR_PD, y_pred_ONPAR_PD, avg_num_steps)
    RA_ONPAR_PD_avg = np.mean(abs(y_pred_ONPAR_PD_avg - y_ONPAR_PD_avg) / y_ONPAR_PD_avg) * percentage_factor
    RMSE_ONPAR_PD_avg_list.append(RMSE_ONPAR_PD_avg)
    RA_ONPAR_PD_avg_list.append(RA_ONPAR_PD_avg)
    
    # Calculate ICC
    ICC_test = Calc_ICC(y_test, y_pred)
    ICC_MS_HC = Calc_ICC(y_MS_HC, y_pred_MS_HC)
    ICC_MS_MS = Calc_ICC(y_MS_MS, y_pred_MS_MS)
    ICC_ONPAR_HC = Calc_ICC(y_ONPAR_HC, y_pred_ONPAR_HC)
    ICC_ONPAR_PD = Calc_ICC(y_ONPAR_PD, y_pred_ONPAR_PD)
    ICC_list_test.append(ICC_test)
    ICC_list_MS_HC.append(ICC_MS_HC)
    ICC_list_MS_MS.append(ICC_MS_MS)
    ICC_list_ONPAR_HC.append(ICC_ONPAR_HC)
    ICC_list_ONPAR_PD.append(ICC_ONPAR_PD)
   
