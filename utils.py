# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 18:09:08 2024

@author: assafz
"""
from datetime import datetime
from Consts import train_partition
import numpy as np
import pandas as pd
import pingouin as pg
from statistics import mean
from itertools import islice
from sklearn.metrics import mean_squared_error
from scipy import stats


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))
        

def train_test_split(data,seed, partition=train_partition):
    # Split the data so each participant will appear in either the train or the test set
    unique_users = data['ID'].unique()
    train_users, test_users = np.split(
        np.random.RandomState(seed=seed).permutation((unique_users)), [int(partition * len(unique_users))])

    df_train = data[data['ID'].isin(train_users)]
    df_test = data[data['ID'].isin(test_users)]
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    
    groups = df_train.loc[:, 'ID']
    X_train = df_train.loc[:, :]
    Y_train = df_train.loc[:, 'StepLength']
    X_test = df_test.loc[:, :]
    Y_test = df_test.loc[:, 'StepLength']

    return X_train, Y_train, X_test, Y_test, groups


def Calc_ICC(y_test, y_pred):
    step_list = np.arange(len(y_test))
    test_list = ['test'] * len(y_test)
    pred_list = ['pred'] * len(y_pred)
    ICC_step_list = np.hstack((step_list, step_list))
    ICC_name_list = np.hstack((test_list, pred_list))
    ICC_res_list = np.hstack((y_test, y_pred))
    ICC_dictionary = {'exam': ICC_step_list, 'judge': ICC_name_list, 'result': ICC_res_list}
    ICC_data = pd.DataFrame(ICC_dictionary)
    icc = pg.intraclass_corr(data=ICC_data, targets='exam', raters='judge', ratings='result')
    return icc


def Calc_Avg_Error(y_test, y_pred, samples):
    zip_list_pred =  zip(*(islice(y_pred, i, None) for i in range(samples)))
    y_pred_samp = np.array(list(map(mean, zip_list_pred)))
    zip_list_samp =  zip(*(islice(y_test, i, None) for i in range(samples)))
    y_test_samp = np.array(list(map(mean, zip_list_samp)))
    RMSE_samp = mean_squared_error(y_test_samp, y_pred_samp, squared=False)
    pearson_coefficient, p_value = stats.pearsonr(y_test_samp, y_pred_samp)
    
    return RMSE_samp, pearson_coefficient, y_pred_samp, y_test_samp



def load_data(data,HC_ID):
    data_HC = data[data['ID'].isin(HC_ID)]
    X = data_HC.loc[:, :]
    Y = data_HC.loc[:, 'StepLength']

    return X, Y


def load_data_new(data,HC_ID, HC): #New for checking differences
    data_HC = data[data['ID'].isin(HC_ID)]
    group_dict = dict(zip(HC_ID, HC['Group']))
    HC.rename(columns={'SubNumber': 'ID'}, inplace=True)
    data_HC['Group'] = data_HC['ID'].map(group_dict)
    X = data_HC.loc[:, :]
    Y = data_HC.loc[:, 'StepLength']

    return X, Y

