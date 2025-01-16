# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 18:54:13 2024

@author: assafz
"""

columns_names = ['acc_min_1', 'acc_min_2', 'acc_min_3', 'acc_min_mag', 'acc_max_1', 'acc_max_2', 'acc_max_3',
                 'acc_max_mag','acc_mean_1', 'acc_mean_2', 'acc_mean_3', 'acc_mean_mag', 'acc_std_1', 'acc_std_2', 
                 'acc_std_3', 'acc_std_mag', 'acc_v_1', 'acc_v_2', 'acc_v_3', 'acc_v_mag', 'acc_s_1', 'acc_s_2', 
                 'acc_s_3', 'acc_s_mag','r_acc_1_2', 'r_acc_1_3', 'r_acc_2_3', 'gyro_min_1', 'gyro_min_2', 
                 'gyro_min_3', 'gyro_min_mag', 'gyro_max_1', 'gyro_max_2', 'gyro_max_3', 'gyro_max_mag', 
                 'gyro_mean_1', 'gyro_mean_2', 'gyro_mean_3', 'gyro_mean_mag', 'gyro_std_1', 'gyro_std_2',
                 'gyro_std_3', 'gyro_std_mag', 'gyro_v_1', 'gyro_v_2', 'gyro_v_3', 'gyro_v_mag', 'gyro_s_1',
                 'gyro_s_2', 'gyro_s_3', 'gyro_s_mag', 'r_gyro_1_2', 'r_gyro_1_3', 'r_gyro_2_3','total_pow_acc_mag', 
                'total_pow_acc_1', 'total_pow_acc_2', 'total_pow_acc_3', 'total_pow_gyro_mag', 'total_pow_gyro_1', 
                'total_pow_gyro_2', 'total_pow_gyro_3', 'StepFreqancy', 'P1_acc_mag_1', 'P1_acc_1_1',
                'P1_acc_2_1', 'P1_acc_3_1', 'P1_gyro_mag_1', 'P1_gyro_1_1', 'P1_gyro_2_1', 'P1_gyro_3_1',
                'P1_acc_mag_2', 'P1_acc_1_2', 'P1_acc_2_2', 'P1_acc_3_2', 'P1_gyro_mag_2', 'P1_gyro_1_2', 
                'P1_gyro_2_2', 'P1_gyro_3_2', 'P1_acc_mag_3', 'P1_acc_1_3', 'P1_acc_2_3', 'P1_acc_3_3', 
                'P1_gyro_mag_3', 'P1_gyro_1_3', 'P1_gyro_2_3', 'P1_gyro_3_3', 'P1_acc_mag_4', 'P1_acc_1_4', 
                'P1_acc_2_4', 'P1_acc_3_4', 'P1_gyro_mag_4', 'P1_gyro_1_4', 'P1_gyro_2_4', 'P1_gyro_3_4', 
                'P1_acc_mag_5', 'P1_acc_1_5', 'P1_acc_2_5', 'P1_acc_3_5', 'P1_gyro_mag_5', 'P1_gyro_1_5', 
                'P1_gyro_2_5', 'P1_gyro_3_5', 'P1_acc_mag_6', 'P1_acc_1_6', 'P1_acc_2_6', 'P1_acc_3_6', 
                'P1_gyro_mag_6', 'P1_gyro_1_6', 'P1_gyro_2_6', 'P1_gyro_3_6', 'ID', 'Visit', 'Walk', 'Height', 
                'IP_StepLength', 'Step time', 'StepLength']

features_cols = list(range(111))
sub_col = 111
visit_col = 112
walk_col = 113
label_col = 117
obstacle_num = 4
train_partition = 0.8
best_feat_num = [0, 2, 3, 7, 12, 13, 14, 19, 20, 21, 23, 36, 42, 48, 51, 54, 62, 65, 68, 69, 72, 74, 77, 79, 81, 82, 83, 87, 95, 97, 103, 105, 106, 108] # The features numbers that were pre-selected
PropCols = ['ID', 'Visit', 'Walk','IP_StepLength']
iterations = 5
seed_list = [42,43,44,45,46]
max_depth = 4
num_neighbors = 20
num_estimators = 100

params = {
    'min_child_weight': [1, 5, 10],
    'gamma': [0, 0.5],
    'learning_rate': [0.1, 0.01,0.05],
    'max_depth': [3, 4, 5, 6]
    }

LR_order = 0
RT_order = 1
SVR_order = 2
KNN_order = 3
GB_order = 4
XGB_order = 5
IP_order = 6

folds = 5
param_comb = 50
speed_factor = 100
percentage_factor = 100