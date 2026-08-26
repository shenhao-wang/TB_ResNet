#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 09:48:04 2019

s1_training_cm

@author: shenhao
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pickle
import util_functions as util
import util_helpers
import copy

with open('data/process/cm.pickle', 'rb') as data:
    cm_data = pickle.load(data)


###
num_replications = 10
### 
x0_vars = ['walk_walktime']
x1_vars = ['bus_cost', 'bus_walktime', 'bus_waittime', 'bus_ivt']
x2_vars = ['ridesharing_cost', 'ridesharing_waittime', 'ridesharing_ivt']
x3_vars = ['drive_cost', 'drive_walktime', 'drive_ivt']
x4_vars = ['av_cost', 'av_waittime', 'av_ivt']

all_elas_var = { 'x0_vars':['walk_walktime'],
                 'x1_vars':['bus_cost', 'bus_walktime', 'bus_waittime', 'bus_ivt'],
                 'x2_vars':['ridesharing_cost', 'ridesharing_waittime', 'ridesharing_ivt'],
                 'x3_vars':['drive_cost', 'drive_walktime', 'drive_ivt'],
                 'x4_vars':['av_cost', 'av_waittime', 'av_ivt']} #sequence is important!

z_vars = ['male','young_age','old_age','low_edu','high_edu',
          'low_inc', 'high_inc', 'full_job', 'age', 'inc',
          'edu']
y_vars = ['choice']
modes_list = ['Walk', 'PT', 'RH', 'Drive', 'AV']
key_choice_index = ['Walk','PT','RH','Drive','AV']
# grid data as default for est_cm and est_resnet
data_grid = {}
n_grid = 100
data_grid['training'] = {}
data_grid['testing'] = {}
data_grid['training'] = [np.tile(np.mean(cm_data['training'][x0_vars].values, axis = 0), [n_grid**2, 1]), np.tile(np.mean(cm_data['training'][x1_vars].values, axis = 0), [n_grid**2, 1]), np.tile(np.mean(cm_data['training'][x2_vars].values, axis = 0), [n_grid**2, 1]), np.tile(np.mean(cm_data['training'][x3_vars].values, axis = 0), [n_grid**2, 1]), np.tile(np.mean(cm_data['training'][x4_vars].values, axis = 0), [n_grid**2, 1]), np.tile(np.mean(cm_data['training'][z_vars].values, axis = 0), [n_grid**2, 1]), np.zeros(n_grid**2)] # the order: x0, x1, x2, x3, x4, z, y
# vary the drive_cost and drive_ivt to observe the changes
x_min=np.min(cm_data['training']['bus_cost'])
x_max=np.max(cm_data['training']['bus_cost'])
y_min=np.min(cm_data['training']['bus_ivt'])
y_max=np.max(cm_data['training']['bus_ivt'])
data_grid['training'][1][:,0] = util_helpers.generate_mesh_data(n_grid, x_min, x_max, y_min, y_max).values[:, 0]
data_grid['training'][1][:,3] = util_helpers.generate_mesh_data(n_grid, x_min, x_max, y_min, y_max).values[:, 1]
# copy training to testing
data_grid['testing'] = copy.copy(data_grid['training'])

df_sp_test_nonstand = pd.read_csv('data/raw/data_AV_Singapore_v1_sp_test_nonstand.csv') # for elasticity

############################################################
#penalty_const_list = [1e-50]

#epsilon_list = [0.1]
epsilon_list = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
n_epoches = 5000
K = 5 # num of classes
n_class = 5 # same as above



############################################################
## model 1: cm
data = {}
data['training'] = {}
data['testing'] = {}
#
data['training'] = [cm_data['training'][x0_vars].values, cm_data['training'][x1_vars].values,
                    cm_data['training'][x2_vars].values, cm_data['training'][x3_vars].values,
                    cm_data['training'][x4_vars].values, cm_data['training'][z_vars].values,
                    cm_data['training'][y_vars].values[:,0]] # the order: x0, x1, x2, x3, x4, z, y

data['testing'] = [cm_data['testing'][x0_vars].values, cm_data['testing'][x1_vars].values,
                   cm_data['testing'][x2_vars].values, cm_data['testing'][x3_vars].values,
                   cm_data['testing'][x4_vars].values, cm_data['testing'][z_vars].values,
                   cm_data['testing'][y_vars].values[:,0]]
#
MODEL_NAME = 'cm_est'


elast_records = util.est_cm_elasticity(data, MODEL_NAME, all_elas_var, df_sp_test_nonstand)
###
elast_records_cm = {}
for key in elast_records:  # change index to name
    mode = key_choice_index[int(key.split('___')[0])]
    var = key.split('___')[1]
    new_key = mode + '___' + var
    elast_records_cm[new_key] = elast_records[key]


var_list_for_elast = ['walk_walktime','bus_cost','bus_ivt','ridesharing_cost','ridesharing_ivt',
            'drive_cost','drive_ivt','av_cost','av_ivt']


elast_records_cm_save = {'Variables': var_list_for_elast}
for mode in modes_list:
    elast_records_cm_save[mode] = [0] * len(var_list_for_elast)
elast_records_cm_save = pd.DataFrame(elast_records_cm_save)
for col in elast_records_cm:
    mode = col.split('___')[0]
    var = col.split('___')[1]
    elast_records_cm_save.loc[elast_records_cm_save['Variables'] == var, mode] = elast_records_cm[col]
elast_records_cm_save.to_csv('output/table/elasticity_cm.csv', index=False)
#



############################################################
# model 2: DNN
data = {}
data['training'] = {}
data['testing'] = {}
data['training'] = [np.concatenate([cm_data['training'][x0_vars].values, cm_data['training'][x1_vars].values, cm_data['training'][x2_vars].values,
                    cm_data['training'][x3_vars].values, cm_data['training'][x4_vars].values, cm_data['training'][z_vars].values], axis = 1),
                    cm_data['training'][y_vars].values[:,0]] #sequence is important!
data['testing'] = [np.concatenate([cm_data['testing'][x0_vars].values, cm_data['testing'][x1_vars].values, cm_data['testing'][x2_vars].values,
                    cm_data['testing'][x3_vars].values, cm_data['testing'][x4_vars].values, cm_data['testing'][z_vars].values], axis = 1),
                   cm_data['testing'][y_vars].values[:,0]] #sequence is important!
data_grid_dnn = {}
data_grid_dnn['training']= [np.concatenate([data_grid['training'][0], data_grid['training'][1], data_grid['training'][2], data_grid['training'][3], data_grid['training'][4], data_grid['training'][5]], axis = 1), data_grid['training'][-1]]
data_grid_dnn['testing'] = copy.copy(data_grid_dnn['training'])

# epsilon list for attacks
penalty_const_list = [1e-50]

all_elas_var_id = {}
count = 0
for key in all_elas_var:
    all_elas_var_id[key] = []
    for var in all_elas_var[key]:
        all_elas_var_id[key].append(count)
        count += 1

for penalty_const in penalty_const_list:
    #
    MODEL_NAME = 'cm_dnn_' + str(penalty_const)

    elast_records = util.est_dnn_elasticity(data, MODEL_NAME, all_elas_var,all_elas_var_id, df_sp_test_nonstand,penalty_const, n_epoches = n_epoches, K = len(modes_list))

    var_list_for_elast = ['walk_walktime','bus_cost','bus_ivt','ridesharing_cost','ridesharing_ivt',
                'drive_cost','drive_ivt','av_cost','av_ivt']
    modes_list = ['Walk','PT','RH','Drive','AV']

    elast_records_dnn = {}
    for key in elast_records:  # change index to name
        mode = key_choice_index[int(key.split('___')[0])]
        var = key.split('___')[1]
        new_key = mode + '___' + var
        elast_records_dnn[new_key] = elast_records[key]

    elast_records_cm_dnn_save = {'Variables': var_list_for_elast}
    for mode in modes_list:
        elast_records_cm_dnn_save[mode] = [0] * len(var_list_for_elast)
    elast_records_cm_dnn_save = pd.DataFrame(elast_records_cm_dnn_save)
    for col in elast_records_dnn:
        mode = col.split('___')[0]
        var = col.split('___')[1]
        elast_records_cm_dnn_save.loc[elast_records_cm_dnn_save['Variables'] == var, mode] = elast_records_dnn[col]
    elast_records_cm_dnn_save.to_csv('output/table/elasticity_cm_dnn_' + str(penalty_const) + '.csv', index=False)
    #



############################################################
# model 3: cm_resnet
data = {}
data['training'] = {}
data['testing'] = {}
#
data['training'] = [cm_data['training'][x0_vars].values, cm_data['training'][x1_vars].values, cm_data['training'][x2_vars].values, cm_data['training'][x3_vars].values, cm_data['training'][x4_vars].values, cm_data['training'][z_vars].values, cm_data['training'][y_vars].values[:,0]] # the order: x0, x1, x2, x3, x4, z, y
data['testing'] = [cm_data['testing'][x0_vars].values, cm_data['testing'][x1_vars].values, cm_data['testing'][x2_vars].values, cm_data['testing'][x3_vars].values, cm_data['testing'][x4_vars].values, cm_data['testing'][z_vars].values, cm_data['testing'][y_vars].values[:,0]] # the order: x0, x1, x2, x3, x4, z, y
#
penalty_const_list = [0.01]#[0.005] # best
##
# load best cm model para
with open('cm_info.pickle', 'rb') as f:
    cm_info = pickle.load(f)
MODEL_NAME_cm = 'cm_est'
acc_test_list = []
for replic in range(num_replications):
    # print(info[MODEL_NAME]['data_output'][replic]['accuracy_testing'])
    acc_test_list.append(cm_info[MODEL_NAME_cm]['data_output'][replic]['accuracy_testing'][-1])
best_index = acc_test_list.index(max(acc_test_list))

cm_param_dic = cm_info[MODEL_NAME_cm]['model_para'][best_index]
##


for penalty_const in penalty_const_list:
    ###
    MODEL_NAME = 'cm_resnet_'+str(penalty_const)



    elast_records = util.est_cm_resnet_elasticity(data, cm_param_dic, penalty_const, MODEL_NAME, all_elas_var, df_sp_test_nonstand,
                                                  n_epoches = n_epoches, K = len(modes_list))

    var_list_for_elast = ['walk_walktime','bus_cost','bus_ivt','ridesharing_cost','ridesharing_ivt',
                'drive_cost','drive_ivt','av_cost','av_ivt']
    modes_list = ['Walk','PT','RH','Drive','AV']

    elast_records_resnet = {}
    for key in elast_records:  # change index to name
        mode = key_choice_index[int(key.split('___')[0])]
        var = key.split('___')[1]
        new_key = mode + '___' + var
        elast_records_resnet[new_key] = elast_records[key]

    elast_records_cm_resnet_save = {'Variables': var_list_for_elast}
    for mode in modes_list:
        elast_records_cm_resnet_save[mode] = [0] * len(var_list_for_elast)
    elast_records_cm_resnet_save = pd.DataFrame(elast_records_cm_resnet_save)
    for col in elast_records_resnet:
        mode = col.split('___')[0]
        var = col.split('___')[1]
        elast_records_cm_resnet_save.loc[elast_records_cm_resnet_save['Variables'] == var, mode] = elast_records_resnet[col]
    elast_records_cm_resnet_save.to_csv('output/table/elasticity_cm_resnet_' + str(penalty_const) + '.csv', index=False)




