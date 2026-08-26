#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 12:52:26 2019

s1_cm_analysis

@author: shenhao
"""

import numpy as np
import pandas as pd
#import matplotlib as mpl
#mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib
import util_helpers
import pickle
import tensorflow as tf
import os
from scipy import stats
import copy
from sklearn.metrics import f1_score

### set the working directory
#os.chdir('./output_cm')

#


###
num_replications = 10
###

#
with open('cm_info.pickle', 'rb') as f:
    cm_info = pickle.load(f)
with open('pt_info.pickle', 'rb') as f:
    pt_info = pickle.load(f)
with open('hd_info.pickle', 'rb') as f:
    hd_info = pickle.load(f)


with open('cm_info_simul.pickle', 'rb') as f:
    cm_info_simul = pickle.load(f)
with open('pt_info_simul.pickle', 'rb') as f:
    pt_info_simul = pickle.load(f)
with open('hd_info_simul.pickle', 'rb') as f:
    hd_info_simul = pickle.load(f)

# with open('cm_info_reverse.pickle', 'rb') as f:
#     cm_info_reverse = pickle.load(f)
# with open('pt_info_reverse.pickle', 'rb') as f:
#     pt_info_reverse = pickle.load(f)
# with open('hd_info_reverse.pickle', 'rb') as f:
#     hd_info_reverse = pickle.load(f)




# add simul info
# cm_info = {}
# pt_info = {}
# hd_info = {}
for key in cm_info_simul:
    if key not in cm_info:
        cm_info[key] = cm_info_simul[key]
for key in pt_info_simul:
    if key not in pt_info:
        pt_info[key] = pt_info_simul[key]
for key in hd_info_simul:
    if key not in hd_info:
        hd_info[key] = hd_info_simul[key]

#
with open('data/process/cm.pickle', 'rb') as data:
    cm_data = pickle.load(data)
with open('data/process/risk.pickle', 'rb') as data:
    pt_data = pickle.load(data)
with open('data/process/time.pickle', 'rb') as data:
    hd_data = pickle.load(data)



########################################process cm_data###########################
y_vars = ['choice']
cm_y_training = cm_data['training'][y_vars].values[:,0]
cm_y_testing = cm_data['testing'][y_vars].values[:,0]
########################################process hd_data###########################
training_df = hd_data['training']
testing_df = hd_data['testing']

# reverse the choice indicator from the initial Tanaka paper
# new alternative: x1 has time dimension.
training_df['choice'] = 1.0 - training_df['choice']
testing_df['choice'] = 1.0 - testing_df['choice']
y_var = ['choice']
hd_y_training = np.int_(training_df[y_var].values[:,0])
hd_y_testing = np.int_(testing_df[y_var].values[:,0])
########################################process pt_data###########################
## 1. pre-process data for PT

y_var = ['choice']
training_df = pt_data['training']
testing_df = pt_data['testing']
pt_y_training = training_df[y_var].values[:,0]
pt_y_testing = testing_df[y_var].values[:,0]



def output_table(info, MODEL_NAME_LIST, task_type):
    '''
    task_type == 'cm', 'pt', 'hd'
    the output is table dictionary:
        keys: model name list
        list for each key: accuracy_training, accuracy_testing, cost_training, cost_testing, log_loss_training, log_loss_testing, 
                           gradient_training, gradient_testing,mono_correct_rate 
        The order of the list CANNOT be changed, since the following coding relies on this order.
    '''
    assert task_type in ['cm', 'pt', 'hd']
    #     
    table = {}
    for MODEL_NAME in MODEL_NAME_LIST:
#        print(MODEL_NAME)
        acc_test_list = []
        acc_training_list = []
        cost_training_list = []
        cost_testing_list = []
        log_loss_training_list = []
        log_loss_testing_list = []

        f1_score_training_list = []
        f1_score_testing_list = []
        for replic in range(num_replications):
            # print(info[MODEL_NAME]['data_output'][replic]['accuracy_testing'])
            acc_test_list.append(info[MODEL_NAME]['data_output'][replic]['accuracy_testing'][-1])
            acc_training_list.append(info[MODEL_NAME]['data_output'][replic]['accuracy_training'][-1])
            cost_training_list.append(info[MODEL_NAME]['data_output'][replic]['cost_training'][-1])
            cost_testing_list.append(info[MODEL_NAME]['data_output'][replic]['cost_testing'][-1])
            log_loss_training_list.append(info[MODEL_NAME]['data_output'][replic]['log_loss_training'][-1])
            log_loss_testing_list.append(info[MODEL_NAME]['data_output'][replic]['log_loss_testing'][-1])
            prob_training = info[MODEL_NAME]['data_output'][replic]['prob_training']
            prob_testing = info[MODEL_NAME]['data_output'][replic]['prob_testing']
            current_pred_y_train = np.argmax(prob_training,1)
            current_pred_y_test = np.argmax(prob_testing,1)
            if task_type == 'cm':
                y_training = copy.deepcopy(cm_y_training)
                y_testing = copy.deepcopy(cm_y_testing)

            elif task_type == 'pt':
                y_training = copy.deepcopy(pt_y_training)
                y_testing = copy.deepcopy(pt_y_testing)
            else:
                y_training = copy.deepcopy(hd_y_training)
                y_testing = copy.deepcopy(hd_y_testing)

            current_f1_train = f1_score(y_training, current_pred_y_train, average='weighted')
            current_f1_test = f1_score(y_testing, current_pred_y_test, average='weighted')
            f1_score_training_list.append(current_f1_train)
            f1_score_testing_list.append(current_f1_test)


        best_index = acc_test_list.index(max(acc_test_list))
        accuracy_training = sum(acc_training_list)/len(acc_training_list)
        accuracy_testing = sum(acc_test_list)/len(acc_test_list)

        f1_score_training = sum(f1_score_training_list) / len(f1_score_training_list)
        f1_score_testing = sum(f1_score_testing_list) / len(f1_score_testing_list)

        cost_training = sum(cost_training_list)/len(cost_training_list)
        cost_testing = sum(cost_testing_list)/len(cost_testing_list)
        log_loss_training = sum(log_loss_training_list)/len(log_loss_training_list)
        log_loss_testing = sum(log_loss_testing_list)/len(log_loss_testing_list)
        #
        gradient_u1_x_training = info[MODEL_NAME]['data_output'][best_index]['gradient_prob1_x_training']
        gradient_u1_x_testing = info[MODEL_NAME]['data_output'][best_index]['gradient_prob1_x_testing']
        #
        gradient_training = np.linalg.norm(gradient_u1_x_training,ord=2)
        gradient_testing = np.linalg.norm(gradient_u1_x_testing,ord=2)
        # 
        if task_type == 'cm':
            # this is the gradient of probability 1 (using bus) w.r.t. X.
            # should be negative w.r.t. all bus related costs. column index: 1, 2, 3, 4
            correct_sign = -1 # pos    
            index_list = [1,2,3,4]    
            correct_sign_m = np.sign(info[MODEL_NAME]['data_output'][best_index]['gradient_prob1_x_training'][:, index_list]) == correct_sign
            mono_correct_rate = np.sum(correct_sign_m)/(len(index_list) * correct_sign_m.shape[0])
        elif task_type == 'pt':
            # this is the gradient of prob 1 (choosing alternative 1) w.r.t. X
            # should be positive w.r.t. all the x1 attributes: column index: 2,3,6,7
            correct_sign = 1 # pos
            index_list = [2,3,6,7]
            correct_sign_m = np.sign(info[MODEL_NAME]['data_output'][best_index]['gradient_prob1_x_training'][:, index_list]) == correct_sign
            mono_correct_rate = np.sum(correct_sign_m)/(len(index_list) * correct_sign_m.shape[0])
#            print(correct_rate)
        elif task_type == 'hd':
            # this is the gradient of prob 1 (choosing alternative 1) w.r.t. X
            # should be negative w.r.t. t1. column index: 2
            correct_sign = -1 # neg
            index_list = [2]
            correct_sign_m = np.sign(info[MODEL_NAME]['data_output'][best_index]['gradient_prob1_x_training'][:, index_list]) == correct_sign
            mono_correct_rate = np.sum(correct_sign_m)/correct_sign_m.shape[0]
#            print(correct_rate)
#
        table[MODEL_NAME] = [accuracy_training, accuracy_testing, cost_training, cost_testing,
                             f1_score_training, f1_score_testing,
                             log_loss_training, log_loss_testing, 
                             gradient_training, gradient_testing,
                             mono_correct_rate]
    return table

# export tables
def export_performance_table(table, model_name_list, model_name_list_formal, performance_name_list, 
                             performance_name_list_formal, table_name):
    # 
    table_df = pd.DataFrame(table).T.loc[model_name_list, :5] # only need the first four values: pred accuracy and log loss
    table_df.columns = performance_name_list
    table_df_output = copy.copy(table_df)
    table_df_output.columns = performance_name_list_formal
    table_df_output.index = model_name_list_formal
    np.round(table_df_output, decimals = 3).to_csv("output/table/"+table_name)
    return table_df



#pd.DataFrame(cm_table).T.to_csv('tmp_cm.csv')


##############################  Part 1. Performance of cm, pt, and hd ##############################
### 1. cm
CM_MODEL_NAME_LIST = ['cm_est']
penalty_const_list = [1e-50, 1e-10, 1e-5, 0.0001, 0.001, 0.002, 0.005, 0.008, 0.01, 0.05, 0.1, 1.0]
task_type = 'cm'
for name in ['cm_dnn_','cm_resnet_simul']:
    for penalty_const in penalty_const_list:
        CM_MODEL_NAME_LIST.append(name + str(penalty_const))
# convert to full CM table
cm_table = output_table(cm_info, CM_MODEL_NAME_LIST, task_type)

### export tables
### table parameters
cm_model_name_list = ['cm_dnn_1e-50',
                      'cm_resnet_simul1e-10',
                      'cm_resnet_simul0.005',
                      'cm_resnet_simul0.01',
                      'cm_est']
cm_model_name_list_formal = ['DNN',
                             'Resnet_simul (1e-10)',
                             'Resnet_simul (0.005)',
                             'Resnet_simul (0.01)',
                             'CM']

performance_name_list = ['acc_training', 'acc_testing', 'loss_training', 'loss_testing','f1_training','f1_testing']
performance_name_list_formal = ['Prediction Accuracy (Training)', 'Prediction Accuracy (Testing)', 'Cross-entropy Loss (Training)', 'Cross-entropy Loss (Training)',
                                'f1-score (Training)','f1-score (Testing)']
# 
cm_table_df = export_performance_table(cm_table, cm_model_name_list, cm_model_name_list_formal, 
                                       performance_name_list, performance_name_list_formal, '0_cm_table_simul.csv')

### three visualizations
# plot pred accuracy of cm
util_helpers.export_performance_figure(table_df = cm_table_df, figsize = (8, 4), var_name = 'acc_testing', 
                          x_tick_old = cm_model_name_list, x_tick_new = cm_model_name_list_formal, 
                          xlabel = 'CM Models', ylabel = 'Prediction Accuracy', figure_name = 'cm_pred_accuracy_simul.png')
# plot parameters for cross entropy loss of cm
util_helpers.export_performance_figure(table_df = cm_table_df, figsize = (8, 4), var_name = 'loss_testing', 
                          x_tick_old = cm_model_name_list, x_tick_new = cm_model_name_list_formal, 
                          xlabel = 'CM Models', ylabel = 'Cross-entropy Loss', figure_name = 'cm_loss_simul.png')
# plot monotonicity evaluation
cm_mono_table = pd.DataFrame(cm_table).T.loc[cm_model_name_list, 10:]
cm_mono_table.columns = ['mono']
util_helpers.export_performance_figure(table_df = cm_mono_table, figsize = (8, 4), var_name = 'mono', 
                          x_tick_old = cm_model_name_list, x_tick_new = cm_model_name_list_formal, 
                          xlabel = 'CM Models', ylabel = 'Monotonicity Accuracy', figure_name = 'cm_mono_simul.png')


### 2. pt
PT_MODEL_NAME_LIST = ['pt_est']
penalty_const_list = [1e-50, 1e-10, 1e-5, 0.0001, 0.001, 0.01, 0.1, 1.0]
task_type = 'pt'
for name in ['pt_dnn_','pt_resnet_simul']:
    for penalty_const in penalty_const_list:
        PT_MODEL_NAME_LIST.append(name + str(penalty_const))
        
# convert to full PT table
pt_table = output_table(pt_info, PT_MODEL_NAME_LIST, task_type)

### export tables
### table parameters

pt_model_name_list = ['pt_dnn_1e-50',
                      'pt_resnet_simul1e-05',
                      'pt_resnet_simul0.0001',
                      'pt_resnet_simul0.01',
                      'pt_est']
pt_model_name_list_formal = ['DNN',
                             'Resnet_simul (1e-05)',
                             'Resnet_simul (0.0001)',
                             'Resnet_simul (0.01)',
                             'PT']


pt_table_df = export_performance_table(pt_table, pt_model_name_list, pt_model_name_list_formal, 
                                       performance_name_list, performance_name_list_formal, '0_pt_table_simul.csv')

### three visualizations
# plot pred accuracy of pt
util_helpers.export_performance_figure(table_df = pt_table_df, figsize = (8, 4), var_name = 'acc_testing', 
                          x_tick_old = pt_model_name_list, x_tick_new = pt_model_name_list_formal, 
                          xlabel = 'PT Models', ylabel = 'Prediction Accuracy', figure_name = 'pt_pred_accuracy_simul.png')
# plot parameters for cross entropy loss of pt
util_helpers.export_performance_figure(table_df = pt_table_df, figsize = (8, 4), var_name = 'loss_testing', 
                          x_tick_old = pt_model_name_list, x_tick_new = pt_model_name_list_formal, 
                          xlabel = 'PT Models', ylabel = 'Cross-entropy Loss', figure_name = 'pt_loss_simul.png')
# plot monotonicity evaluation
pt_mono_table = pd.DataFrame(pt_table).T.loc[pt_model_name_list, 10:]
pt_mono_table.columns = ['mono']
util_helpers.export_performance_figure(table_df = pt_mono_table, figsize = (8, 4), var_name = 'mono', 
                          x_tick_old = pt_model_name_list, x_tick_new = pt_model_name_list_formal, 
                          xlabel = 'PT Models', ylabel = 'Monotonicity Accuracy', figure_name = 'pt_mono_simul.png')


### 3. hd
HD_MODEL_NAME_LIST = ['hd_est']
penalty_const_list = [1e-50, 1e-10, 1e-5, 0.0001, 0.001, 0.002, 0.003, 0.004, 0.005, 0.008, 0.01, 0.1, 1.0]
task_type = 'hd'
for name in ['hd_dnn_','hd_resnet_simul']:
    for penalty_const in penalty_const_list:
        HD_MODEL_NAME_LIST.append(name + str(penalty_const))

# convert to full HD table
hd_table = output_table(hd_info, HD_MODEL_NAME_LIST, task_type)

### export tables
### table parameters
hd_model_name_list = ['hd_dnn_1e-50',
                      'hd_resnet_simul1e-05',
                      'hd_resnet_simul0.001',
                      'hd_resnet_simul0.01',
                      'hd_est']
hd_model_name_list_formal = ['DNN',
                             'Resnet_simul (1e-05)',
                             'Resnet_simul (0.001)',
                             'Resnet_simul (0.01)',
                             'HD']


hd_table_df = export_performance_table(hd_table, hd_model_name_list, hd_model_name_list_formal, 
                                       performance_name_list, performance_name_list_formal, '0_hd_table_simul.csv')

### three visualizations
# plot pred accuracy of hd
util_helpers.export_performance_figure(table_df = hd_table_df, figsize = (8, 4), var_name = 'acc_testing', 
                          x_tick_old = hd_model_name_list, x_tick_new = hd_model_name_list_formal, 
                          xlabel = 'HD Models', ylabel = 'Prediction Accuracy', figure_name = 'hd_pred_accuracy_simul.png')
# plot parameters for cross entropy loss of hd
util_helpers.export_performance_figure(table_df = hd_table_df, figsize = (8, 4), var_name = 'loss_testing', 
                          x_tick_old = hd_model_name_list, x_tick_new = hd_model_name_list_formal, 
                          xlabel = 'HD Models', ylabel = 'Cross-entropy Loss', figure_name = 'hd_loss_simul.png')
# plot monotonicity evaluation
hd_mono_table = pd.DataFrame(hd_table).T.loc[hd_model_name_list, 10:]
hd_mono_table.columns = ['mono']
util_helpers.export_performance_figure(table_df = hd_mono_table, figsize = (8, 4), var_name = 'mono', 
                          x_tick_old = hd_model_name_list, x_tick_new = hd_model_name_list_formal, 
                          xlabel = 'HD Models', ylabel = 'Monotonicity Accuracy', figure_name = 'hd_mono_simul.png')




##############################  Part 2. Interpretability of cm, pt, and hd ##############################
### 1. cm
# generate grids.
n_grid = 100
x_min=np.min(cm_data['training']['bus_cost'])
x_max=np.max(cm_data['training']['bus_cost'])
y_min=np.min(cm_data['training']['bus_ivt'])
y_max=np.max(cm_data['training']['bus_ivt'])
x0 = np.linspace(x_min, x_max, n_grid)
x1 = np.linspace(y_min, y_max, n_grid)
x0_grid, x1_grid = np.meshgrid(x0, x1)

# Focus on the models defined in Part 1.
cm_model_name_list = ['cm_dnn_1e-50', 'cm_resnet_simul1e-10', 'cm_resnet_simul0.005',
                      'cm_resnet_simul0.01', 'cm_resnet_simul0.1', 'cm_est']

# cm_model_name_list = ['cm_dnn_1e-50', 'cm_resnet_1e-10', 'cm_resnet_0.005', 'cm_resnet_0.01', 'cm_resnet_0.1', 'cm_est']
cm_model_name_list_save = [model_name.replace('.','') for model_name in cm_model_name_list]

# visualization meta
for MODEL_NAME in cm_model_name_list:
    idx = cm_model_name_list.index(MODEL_NAME)
    Z = cm_info[MODEL_NAME]['grid_output']['u_training'][:, 1].reshape(n_grid, n_grid)
    util_helpers.visualize_contour(x0_grid, x1_grid, Z, figsize = (4, 4), linewidths = 3, 
                                   xlabel = 'Bus Cost', ylabel = 'Bus IVT', 
                                   figure_name = cm_model_name_list_save[idx] + '_field.png')
    util_helpers.export_one_row_figure(Z[50, :], figsize = (3, 3), xlabel = 'Cost', ylabel = 'Utility', figure_name = cm_model_name_list_save[idx]+'_x0.png')
    util_helpers.export_one_row_figure(Z[:, 50], figsize = (3, 3), xlabel = 'IVT', ylabel = 'Utility', figure_name = cm_model_name_list_save[idx]+'_x1.png')

    
### 2. pt
# generate grids.
n_grid = 100
x_min=0.0
x_max=20.0
y_min=0.0
y_max=1.0
x0 = np.linspace(x_min, x_max, n_grid)
x1 = np.linspace(y_min, y_max, n_grid)
x0_grid, x1_grid = np.meshgrid(x0, x1)

# Focus on the models defined in Part 1.

pt_model_name_list_save = [model_name.replace('.','') for model_name in pt_model_name_list]
# visualization meta
for MODEL_NAME in pt_model_name_list:
    idx = pt_model_name_list.index(MODEL_NAME)
    Z = pt_info[MODEL_NAME]['grid_output']['u_training'][:, 1].reshape(n_grid, n_grid)
    util_helpers.visualize_contour(x0_grid, x1_grid, Z, figsize = (4, 4), linewidths = 3, 
                                   xlabel = 'Monetary Values', ylabel = 'Winning Probability', 
                                   figure_name = pt_model_name_list_save[idx] + '_field.png')
    util_helpers.export_one_row_figure(Z[50, :], figsize = (3, 3), xlabel = 'Values', ylabel = 'Utility', figure_name = pt_model_name_list_save[idx]+'_x0.png')
    util_helpers.export_one_row_figure(Z[:, 50], figsize = (3, 3), xlabel = 'Probability', ylabel = 'Utility', figure_name = pt_model_name_list_save[idx]+'_x1.png')


### 3. hd
n_grid = 100
x_min = 0.0
x_max = 3.0 # money values of alt 1
y_min = 0.0
y_max = 10.0 # time
x0 = np.linspace(x_min, x_max, n_grid)
x1 = np.linspace(y_min, y_max, n_grid)
x0_grid, x1_grid = np.meshgrid(x0, x1)

# Focus on the models defined in Part 1.

hd_model_name_list_save = [model_name.replace('.','') for model_name in hd_model_name_list]

# visualization meta
for MODEL_NAME in hd_model_name_list:
    idx = hd_model_name_list.index(MODEL_NAME)
    Z = hd_info[MODEL_NAME]['grid_output']['u_training'][:, 1].reshape(n_grid, n_grid)
    util_helpers.visualize_contour(x0_grid, x1_grid, Z, figsize = (4, 4), linewidths = 3, 
                                   xlabel = 'Monetary Values', ylabel = 'Time', 
                                   figure_name = hd_model_name_list_save[idx] + '_field.png')
    util_helpers.export_one_row_figure(Z[50, :], figsize = (3, 3), xlabel = 'Values', ylabel = 'Utility', figure_name = hd_model_name_list_save[idx]+'_x0.png')
    util_helpers.export_one_row_figure(Z[5:, 50], figsize = (3, 3), xlabel = 'Time', ylabel = 'Utility', figure_name = hd_model_name_list_save[idx]+'_x1.png')



##############################  Part 3. Robustness of cm, pt, and hd ##############################
def robust_table(info, MODEL_NAME_LIST, epsilon_list):
    # 
    robust_fgsm_accuracy = {}
    robust_tgsm_accuracy = {}
    for MODEL_NAME in MODEL_NAME_LIST:
        robust_fgsm_accuracy[MODEL_NAME] = []
        robust_tgsm_accuracy[MODEL_NAME] = []
        for epsilon in epsilon_list:
            robust_fgsm_accuracy[MODEL_NAME].append(info[MODEL_NAME]['fgsm_output'][epsilon]['accuracy_testing'][-1])
            robust_tgsm_accuracy[MODEL_NAME].append(info[MODEL_NAME]['tgsm_output'][epsilon]['accuracy_testing'][-1])
    return robust_fgsm_accuracy,robust_tgsm_accuracy


def plot_multiple_lines(data, xlabel, ylabel, figure_name, xtick_old, xtick_new):
    # overall style
#    plt.style.use('seaborn-whitegrid') # set the style; or "ggplot", "dark_background", "classic";
    # plot with basic df.plot
    col = list(data.columns)
    col_new = [ key.replace('_simul','') for key in col]
    data.columns = col_new
    matplotlib.rcParams.update({'font.size': 15})    
    ax = data.plot(kind = 'line', figsize = (4, 4), linewidth = 3.0, marker = 'o') 
    # kind: hist, box, density, scatter, etc. style: linestyle;    
    # set labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(xtick_old)
    ax.set_xticklabels(xtick_new)
#    # size
#    ax.xaxis.label.set_fontsize(20)
#    ax.yaxis.label.set_fontsize(20)
#    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
#        label.set_fontname('Arial')
#        label.set_fontsize(13)
    # legend
    ax.legend(loc = 2, fontsize = 15, title = None) # remove the legend title of "City_province"
    fig = ax.get_figure() # Without initialize fig, we get figure in the end.
    plt.tight_layout()
    fig.savefig('output/robust/'+figure_name)
    plt.close()


### meta
epsilon_list = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
cut_index = 3
#%matplotlib inline

### 1) cm
cm_model_name_list = ['cm_dnn_1e-50', 'cm_resnet_simul0.005', 'cm_est']
cm_model_name_list_formal = ['DNN', 'Resnet_simul (0.005)', 'CM']
cm_robust_fgsm_accuracy,cm_robust_tgsm_accuracy = robust_table(info = cm_info, MODEL_NAME_LIST = cm_model_name_list, 
                                                               epsilon_list = epsilon_list)
cm_robust_fgsm_accuracy_table = pd.DataFrame(cm_robust_fgsm_accuracy).loc[:cut_index, :]
cm_robust_tgsm_accuracy_table = pd.DataFrame(cm_robust_tgsm_accuracy).loc[:cut_index, :]
cm_robust_fgsm_accuracy_table.columns = cm_model_name_list_formal
cm_robust_tgsm_accuracy_table.columns = cm_model_name_list_formal

# plot
plot_multiple_lines(cm_robust_fgsm_accuracy_table, xlabel = 'epsilon', ylabel = 'Prediction Accuracy', 
                    figure_name = 'cm_fgsm_simul.png', xtick_old = np.arange(0, cut_index+1), xtick_new = epsilon_list[:cut_index+1])
plot_multiple_lines(cm_robust_tgsm_accuracy_table, xlabel = 'epsilon', ylabel = 'Prediction Accuracy', 
                    figure_name = 'cm_tgsm_simul.png', xtick_old = np.arange(0, cut_index+1), xtick_new = epsilon_list[:cut_index+1])


### 2) pt
pt_model_name_list = ['pt_dnn_1e-50', 'pt_resnet_simul0.0001', 'pt_est']
pt_model_name_list_formal = ['DNN', 'Resnet_simul (0.0001)', 'PT']

pt_robust_fgsm_accuracy,pt_robust_tgsm_accuracy = robust_table(info = pt_info, MODEL_NAME_LIST = pt_model_name_list, 
                                                               epsilon_list = epsilon_list)

pt_robust_fgsm_accuracy_table = pd.DataFrame(pt_robust_fgsm_accuracy).loc[:cut_index, :]
pt_robust_tgsm_accuracy_table = pd.DataFrame(pt_robust_tgsm_accuracy).loc[:cut_index, :]
pt_robust_fgsm_accuracy_table.columns = pt_model_name_list_formal
pt_robust_tgsm_accuracy_table.columns = pt_model_name_list_formal

# plot
plot_multiple_lines(pt_robust_fgsm_accuracy_table, xlabel = 'epsilon', ylabel = 'Prediction Accuracy', 
                    figure_name = 'pt_fgsm_simul.png', xtick_old = np.arange(0, cut_index+1), xtick_new = epsilon_list[:cut_index+1])
plot_multiple_lines(pt_robust_tgsm_accuracy_table, xlabel = 'epsilon', ylabel = 'Prediction Accuracy', 
                    figure_name = 'pt_tgsm_simul.png', xtick_old = np.arange(0, cut_index+1), xtick_new = epsilon_list[:cut_index+1])

### 3) hd
hd_model_name_list = ['hd_dnn_1e-50', 'hd_resnet_simul0.001', 'hd_est']
hd_model_name_list_formal = ['DNN', 'Resnet_simul (0.001)', 'HD']

hd_robust_fgsm_accuracy,hd_robust_tgsm_accuracy = robust_table(info = hd_info, MODEL_NAME_LIST = hd_model_name_list, 
                                                               epsilon_list = epsilon_list)

hd_robust_fgsm_accuracy_table = pd.DataFrame(hd_robust_fgsm_accuracy).loc[:cut_index, :]
hd_robust_tgsm_accuracy_table = pd.DataFrame(hd_robust_tgsm_accuracy).loc[:cut_index, :]
hd_robust_fgsm_accuracy_table.columns = hd_model_name_list_formal
hd_robust_tgsm_accuracy_table.columns = hd_model_name_list_formal

# plot
plot_multiple_lines(hd_robust_fgsm_accuracy_table, xlabel = 'epsilon', ylabel = 'Prediction Accuracy', 
                    figure_name = 'hd_fgsm_simul.png', xtick_old = np.arange(0, cut_index+1), xtick_new = epsilon_list[:cut_index+1])
plot_multiple_lines(hd_robust_tgsm_accuracy_table, xlabel = 'epsilon', ylabel = 'Prediction Accuracy', 
                    figure_name = 'hd_tgsm_simul.png', xtick_old = np.arange(0, cut_index+1), xtick_new = epsilon_list[:cut_index+1])











