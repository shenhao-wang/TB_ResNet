#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: baichuan
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
with open('cm_info_Delta_Lambda_relation.pickle', 'rb') as f:
    cm_info = pickle.load(f)
# with open('pt_info_Delta_Lambda_relation.pickle', 'rb') as f:
#     pt_info = pickle.load(f)
# with open('hd_info_Delta_Lambda_relation.pickle', 'rb') as f:
#     hd_info = pickle.load(f)

with open('data/process/cm.pickle', 'rb') as data:
    cm_data = pickle.load(data)
with open('data/process/risk.pickle', 'rb') as data:
    pt_data = pickle.load(data)
with open('data/process/time.pickle', 'rb') as data:
    hd_data = pickle.load(data)

###
num_replications = 1
###

num_class_dict = {'cm':5,'pt':2,'hd':2}

def output_table(info, MODEL_NAME_LIST, task_type):
    assert task_type in ['cm', 'pt', 'hd']

    result_table = {'Model_name':[]}
    for k in range(num_class_dict[task_type]):
        result_table['u_model_class_'+str(int(k))] = []
        result_table['u_dnn_class_' + str(int(k))] = []



    #
    table = {}
    for MODEL_NAME in MODEL_NAME_LIST:
#        print(MODEL_NAME)
        acc_test_list = []
        acc_training_list = []
        for replic in range(num_replications):
            # print(info[MODEL_NAME]['data_output'][replic]['accuracy_testing'])
            acc_test_list.append(info[MODEL_NAME]['data_output'][replic]['accuracy_testing'][-1])
            acc_training_list.append(info[MODEL_NAME]['data_output'][replic]['accuracy_training'][-1])

        best_index = acc_test_list.index(max(acc_test_list))

        if task_type == 'cm':
            # this is the gradient of probability 1 (using bus) w.r.t. X.
            # should be negative w.r.t. all bus related costs. column index: 1, 2, 3, 4
            correct_sign = -1 # pos
            index_list = [1,2,3,4]
            correct_sign_m = np.sign(info[MODEL_NAME]['data_output'][best_index]['gradient_prob1_x_training'][:, index_list]) == correct_sign
            mono_correct_rate = np.sum(correct_sign_m)/(len(index_list) * correct_sign_m.shape[0])

            # u_cm_training = info[MODEL_NAME]['data_output'][best_index]['u_cm_training']
            u_cm_testing = info[MODEL_NAME]['data_output'][best_index]['u_cm_testing']
            u_dnn_testing = info[MODEL_NAME]['data_output'][best_index]['u_dnn_testing']
            result_table['Model_name'].append(MODEL_NAME)
            for k in range(len(u_cm_testing)):
                result_table['u_model_class_'+str(int(k))].append(u_cm_testing[k])
                result_table['u_dnn_class_' + str(int(k))].append(u_dnn_testing[k])

            a=1

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
        table[MODEL_NAME] = [accuracy_training, accuracy_testing,
                             cost_training, cost_testing,
                             f1_score_training, f1_score_testing,
                             log_loss_training, log_loss_testing,
                             gradient_training, gradient_testing,
                             mono_correct_rate]
    return table

# export tables
def export_performance_table(table, model_name_list, model_name_list_formal, performance_name_list,
                             performance_name_list_formal, table_name):
    #
    table_df = pd.DataFrame(table).T.loc[model_name_list, :5] # only need the first 6 values: pred accuracy and log loss and f1
    table_df.columns = performance_name_list
    table_df_output = copy.copy(table_df)
    table_df_output.columns = performance_name_list_formal
    table_df_output.index = model_name_list_formal
    np.round(table_df_output, decimals = 3).to_csv("output/table/"+table_name)
    return table_df



#pd.DataFrame(cm_table).T.to_csv('tmp_cm.csv')

def get_utility():
    ##############################  Part 1. Performance of cm, pt, and hd ##############################
    ### 1. cm
    CM_MODEL_NAME_LIST = []
    penalty_const_list = [1e-50, 1e-10, 1e-5, 0.0001, 0.001, 0.002, 0.005, 0.008, 0.01, 0.05, 0.1, 1.0]
    # penalty_const_list = []
    task_type = 'cm'
    for name in ['cm_resnet_']: # only
        for penalty_const in penalty_const_list:
            CM_MODEL_NAME_LIST.append(name + str(penalty_const))
    # convert to full CM table
    cm_table = output_table(cm_info, CM_MODEL_NAME_LIST, task_type)

    ### export tables
    ### table parameters
    cm_model_name_list = ['cm_dnn_1e-50',
                          'cm_resnet_1e-10',
                          'cm_resnet_0.005',
                          'cm_resnet_0.01',
                          'cm_est']
    cm_model_name_list_formal = ['DNN',
                                 'Resnet (1e-10)',
                                 'Resnet (0.005)',
                                 'Resnet (0.01)',
                                 'CM']
    performance_name_list = ['acc_training', 'acc_testing', 'loss_training', 'loss_testing','f1_training','f1_testing']
    performance_name_list_formal = ['Prediction Accuracy (Training)', 'Prediction Accuracy (Testing)', 'Cross-entropy Loss (Training)', 'Cross-entropy Loss (Training)',
                                    'f1-score (Training)','f1-score (Testing)']
    #
    cm_table_df = export_performance_table(cm_table, cm_model_name_list, cm_model_name_list_formal,
                                           performance_name_list, performance_name_list_formal, '0_cm_table.csv')

    ### three visualizations
    # plot pred accuracy of cm
    util_helpers.export_performance_figure(table_df = cm_table_df, figsize = (8, 4), var_name = 'acc_testing',
                              x_tick_old = cm_model_name_list, x_tick_new = cm_model_name_list_formal,
                              xlabel = 'CM Models', ylabel = 'Prediction Accuracy', figure_name = 'cm_pred_accuracy.png')
    # plot parameters for cross entropy loss of cm
    util_helpers.export_performance_figure(table_df = cm_table_df, figsize = (8, 4), var_name = 'loss_testing',
                              x_tick_old = cm_model_name_list, x_tick_new = cm_model_name_list_formal,
                              xlabel = 'CM Models', ylabel = 'Cross-entropy Loss', figure_name = 'cm_loss.png')
    # plot monotonicity evaluation
    cm_mono_table = pd.DataFrame(cm_table).T.loc[cm_model_name_list, 10:]
    cm_mono_table.columns = ['mono']
    util_helpers.export_performance_figure(table_df = cm_mono_table, figsize = (8, 4), var_name = 'mono',
                              x_tick_old = cm_model_name_list, x_tick_new = cm_model_name_list_formal,
                              xlabel = 'CM Models', ylabel = 'Monotonicity Accuracy', figure_name = 'cm_mono.png')


    ### 2. pt
    PT_MODEL_NAME_LIST = ['pt_est']
    penalty_const_list = [1e-50, 1e-10, 1e-5, 0.0001, 0.001, 0.01, 0.1, 1.0]
    task_type = 'pt'
    for name in ['pt_dnn_', 'pt_resnet_']:
        for penalty_const in penalty_const_list:
            PT_MODEL_NAME_LIST.append(name + str(penalty_const))

    # convert to full PT table
    pt_table = output_table(pt_info, PT_MODEL_NAME_LIST, task_type)

    ### export tables
    ### table parameters
    pt_model_name_list = ['pt_dnn_1e-50', 'pt_resnet_1e-05', 'pt_resnet_0.0001', 'pt_resnet_0.01', 'pt_est']
    pt_model_name_list_formal = ['DNN', 'Resnet (1e-05)', 'Resnet (0.0001)', 'Resnet (0.01)', 'PT']

    #
    pt_table_df = export_performance_table(pt_table, pt_model_name_list, pt_model_name_list_formal,
                                           performance_name_list, performance_name_list_formal, '0_pt_table.csv')

    ### three visualizations
    # plot pred accuracy of pt
    util_helpers.export_performance_figure(table_df = pt_table_df, figsize = (8, 4), var_name = 'acc_testing',
                              x_tick_old = pt_model_name_list, x_tick_new = pt_model_name_list_formal,
                              xlabel = 'PT Models', ylabel = 'Prediction Accuracy', figure_name = 'pt_pred_accuracy.png')
    # plot parameters for cross entropy loss of pt
    util_helpers.export_performance_figure(table_df = pt_table_df, figsize = (8, 4), var_name = 'loss_testing',
                              x_tick_old = pt_model_name_list, x_tick_new = pt_model_name_list_formal,
                              xlabel = 'PT Models', ylabel = 'Cross-entropy Loss', figure_name = 'pt_loss.png')
    # plot monotonicity evaluation
    pt_mono_table = pd.DataFrame(pt_table).T.loc[pt_model_name_list, 10:]
    pt_mono_table.columns = ['mono']
    util_helpers.export_performance_figure(table_df = pt_mono_table, figsize = (8, 4), var_name = 'mono',
                              x_tick_old = pt_model_name_list, x_tick_new = pt_model_name_list_formal,
                              xlabel = 'PT Models', ylabel = 'Monotonicity Accuracy', figure_name = 'pt_mono.png')


    ### 3. hd
    HD_MODEL_NAME_LIST = ['hd_est']
    penalty_const_list = [1e-50, 1e-10, 1e-5, 0.0001, 0.001, 0.002, 0.003, 0.004, 0.005, 0.008, 0.01, 0.1, 1.0]
    task_type = 'hd'
    for name in ['hd_dnn_', 'hd_resnet_']:
        for penalty_const in penalty_const_list:
            HD_MODEL_NAME_LIST.append(name + str(penalty_const))

    # convert to full HD table
    hd_table = output_table(hd_info, HD_MODEL_NAME_LIST, task_type)

    ### export tables
    ### table parameters
    hd_model_name_list = ['hd_dnn_1e-50', 'hd_resnet_1e-05', 'hd_resnet_0.001', 'hd_resnet_0.01', 'hd_est']
    hd_model_name_list_formal = ['DNN', 'Resnet (1e-05)', 'Resnet (0.001)', 'Resnet (0.01)', 'HD']

    #
    hd_table_df = export_performance_table(hd_table, hd_model_name_list, hd_model_name_list_formal,
                                           performance_name_list, performance_name_list_formal, '0_hd_table.csv')

    ### three visualizations
    # plot pred accuracy of hd
    util_helpers.export_performance_figure(table_df = hd_table_df, figsize = (8, 4), var_name = 'acc_testing',
                              x_tick_old = hd_model_name_list, x_tick_new = hd_model_name_list_formal,
                              xlabel = 'HD Models', ylabel = 'Prediction Accuracy', figure_name = 'hd_pred_accuracy.png')
    # plot parameters for cross entropy loss of hd
    util_helpers.export_performance_figure(table_df = hd_table_df, figsize = (8, 4), var_name = 'loss_testing',
                              x_tick_old = hd_model_name_list, x_tick_new = hd_model_name_list_formal,
                              xlabel = 'HD Models', ylabel = 'Cross-entropy Loss', figure_name = 'hd_loss.png')
    # plot monotonicity evaluation
    hd_mono_table = pd.DataFrame(hd_table).T.loc[hd_model_name_list, 10:]
    hd_mono_table.columns = ['mono']
    util_helpers.export_performance_figure(table_df = hd_mono_table, figsize = (8, 4), var_name = 'mono',
                              x_tick_old = hd_model_name_list, x_tick_new = hd_model_name_list_formal,
                              xlabel = 'HD Models', ylabel = 'Monotonicity Accuracy', figure_name = 'hd_mono.png')



if __name__ == '__main__':

    get_utility()







