#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 10:08:15 2019

util_helpers

@author: shenhao
"""

import numpy as np
import pandas as pd
import matplotlib
#mpl.use('TkAgg')
import matplotlib.pyplot as plt
import copy
import util_functions as util
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import MaxNLocator

def generate_mesh_data(n_grid, x_min, x_max, y_min, y_max):
    '''
    generate a mesh dataset for simulation
    '''
    x0 = np.linspace(x_min, x_max, n_grid)
    x1 = np.linspace(y_min, y_max, n_grid)
    x0_grid, x1_grid = np.meshgrid(x0, x1)
    mesh = np.vstack([x0_grid.ravel(), x1_grid.ravel()]).T
    data = pd.DataFrame(mesh, columns = ['x0', 'x1'])
    # randomnize some choices to fit estimate_dgp1_with_true_param functions
    return data

def export_performance_figure(table_df, figsize, var_name, x_tick_old, x_tick_new, xlabel, ylabel, figure_name):
    '''
    save the performance figure
    '''
    matplotlib.rcParams.update({'font.size': 15})
    plt.figure(figsize = figsize)
    plt.plot(table_df[var_name], marker = "o", linewidth = 3)
    plt.plot([0], [table_df[var_name][0]], marker = "o", color = 'r')
    plt.plot([len(x_tick_old)-1], [table_df[var_name][-1]], marker = "o", color = 'r')
    plt.xticks(x_tick_old, x_tick_new, rotation = 45)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig("output/performance/"+figure_name)
    plt.close()


def visualize_contour(X1_mesh, X2_mesh, Z, figsize, linewidths, xlabel, ylabel, figure_name):
    '''
    X1_mesh, X2_mesh are two grid meshes
    Z is the target variable to be visualized
    '''
    matplotlib.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(figsize = figsize)
    contours = ax.contour(X1_mesh, X2_mesh, Z, cmap='winter', linewidths = linewidths)
    plt.clabel(contours, inline=True)
#    plt.colorbar(contours)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
#    fig.suptitle(title, fontsize = 20)
    plt.savefig('output/interpretation/'+figure_name)
    plt.close()

def export_one_row_figure(one_row, figsize, xlabel, ylabel, figure_name):
    '''
    export one row/column of the field. Just one list data
    '''
    matplotlib.rcParams.update({'font.size': 20})

    fig, ax = plt.subplots(figsize = figsize)

    plt.plot(one_row, linewidth = 3)
#    plt.xticks(x_tick_old, x_tick_new, rotation = 45)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    y_max = np.max(one_row)
    y_min = np.min(one_row)
    y_lim = [np.floor(y_min), np.ceil(y_max)]
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    # ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.ylim(y_lim)
    plt.yticks(y_lim)
    plt.xticks([0,100])
    plt.tight_layout()
    plt.savefig("output/interpretation/"+figure_name)
    plt.close()
    
def fgsm_adv_examples(data, data_len_list, epsilon, gradient_training, gradient_testing):
    '''
    Use fast gradient sign methods to generate adversarial examples
    data is a dictionary with training and testing
    data_len_list: the list of lenth of the dataframes in data[training] and data[testing] (excluding y).
    gradients: gradients of log-loss w.r.t. inputs X. 
    Output:
        adversarial examples data_adv, which has the same format as data.
    '''
    assert np.sum(data_len_list) == gradient_training.shape[1]
    assert np.sum(data_len_list) == gradient_testing.shape[1]
    
    data_adv = {}
    data_adv['training'] = []
    data_adv['testing'] = []

    col_count = 0
    for idx in range(len(data_len_list)):
        # data_adv['training'].append(data['training'][idx]+epsilon*np.sign(gradient_training[:, col_count:col_count+data_len_list[idx]]))
        data_adv['training'].append(
            data['training'][idx])
        data_adv['testing'].append(data['testing'][idx]+epsilon*np.sign(gradient_testing[:, col_count:col_count+data_len_list[idx]]))
        #
        col_count += data_len_list[idx]
    # attach y
    data_adv['training'].append(data['training'][-1])
    data_adv['testing'].append(data['testing'][-1])
    # 
    return data_adv


def GN_adv_examples(data, data_len_list, epsilon):
    '''
    Use fast gradient sign methods to generate adversarial examples
    data is a dictionary with training and testing
    data_len_list: the list of lenth of the dataframes in data[training] and data[testing] (excluding y).
    gradients: gradients of log-loss w.r.t. inputs X.
    Output:
        adversarial examples data_adv, which has the same format as data.
    '''

    # idx = 1
    # print(data['training'][idx])
    data_adv = {}
    data_adv['training'] = []
    data_adv['testing'] = []

    col_count = 0
    for idx in range(len(data_len_list)):
        data_adv['training'].append(data['training'][idx])
        # data_adv['training'].append(
        #     data['training'][idx] + epsilon * np.random.normal(0, 1, size = (data['training'][idx].shape[0], data_len_list[idx])))
        data_adv['testing'].append(
            data['testing'][idx] + epsilon * np.random.normal(0, 1,  size = (data['testing'][idx].shape[0], data_len_list[idx])))
        #
        col_count += data_len_list[idx]
    # attach y
    data_adv['training'].append(data['training'][-1])
    data_adv['testing'].append(data['testing'][-1])
    #
    return data_adv


def clip(data_new, data_initial, data_len_list, epsilon):
    '''
    data has data['training'] and data['testing']
    Function clips data_new to data_initial +/- epsilon
    '''
    clipped_data_new = copy.copy(data_new)
    
    for idx in range(len(data_len_list)):
        clipped_data_new['training'][idx] = np.clip(data_new['training'][idx], 
                        a_min = data_initial['training'][idx]-epsilon, a_max = data_initial['training'][idx]+epsilon)
        clipped_data_new['testing'][idx] = np.clip(data_new['testing'][idx], 
                        a_min = data_initial['testing'][idx]-epsilon, a_max = data_initial['testing'][idx]+epsilon)
    return clipped_data_new


### new targets by tgsm
def tgsm_target_gradients(data, n_class, MODEL_NAME, fun, l2_regu = None, penalty_const = None, param_dic = None):
    '''
    return the tgsm gradients for the data set
    the target class the the true class + 1
    '''
    data_with_new_target = {}
    data_with_new_target['training'] = copy.copy(data['training'])
    data_with_new_target['testing'] = copy.copy(data['testing'])
    y_target_training = (data_with_new_target['training'][-1] + 1) % n_class # +1 as the target class
    y_target_testing = (data_with_new_target['testing'][-1] + 1) % n_class # +1 as the target class
    # replace the y in data_with_new_target
    data_with_new_target['training'][-1] = y_target_training
    data_with_new_target['testing'][-1] = y_target_testing
    #
    restore = True
    n_epoches = 5000

    if fun in [util.est_cm, util.est_pt, util.est_hd]:
        _,_,target_output_dic = fun(data_with_new_target, MODEL_NAME, restore, n_epoches)
    elif fun in [util.est_dnn]:
        _,_,target_output_dic = fun(data_with_new_target, l2_regu, MODEL_NAME, restore, n_epoches, K = n_class)
    elif fun in [util.est_cm_resnet, util.est_pt_resnet, util.est_hd_resnet]:
        _,_,target_output_dic = fun(data_with_new_target, param_dic, penalty_const, MODEL_NAME, restore, n_epoches,  K = n_class)
    elif fun in [util.est_cm_resnet_reverse, util.est_pt_resnet_reverse, util.est_hd_resnet_reverse]:
        _,_,target_output_dic = fun(data_with_new_target, param_dic, penalty_const, MODEL_NAME, restore, n_epoches,  K = n_class)
    elif fun in [util.est_cm_resnet_simultaneous, util.est_pt_resnet_simultaneous, util.est_hd_resnet_simultaneous]:
        _,_,target_output_dic = fun(data_with_new_target, penalty_const, MODEL_NAME, restore, n_epoches,  K = n_class)
    elif fun in [util.est_cm_resnet_simultaneous_use_delta, util.est_pt_resnet_simultaneous_use_delta, util.est_hd_resnet_simultaneous_use_delta]:
        _, _, target_output_dic = fun(data_with_new_target, penalty_const, MODEL_NAME, restore, n_epoches, K=n_class)
    elif fun in [util.est_cm_resnet_use_delta, util.est_pt_resnet_use_delta, util.est_hd_resnet_use_delta]:
        _,_,target_output_dic = fun(data_with_new_target, param_dic, penalty_const, MODEL_NAME, restore, n_epoches,  K = n_class)
    #
    gradient_target_training = target_output_dic['gradient_cost_x_training']
    gradient_target_testing = target_output_dic['gradient_cost_x_testing']
    #
    return gradient_target_training,gradient_target_testing


def tgsm_adv_examples(data, data_len_list, epsilon, gradient_training, gradient_testing):
    '''
    Use target gradient sign methods to generate adversarial examples
    data is a dictionary with training and testing
    data_len_list: the list of lenth of the dataframes in data[training] and data[testing] (excluding y).
    gradients: gradients of log-loss of target y w.r.t. inputs X. 
    Output:
        adversarial examples data_adv, which has the same format as data.
    '''    
    assert np.sum(data_len_list) == gradient_training.shape[1]
    assert np.sum(data_len_list) == gradient_testing.shape[1]
    # 
    data_adv = {}
    data_adv['training']=[]
    data_adv['testing']=[]
    # 
    col_count = 0
    for idx in range(len(data_len_list)):
        # data_adv['training'].append(data['training'][idx] - epsilon*np.sign(gradient_training[:, col_count:col_count+data_len_list[idx]]))
        data_adv['training'].append(
            data['training'][idx])
        data_adv['testing'].append(data['testing'][idx] - epsilon*np.sign(gradient_testing[:, col_count:col_count+data_len_list[idx]]))
        col_count += data_len_list[idx]
    data_adv['training'].append(data['training'][-1])
    data_adv['testing'].append(data['testing'][-1])
    #
    return data_adv



def fgsm_adv_iter_examples():
    '''
    iterative attacks by fgsm
    '''
    pass


def tgsm_adv_iter_examples():
    '''
    iterative attacks by tgsm
    '''
    pass














