import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import re
from scipy.interpolate import make_interp_spline, BSpline

#import matplotlib as mpl
#mpl.use('TkAgg')

import matplotlib
import util_helpers
import pickle
import tensorflow as tf
import os
from scipy import stats
import copy
from sklearn.metrics import f1_score
from matplotlib.ticker import FormatStrFormatter



colors = sns.color_palette('Paired')



def plot_curve(task, log_loss_training_mean, log_loss_testing_mean, accuracy_training_mean, accuracy_testing_mean, y_lim_loss = None,y_lim_err =None, save_fig = 0):
    font_size = 16
    fig, ax = plt.subplots(figsize=(8, 5))
    x_epoch = np.arange(len(log_loss_training_mean)) +1

    l0 = ax.plot(x_epoch, log_loss_training_mean, color=colors[0], alpha=0.5)
    l1 = ax.plot(x_epoch, log_loss_testing_mean, color=colors[1], alpha=0.5)
    # y_ticks = list(np.arange(0.49, 0.60, 0.02))
    # ax.set_yticks(y_ticks)
    # y_tickslabel = []
    # for y in y_ticks:
    #     y_tickslabel.append(round(y, 2))
    # ax.set_yticklabels(y_tickslabel, fontsize=font_size)
    # x_ticks = np.array(x_old)[index_list]
    # new_ticks = np.arange(0, len(log_loss_training_mean), 1000) +1
    if y_lim_loss is not None:
        ax.set_ylim([y_lim_loss[0], y_lim_loss[1]])
    ax.set_xlim([-30, 5030])
    # ax.set_xticks(x_ticks)
    # ax.set_xticklabels(new_ticks, fontsize=font_size)

    ax.tick_params(axis="y", labelsize=font_size)
    ax.tick_params(axis="x", labelsize=font_size)
    ax.set_xlabel('Epoch', fontsize=font_size)
    ax.set_ylabel('Cross-entropy loss', fontsize=font_size, color =colors[1])
    #================


    ax2 = ax.twinx()  #
    # y_ticks = list(np.arange(1.1, 2.8+0.3, 0.3))
    # ax2.set_yticks(y_ticks)
    # y_tickslabel =[]
    # for y in y_ticks:
    #     y_tickslabel.append(round(y,1))
    # ax2.set_yticklabels(y_tickslabel, fontsize=font_size)
    if y_lim_loss is not None:
        ax2.set_ylim([y_lim_err[0], y_lim_err[1]])
    error_train = 1 - np.array(accuracy_training_mean)
    error_test = 1 - np.array(accuracy_testing_mean)
    l2 = ax2.plot(x_epoch, error_train, color=colors[2])
    l3 = ax2.plot(x_epoch, error_test, color=colors[3])
    ax2.tick_params(axis="y", labelsize=font_size)
    ax2.set_ylabel('Prediction errors', fontsize=font_size, color =colors[3])

    lns = l0 + l1 + l2 +  l3
    labs = ['Training loss','Testing loss', 'Training error','Testing error']
    plt.legend(lns, labs, fontsize=font_size - 3, ncol = 2)
    plt.tight_layout()
    if save_fig == 1:
        plt.savefig('output/performance/training_curve_' + task + '.png', dpi=200)
    else:
        plt.show()



def proces_data_for_plot(save_fig= 0):
    #
    with open('cm_info_use_delta_save_train.pickle', 'rb') as f:
        cm_info = pickle.load(f)
    with open('pt_info_use_delta_save_train.pickle', 'rb') as f:
        pt_info = pickle.load(f)
    with open('hd_info_use_delta_save_train.pickle', 'rb') as f:
        hd_info = pickle.load(f)

    with open('data/process/cm.pickle', 'rb') as data:
        cm_data = pickle.load(data)
    with open('data/process/risk.pickle', 'rb') as data:
        pt_data = pickle.load(data)
    with open('data/process/time.pickle', 'rb') as data:
        hd_data = pickle.load(data)

    ###

    ###

    ########################################process cm_data###########################
    y_vars = ['choice']
    cm_y_training = cm_data['training'][y_vars].values[:, 0]
    cm_y_testing = cm_data['testing'][y_vars].values[:, 0]
    ########################################process hd_data###########################
    training_df = hd_data['training']
    testing_df = hd_data['testing']

    # reverse the choice indicator from the initial Tanaka paper
    # new alternative: x1 has time dimension.
    training_df['choice'] = 1.0 - training_df['choice']
    testing_df['choice'] = 1.0 - testing_df['choice']
    y_var = ['choice']
    hd_y_training = np.int_(training_df[y_var].values[:, 0])
    hd_y_testing = np.int_(testing_df[y_var].values[:, 0])
    ########################################process pt_data###########################
    ## 1. pre-process data for PT

    y_var = ['choice']
    training_df = pt_data['training']
    testing_df = pt_data['testing']
    pt_y_training = training_df[y_var].values[:, 0]
    pt_y_testing = testing_df[y_var].values[:, 0]

    for task_type in ['cm', 'pt', 'hd']:

        assert task_type in ['cm', 'pt', 'hd']
        #
        table = {}

        if task_type == 'cm':
            MODEL_NAME_LIST = ['cm_resnet_0.008']
            info = cm_info.copy()
        elif task_type == 'hd':
            MODEL_NAME_LIST = ['hd_resnet_0.05']
            info = hd_info.copy()
        else:
            MODEL_NAME_LIST = ['pt_resnet_0.9']
            info = pt_info.copy()

        for MODEL_NAME in MODEL_NAME_LIST:
            num_replications = len(info[MODEL_NAME]['data_output'])
            # print(MODEL_NAME)
            acc_test_list = []
            acc_training_list = []
            cost_training_list = []
            cost_testing_list = []
            log_loss_training_list = []
            log_loss_testing_list = []
            f1_score_training_list = []
            f1_score_testing_list = []
            if MODEL_NAME not in info:
                print(MODEL_NAME, 'is not evaluated, skip it')
                continue
            # model load training process. try otherwise rerun


            for replic in range(num_replications):
                # print(info[MODEL_NAME]['data_output'][replic]['accuracy_testing'])
                acc_test_list.append(info[MODEL_NAME]['data_output'][replic]['accuracy_testing'])
                acc_training_list.append(info[MODEL_NAME]['data_output'][replic]['accuracy_training'])
                log_loss_training_list.append(info[MODEL_NAME]['data_output'][replic]['log_loss_training'])
                log_loss_testing_list.append(info[MODEL_NAME]['data_output'][replic]['log_loss_testing'])

            acc_test_list_np = np.array(acc_test_list)
            acc_training_list_np = np.array(acc_training_list)
            log_loss_training_list_np = np.array(log_loss_training_list)
            log_loss_testing_list_np = np.array(log_loss_testing_list)

            a=1

            accuracy_training_mean = np.mean(acc_training_list_np,axis=0)
            accuracy_testing_mean = np.mean(acc_test_list_np,axis=0)
            log_loss_training_mean = np.mean(log_loss_training_list_np,axis=0)
            log_loss_testing_mean = np.mean(log_loss_testing_list_np,axis=0)
            if task_type == 'cm':
                y_lim_loss = [0.5,1.5]
                y_lim_err = [0.2,0.7]
            elif task_type == 'hd':
                y_lim_loss = [0.4, 0.8]
                y_lim_err = [0.1, 0.6]
            else:
                y_lim_loss =[0.0, 0.9]
                y_lim_err =  [0,0.6]

            plot_curve(task_type,log_loss_training_mean, log_loss_testing_mean, accuracy_training_mean, accuracy_testing_mean, y_lim_loss,y_lim_err, save_fig = save_fig)


if __name__ == '__main__':
    proces_data_for_plot(save_fig= 1)
    a=1
