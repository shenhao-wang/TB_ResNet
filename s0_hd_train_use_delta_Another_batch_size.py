"""
Created on Thu May 23 19:23:38 2019

s2_hyper_training_time

@author: shenhao
"""

#cd "/Users/shenhao/Dropbox (MIT)/Shenhao_Jinhua (1)/10_ut_resnet/code"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import util_functions as util
import util_helpers
import copy


def model(HD, DNN, RESNET, Load_old_results):
    #with open('time.pickle', 'rb') as data:
    #    data_dic = pickle.load(data)
    with open('data/process/time.pickle', 'rb') as data:
        data_dic = pickle.load(data)

    ###
    num_replications = 1
    ###

    training_df = data_dic['training']
    testing_df = data_dic['testing']

    # reverse the choice indicator from the initial Tanaka paper
    # new alternative: x1 has time dimension.
    training_df['choice'] = 1.0 - training_df['choice']
    testing_df['choice'] = 1.0 - testing_df['choice']
    #print(training_df.columns)


    ### preprocessing data
    z_var_names = ['chinese', 'moneykeeper', 'age', 'gender','edu', 'income', 'market', 'south', 'payment_g2']
    y_var = ['choice']
    x0_var = ['reward']
    x1_var = ['y']
    t1_var = ['t']

    scaler = StandardScaler()
    D_z = len(z_var_names)

    z_training = scaler.fit_transform(training_df[z_var_names].values)
    y_training = np.int_(training_df[y_var].values[:,0])
    x0_training = training_df[x0_var].values
    x1_training = training_df[x1_var].values
    t1_training = training_df[t1_var].values
    x_training = np.concatenate([x0_training, x1_training, t1_training, z_training], axis = 1)
    #zeros_training = np.zeros((training_df.shape[0], 2))

    z_testing = scaler.fit_transform(testing_df[z_var_names].values)
    y_testing = np.int_(testing_df[y_var].values[:,0])
    x0_testing = testing_df[x0_var].values
    x1_testing = testing_df[x1_var].values
    t1_testing = testing_df[t1_var].values
    x_testing = np.concatenate([x0_testing, x1_testing, t1_testing, z_testing], axis = 1)
    #zeros_testing = np.zeros((testing_df.shape[0], 2))
    x_dim = x_training.shape[1]

    # simulation data
    n_grid = 100
    z_simul_1 = np.zeros((n_grid**2, D_z))
    x0_simul_1 = np.tile(np.median(x0_testing, 0), [n_grid**2, 1])
    x1_simul_1 = np.tile(np.median(x1_testing, 0), [n_grid**2, 1])
    t1_simul_1 = np.tile(np.median(t1_testing, 0), [n_grid**2, 1])
    y_simul = np.zeros(n_grid**2)
    # change values
    x1_min = 0.0
    x1_max = 3.0
    t1_min = 0.0
    t1_max = 10.0
    #
    x1_t1 = util_helpers.generate_mesh_data(n_grid, x1_min, x1_max, t1_min, t1_max)
    # replace values
    x1_simul_1[:, 0]=x1_t1.values[:, 0]
    t1_simul_1[:, 0]=x1_t1.values[:, 1]
    # save
    data_grid = {}
    data_grid['training']={}
    data_grid['testing']={}
    #
    data_grid['training'] = x0_simul_1, x1_simul_1, t1_simul_1, z_simul_1, y_simul
    data_grid['testing'] = copy.copy(data_grid['training'])


    ################################################
    # meta
    if Load_old_results:
        with open('hd_info_use_delta.pickle', 'rb') as data:
            hd_info = pickle.load(data)
    else:
        hd_info = {}



    #penalty_const_list = [0.01]
    penalty_const_list = [0.05]

    mini_batch_size_list = [64,128]

    #epsilon_list = [0.1]
    epsilon_list = [0.0, 0.01, 0.03, 0.05, 0.08, 0.1, 0.2, 0.5]
    n_epoches = 5000
    K = 2 # num of classes
    n_class = 2 # same as K.


    ################################################
    ### 1. hd_est
    if HD:
        data = {}
        data['training'] = [x0_training, x1_training, t1_training, z_training, y_training]
        data['testing'] = [x0_testing, x1_testing, t1_testing, z_testing, y_testing]
        MODEL_NAME = 'hd_est'

        # 1. train hd_est
        restore = False

        hd_est_params_dic = []
        hd_est_hyper_params_dic = []
        hd_est_output_dic = []
        acc_test = []
        current_best_acc = 0
        for replica in range(num_replications):
            _hd_est_params_dic,_hd_est_hyper_params_dic,_hd_est_output_dic = util.est_hd(data, MODEL_NAME, restore, n_epoches,
                                                                                      current_best_acc)
            hd_est_params_dic.append(_hd_est_params_dic)
            hd_est_hyper_params_dic.append(_hd_est_hyper_params_dic)
            hd_est_output_dic.append(_hd_est_output_dic)
            acc_test.append(_hd_est_output_dic['accuracy_testing'][-1])
            current_best_acc = _hd_est_output_dic['current_best_acc']

        best_index_hd = acc_test.index(max(acc_test))



        # 2. restore and obtain grid info
        restore = True
        _,_,hd_est_grid_output_dic = util.est_hd(data_grid, MODEL_NAME, restore, n_epoches)



        hd_info[MODEL_NAME] = {}
        hd_info[MODEL_NAME]['data_output']=hd_est_output_dic
        hd_info[MODEL_NAME]['grid_output']=hd_est_grid_output_dic



    ################################################
    ### 2. hd_dnn
    if DNN:
        data = {}
        data['training'] = [np.concatenate([x0_training, x1_training, t1_training, z_training], axis = 1), y_training]
        data['testing'] = [np.concatenate([x0_testing, x1_testing, t1_testing, z_testing], axis = 1), y_testing]
        data_grid = {}
        data_grid['training'] = [np.concatenate([x0_simul_1, x1_simul_1, t1_simul_1, z_simul_1], axis = 1), y_simul]
        data_grid['testing'] = copy.copy(data_grid['training'])
        l2_reg = 1e-50


        MODEL_NAME = 'hd_dnn'
        n_epoches = 5000

        # 1. estimate dnn
        restore = False

        hd_dnn_params_dic = []
        hd_dnn_hyper_params_dic = []
        hd_dnn_output_dic = []
        acc_test = []
        current_best_acc = 0
        for replica in range(num_replications):
            _hd_dnn_params_dic,_hd_dnn_hyper_params_dic,_hd_dnn_output_dic = util.est_dnn(data, l2_reg, MODEL_NAME, restore, n_epoches,
                                                                                      current_best_acc = current_best_acc)
            hd_dnn_params_dic.append(_hd_dnn_params_dic)
            hd_dnn_hyper_params_dic.append(_hd_dnn_hyper_params_dic)
            hd_dnn_output_dic.append(_hd_dnn_output_dic)
            acc_test.append(_hd_dnn_output_dic['accuracy_testing'][-1])
            current_best_acc = _hd_dnn_output_dic['current_best_acc']

        best_index = acc_test.index(max(acc_test))



        # 2. return grid info
        restore = True
        _,_,hd_dnn_grid_output_dic = util.est_dnn(data_grid, l2_reg, MODEL_NAME, restore, n_epoches)

        ###
        hd_info[MODEL_NAME] = {}
        hd_info[MODEL_NAME]['data_output']=hd_dnn_output_dic
        hd_info[MODEL_NAME]['grid_output']=hd_dnn_grid_output_dic

    ################################################
    ### 3. hd_resnet
    if RESNET:
        data = {}
        data['training'] = [x0_training, x1_training, t1_training, z_training, y_training]
        data['testing'] = [x0_testing, x1_testing, t1_testing, z_testing, y_testing]
        data_grid = {}
        data_grid['training'] = [x0_simul_1, x1_simul_1, t1_simul_1, z_simul_1, y_simul]
        data_grid['testing'] = copy.copy(data_grid['training'])
        hd_param_dic = copy.copy(hd_est_params_dic[best_index_hd])
        for mini_batch_size in mini_batch_size_list:
            for penalty_const in penalty_const_list:
                MODEL_NAME = 'hd_resnet_'+str(penalty_const)

                # 1. estimate hd_resnet
                restore = False

                hd_resnet_params_dic = []
                hd_resnet_hyper_params_dic = []
                hd_resnet_output_dic = []
                acc_test = []
                current_best_acc = 0
                for replica in range(num_replications):
                    _hd_resnet_params_dic, _hd_resnet_hyper_params_dic, _hd_resnet_output_dic = util.est_hd_resnet_use_delta(data, hd_param_dic,
                                                                                                                penalty_const,
                                                                                                                MODEL_NAME, restore,
                                                                                                                n_epoches, current_best_acc=current_best_acc, n_mini_batch = mini_batch_size)
                    hd_resnet_params_dic.append(_hd_resnet_params_dic)
                    hd_resnet_hyper_params_dic.append(_hd_resnet_hyper_params_dic)
                    hd_resnet_output_dic.append(_hd_resnet_output_dic)
                    acc_test.append(_hd_resnet_output_dic['accuracy_testing'][-1])
                    current_best_acc = _hd_dnn_output_dic['current_best_acc']

                best_index = acc_test.index(max(acc_test))


                # 2. restore hd_resnet
                restore = True
                _,_,hd_resnet_grid_output_dic = util.est_hd_resnet_use_delta(data_grid, hd_param_dic, penalty_const, MODEL_NAME, restore, n_epoches)


                hd_info[MODEL_NAME] = {}
                hd_info[MODEL_NAME]['data_output']=hd_resnet_output_dic
                hd_info[MODEL_NAME]['grid_output']=hd_resnet_grid_output_dic

    #
            with open('hd_info_use_delta_batch_size_' + str(mini_batch_size) + '.pickle', 'wb') as f:
                pickle.dump(hd_info, f)

if __name__ == '__main__':
    HD = True
    DNN = True
    RESNET = True
    Load_old_results = False
    model(HD, DNN, RESNET, Load_old_results)






























