"""
Created on Thu May 23 13:36:14 2019

s2_hyper_training
Previous Findings:
    Check archive_190523

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
import copy
import util_helpers


def model(PT, DNN, RESNET, Load_old_results):


    with open('data/process/risk.pickle', 'rb') as data:
        data_dic = pickle.load(data)
    #with open('../data/process/risk.pickle', 'rb') as data:
    #    data_dic = pickle.load(data)

    training_df = data_dic['training']
    testing_df = data_dic['testing']

    ###
    num_replications = 1
    ###

    ### preprocessing data
    ## 1. pre-process data for PT
    z_var_names  = ['age','gender','edu','income','market','south','chinese']
    x0_var_names = ['x00', 'x01']
    p0_var_names = ['p00', 'p01']
    x1_var_names = ['x10', 'x11']
    p1_var_names = ['p10', 'p11']
    y_var = ['choice']

    scaler = StandardScaler()
    D_z = len(z_var_names)

    # training
    z_training = scaler.fit_transform(training_df[z_var_names].values)
    x0_training = training_df[x0_var_names].values
    p0_training = training_df[p0_var_names].values
    x1_training = training_df[x1_var_names].values
    p1_training = training_df[p1_var_names].values
    y_training = training_df[y_var].values[:,0]
    #testing
    z_testing = scaler.fit_transform(testing_df[z_var_names].values)
    x0_testing = testing_df[x0_var_names].values
    p0_testing = testing_df[p0_var_names].values
    x1_testing = testing_df[x1_var_names].values
    p1_testing = testing_df[p1_var_names].values
    y_testing = testing_df[y_var].values[:,0]

    ## 2. pre-process data for DNN
    z_var_names  = ['age','gender','edu','income','market','south','chinese']
    x_p_var_names = ['x00', 'x01', 'p00', 'p01', 'x10', 'x11', 'p10', 'p11']
    y_var = ['choice']
    #
    scaler = StandardScaler()
    # training
    z_training = scaler.fit_transform(training_df[z_var_names].values)
    x_p_training = training_df[x_p_var_names].values
    y_training = training_df[y_var].values[:,0]
    x_training = np.concatenate((x_p_training, z_training), axis = 1)
    # testing
    z_testing = scaler.fit_transform(testing_df[z_var_names].values)
    x_p_testing = testing_df[x_p_var_names].values
    y_testing = testing_df[y_var].values[:,0]
    x_testing = np.concatenate([x_p_testing, z_testing], axis = 1)

    ## 3. simulation datasets
    # change x1 and p1 values based on the testing set
    n_grid = 100
    z_simul_1 = np.zeros((n_grid**2, D_z))
    x0_simul_1 = np.tile(np.median(x0_testing, 0), [n_grid**2, 1])
    x1_simul_1 = np.tile(np.median(x1_testing, 0), [n_grid**2, 1])
    p0_simul_1 = np.tile(np.median(p0_testing, 0), [n_grid**2, 1]) # (1,0)
    p1_simul_1 = np.tile(np.median(p1_testing, 0), [n_grid**2, 1])
    y_simul = np.zeros(n_grid**2)
    # change values
    x1_min = 0.0
    x1_max = 20.0
    p1_min = 0.0
    p1_max = 1.0
    #
    x1_p1 = util_helpers.generate_mesh_data(n_grid, x1_min, x1_max, p1_min, p1_max)
    # replace values
    x1_simul_1[:, 0]=x1_p1.values[:, 0]
    p1_simul_1[:, 0]=x1_p1.values[:, 1]
    p1_simul_1[:, 1]=1.0 - p1_simul_1[:, 0]
    # save
    data_grid = {}
    data_grid['training']={}
    data_grid['testing']={}
    #
    data_grid['training'] = [x0_simul_1,x1_simul_1,p0_simul_1,p1_simul_1,z_simul_1,y_simul]
    data_grid['testing'] = copy.copy(data_grid['training'])


    ########### training
    # meta
    if Load_old_results:
        with open('pt_info_use_delta.pickle', 'rb') as data:
            pt_info = pickle.load(data)
    else:
        pt_info = {}

    penalty_const_list = [0.9]
    #epsilon_list = [0.1]

    #epsilon_list = [0.1]
    n_epoches = 5000
    K = 2 # num of classes
    n_class = 2 # same as K.

    ### 1. pt
    if PT:
        data = {}
        data['training'] = [x0_training, x1_training, p0_training, p1_training, z_training, y_training]
        data['testing'] = [x0_testing, x1_testing, p0_testing, p1_testing, z_testing, y_testing]
        MODEL_NAME = 'pt_est'

        # 1. train pt_est
        restore = False

        pt_est_params_dic = []
        pt_est_hyper_params_dic = []
        pt_est_output_dic = []
        acc_test = []
        current_best_acc = 0
        for replica in range(num_replications):
            _pt_est_params_dic,_pt_est_hyper_params_dic,_pt_est_output_dic = util.est_pt(data, MODEL_NAME, restore, n_epoches,
                                                                                      current_best_acc=current_best_acc)
            pt_est_params_dic.append(_pt_est_params_dic)
            pt_est_hyper_params_dic.append(_pt_est_hyper_params_dic)
            pt_est_output_dic.append(_pt_est_output_dic)
            acc_test.append(_pt_est_output_dic['accuracy_testing'][-1])
            current_best_acc = _pt_est_output_dic['current_best_acc']

        best_index_pt = acc_test.index(max(acc_test))


        # 2. restore and obtain grid info
        restore = True
        _,_,pt_est_grid_output_dic = util.est_pt(data_grid, MODEL_NAME, restore, n_epoches)

        #print(pt_est_output_dic['prob_training'].shape)
        #print(pt_est_grid_output_dic['prob_training'].shape)
        pt_info[MODEL_NAME] = {}
        pt_info[MODEL_NAME]['data_output']=pt_est_output_dic
        pt_info[MODEL_NAME]['grid_output']=pt_est_grid_output_dic


    # ### 2. dnn
    if DNN:
        data = {}
        data['training']={}
        data['testing']={}
        data['training'] =[x_training,y_training]
        data['testing'] = [x_testing,y_testing]
        data_grid_dnn = {}
        data_grid_dnn['training'] = [np.concatenate([x0_simul_1,x1_simul_1,p0_simul_1,p1_simul_1,z_simul_1], axis = 1), y_simul]
        data_grid_dnn['testing'] = copy.copy(data_grid_dnn['training'])
        #




        # 1. estimate dnn
        MODEL_NAME = 'pt_dnn'
        restore = False
        l2_reg = 1e-50
        pt_dnn_params_dic = []
        pt_dnn_hyper_params_dic = []
        pt_dnn_output_dic = []
        acc_test = []
        current_best_acc = 0
        for replica in range(num_replications):
            _pt_dnn_params_dic,_pt_dnn_hyper_params_dic,_pt_dnn_output_dic = util.est_dnn(data, l2_reg, MODEL_NAME, restore, n_epoches,
                                                                                       current_best_acc=current_best_acc)
            pt_dnn_params_dic.append(_pt_dnn_params_dic)
            pt_dnn_hyper_params_dic.append(_pt_dnn_hyper_params_dic)
            pt_dnn_output_dic.append(_pt_dnn_output_dic)
            acc_test.append(_pt_dnn_output_dic['accuracy_testing'][-1])
            current_best_acc = _pt_dnn_output_dic['current_best_acc']

        best_index = acc_test.index(max(acc_test))



        # 2. return grid info
        restore = True
        _,_,pt_dnn_grid_output_dic = util.est_dnn(data_grid_dnn, l2_reg, MODEL_NAME, restore, n_epoches)
        #



        # save
        pt_info[MODEL_NAME] = {}
        pt_info[MODEL_NAME]['data_output']=pt_dnn_output_dic
        pt_info[MODEL_NAME]['grid_output']=pt_dnn_grid_output_dic

    ### 3. pt_resnet
    if RESNET:
        data = {}
        data['training']= [x0_training, x1_training, p0_training, p1_training, z_training, y_training]
        data['testing']= [x0_testing, x1_testing, p0_testing, p1_testing, z_testing, y_testing]
        pt_param_dic = copy.copy(pt_est_params_dic[best_index_pt])

        for penalty_const in penalty_const_list:
        #    penalty_const = 0.01
            MODEL_NAME = 'pt_resnet_'+str(penalty_const)

            # 1. estimate pt_resnet
            restore = False

            pt_resnet_params_dic = []
            pt_resnet_hyper_params_dic = []
            pt_resnet_output_dic = []
            acc_test = []
            current_best_acc = 0
            for replica in range(num_replications):
                _pt_resnet_params_dic, _pt_resnet_hyper_params_dic, _pt_resnet_output_dic = util.est_pt_resnet_use_delta_save_train(data, pt_param_dic,
                                                                                                            penalty_const,
                                                                                                            MODEL_NAME, restore,
                                                                                                            n_epoches,
                                                                                                            current_best_acc=current_best_acc)
                pt_resnet_params_dic.append(_pt_resnet_params_dic)
                pt_resnet_hyper_params_dic.append(_pt_resnet_hyper_params_dic)
                pt_resnet_output_dic.append(_pt_resnet_output_dic)
                acc_test.append(_pt_resnet_output_dic['accuracy_testing'][-1])
                current_best_acc = _pt_resnet_output_dic['current_best_acc']

            best_index = acc_test.index(max(acc_test))



            # 2. restore
            restore = True
            _,_,pt_resnet_grid_output_dic = util.est_pt_resnet_use_delta_save_train(data_grid, pt_param_dic, penalty_const, MODEL_NAME, restore, n_epoches)



            pt_info[MODEL_NAME] = {}
            pt_info[MODEL_NAME]['data_output']=pt_resnet_output_dic
            pt_info[MODEL_NAME]['grid_output']=pt_resnet_grid_output_dic

#
        with open('pt_info_use_delta_save_train.pickle', 'wb') as f:
            pickle.dump(pt_info, f)



if __name__ == '__main__':
    PT = True
    DNN = True
    RESNET = True
    Load_old_results = False
    model(PT, DNN, RESNET, Load_old_results)




















