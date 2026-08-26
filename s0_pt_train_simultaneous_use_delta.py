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


def model(RESNET, Load_old_results):
    with open('data/process/risk.pickle', 'rb') as data:
        data_dic = pickle.load(data)
    #with open('../data/process/risk.pickle', 'rb') as data:
    #    data_dic = pickle.load(data)

    training_df = data_dic['training']
    testing_df = data_dic['testing']


    ###
    num_replications = 10
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

    penalty_const_list = [1e-10,1e-8, 1e-7,1e-6,1e-5, 1e-4, 0.001, 0.002, 0.004, 0.005, 0.006,0.007,
                          0.008,0.009, 0.01,0.03, 0.05, 0.1, 0.3, 0.5, 0.8, 0.9,0.95,0.99,0.999,0.9999, 1]
    #epsilon_list = [0.1]
    epsilon_list = [0.0, 0.01, 0.03, 0.05, 0.08, 0.1, 0.2, 0.5]
    n_epoches = 5000
    K = 2 # num of classes
    n_class = 2 # same as K.

    ### 1. pt




    ### 3. pt_resnet
    if RESNET:
        data = {}
        data['training']= [x0_training, x1_training, p0_training, p1_training, z_training, y_training]
        data['testing']= [x0_testing, x1_testing, p0_testing, p1_testing, z_testing, y_testing]


        for penalty_const in penalty_const_list:
        #    penalty_const = 0.01
            MODEL_NAME = 'pt_resnet_simul'+str(penalty_const)

            # 1. estimate pt_resnet
            restore = False

            pt_resnet_params_dic = []
            pt_resnet_hyper_params_dic = []
            pt_resnet_output_dic = []
            acc_test = []
            current_best_acc = 0
            for replica in range(num_replications):
                _pt_resnet_params_dic, _pt_resnet_hyper_params_dic, _pt_resnet_output_dic = util.est_pt_resnet_simultaneous_use_delta(data,
                                                                                                                         penalty_const,
                                                                                                                         MODEL_NAME,
                                                                                                                         restore,
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
            _,_,pt_resnet_grid_output_dic = util.est_pt_resnet_simultaneous_use_delta(data_grid, penalty_const, MODEL_NAME, restore, n_epoches)

            ### attacks
            # 3. fgsm
            pt_resnet_adv_fgsm_output_dic = {}
            for epsilon in epsilon_list:
                print('fgsm ', epsilon)
                data_len_list = [len(x0_var_names), len(x1_var_names), len(p0_var_names), len(p1_var_names), len(z_var_names)]
                gradient_training = pt_resnet_output_dic[best_index]['gradient_cost_x_training']
                gradient_testing = pt_resnet_output_dic[best_index]['gradient_cost_x_testing']
                data_adv_fgsm = util_helpers.fgsm_adv_examples(data, data_len_list, epsilon, gradient_training, gradient_testing)
                restore = True
                _,_,pt_resnet_adv_fgsm_output=util.est_pt_resnet_simultaneous_use_delta(data_adv_fgsm, penalty_const, MODEL_NAME, restore, n_epoches = n_epoches, K = K)
                pt_resnet_adv_fgsm_output_dic[epsilon] = {'accuracy_training': pt_resnet_adv_fgsm_output['accuracy_training'],
                                                          'accuracy_testing': pt_resnet_adv_fgsm_output['accuracy_testing'],
                                                          'log_loss_training': pt_resnet_adv_fgsm_output['log_loss_training'],
                                                          'log_loss_testing': pt_resnet_adv_fgsm_output['log_loss_testing']
                                                          }

            # 4. tgsm
            pt_resnet_adv_tgsm_output_dic = {}
            for epsilon in epsilon_list:
                print('tgsm ', epsilon)
                gradient_target_training,gradient_target_testing = util_helpers.tgsm_target_gradients(data, n_class, MODEL_NAME,
                                                                                                      fun = util.est_pt_resnet_simultaneous_use_delta,
                                                                                                      penalty_const = penalty_const)
                data_len_list = [len(x0_var_names), len(x1_var_names), len(p0_var_names), len(p1_var_names), len(z_var_names)]
                restore = True
                data_adv_tgsm =  util_helpers.tgsm_adv_examples(data, data_len_list, epsilon, gradient_target_training, gradient_target_testing)
                _,_,pt_resnet_adv_tgsm_output=util.est_pt_resnet_simultaneous_use_delta(data_adv_tgsm, penalty_const, MODEL_NAME, restore, n_epoches = n_epoches, K = K)
                pt_resnet_adv_tgsm_output_dic[epsilon] = {'accuracy_training':pt_resnet_adv_tgsm_output['accuracy_training'],
                                                          'accuracy_testing':pt_resnet_adv_tgsm_output['accuracy_testing'],
                                                          'log_loss_training': pt_resnet_adv_tgsm_output['log_loss_training'],
                                                          'log_loss_testing': pt_resnet_adv_tgsm_output['log_loss_testing']
                                                          }

            #

            # 5. GN
            pt_resnet_adv_GN_output_dic = {}
            for epsilon in epsilon_list:
                print('GN ', epsilon)
                #MODEL_NAME = 'pt_est' # use it to generate the target gradients
                data_len_list = [len(x0_var_names), len(x1_var_names), len(p0_var_names), len(p1_var_names), len(z_var_names)]
                restore = True
                data_adv_GN =  util_helpers.GN_adv_examples(data, data_len_list, epsilon)
                _,_,pt_resnet_adv_GN_output=util.est_pt_resnet_simultaneous_use_delta(data_adv_GN, penalty_const, MODEL_NAME, restore, n_epoches = n_epoches, K = K)
                pt_resnet_adv_GN_output_dic[epsilon] = {'accuracy_training':pt_resnet_adv_GN_output['accuracy_training'],
                                                        'accuracy_testing':pt_resnet_adv_GN_output['accuracy_testing'],
                                                        'log_loss_training': pt_resnet_adv_GN_output['log_loss_training'],
                                                        'log_loss_testing': pt_resnet_adv_GN_output['log_loss_testing']
                                                        }




            pt_info[MODEL_NAME] = {}
            pt_info[MODEL_NAME]['data_output']=pt_resnet_output_dic
            pt_info[MODEL_NAME]['grid_output']=pt_resnet_grid_output_dic
            pt_info[MODEL_NAME]['fgsm_output']=pt_resnet_adv_fgsm_output_dic
            pt_info[MODEL_NAME]['tgsm_output']=pt_resnet_adv_tgsm_output_dic
            pt_info[MODEL_NAME]['GN_output']=pt_resnet_adv_GN_output_dic

    #
    with open('pt_info_simul_use_delta.pickle', 'wb') as f:
        pickle.dump(pt_info, f)





if __name__ == '__main__':
    RESNET = False
    Load_old_results = True
    model(RESNET, Load_old_results)




















