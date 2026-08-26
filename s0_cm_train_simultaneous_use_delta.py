#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 09:48:04 2019

s1_training_cm

@author: baichuan
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

###
x0_vars = ['walk_walktime']
x1_vars = ['bus_cost', 'bus_walktime', 'bus_waittime', 'bus_ivt']
x2_vars = ['ridesharing_cost', 'ridesharing_waittime', 'ridesharing_ivt']
x3_vars = ['drive_cost', 'drive_walktime', 'drive_ivt']
x4_vars = ['av_cost', 'av_waittime', 'av_ivt']
z_vars = ['male', 'young_age', 'old_age', 'low_edu', 'high_edu',
          'low_inc', 'high_inc', 'full_job', 'age', 'inc',
          'edu']
y_vars = ['choice']

# grid data as default for est_cm and est_resnet
data_grid = {}
n_grid = 100
data_grid['training'] = {}
data_grid['testing'] = {}
data_grid['training'] = [np.tile(np.mean(cm_data['training'][x0_vars].values, axis=0), [n_grid ** 2, 1]),
                         np.tile(np.mean(cm_data['training'][x1_vars].values, axis=0), [n_grid ** 2, 1]),
                         np.tile(np.mean(cm_data['training'][x2_vars].values, axis=0), [n_grid ** 2, 1]),
                         np.tile(np.mean(cm_data['training'][x3_vars].values, axis=0), [n_grid ** 2, 1]),
                         np.tile(np.mean(cm_data['training'][x4_vars].values, axis=0), [n_grid ** 2, 1]),
                         np.tile(np.mean(cm_data['training'][z_vars].values, axis=0), [n_grid ** 2, 1]),
                         np.zeros(n_grid ** 2)]  # the order: x0, x1, x2, x3, x4, z, y
# vary the drive_cost and drive_ivt to observe the changes
x_min = np.min(cm_data['training']['bus_cost'])
x_max = np.max(cm_data['training']['bus_cost'])
y_min = np.min(cm_data['training']['bus_ivt'])
y_max = np.max(cm_data['training']['bus_ivt'])
data_grid['training'][1][:, 0] = util_helpers.generate_mesh_data(n_grid, x_min, x_max, y_min, y_max).values[:, 0]
data_grid['training'][1][:, 3] = util_helpers.generate_mesh_data(n_grid, x_min, x_max, y_min, y_max).values[:, 1]
# copy training to testing
data_grid['testing'] = copy.copy(data_grid['training'])

############################################################
cm_info = {}
# penalty_const_list = [1e-50]
#penalty_const_list = [1e-50]
penalty_const_list = [1e-10,1e-8, 1e-7,1e-6,1e-5, 1e-4, 0.001, 0.002, 0.004, 0.005, 0.006,0.007,
                      0.008,0.009, 0.01,0.03, 0.05, 0.1, 0.3, 0.5, 0.8, 0.9,0.95,0.99,0.999,0.9999, 1]
# penalty_const_list = [1e-10, 1e-5, 1e-4, 0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9,0.95,0.99,0.999, 1]

#epsilon_list = [0.1]
epsilon_list = [0.0, 0.01, 0.03, 0.05, 0.08, 0.1, 0.2, 0.5]
# epsilon_list = []
n_epoches = 5000
K = 5  # num of classes
n_class = 5  # same as above

############################################################


############################################################
# model 3: cm_resnet_simultaneous
data = {}
data['training'] = {}
data['testing'] = {}
#
data['training'] = [cm_data['training'][x0_vars].values, cm_data['training'][x1_vars].values,
                    cm_data['training'][x2_vars].values, cm_data['training'][x3_vars].values,
                    cm_data['training'][x4_vars].values, cm_data['training'][z_vars].values,
                    cm_data['training'][y_vars].values[:, 0]]  # the order: x0, x1, x2, x3, x4, z, y
data['testing'] = [cm_data['testing'][x0_vars].values, cm_data['testing'][x1_vars].values,
                   cm_data['testing'][x2_vars].values, cm_data['testing'][x3_vars].values,
                   cm_data['testing'][x4_vars].values, cm_data['testing'][z_vars].values,
                   cm_data['testing'][y_vars].values[:, 0]]  # the order: x0, x1, x2, x3, x4, z, y
#
for penalty_const in penalty_const_list:
    ###
    MODEL_NAME = 'cm_resnet_simul' + str(penalty_const)

    # 1. estimate cm_resent
    restore = False


    cm_resnet_params_dic = []
    cm_resnet_hyper_params_dic = []
    cm_resnet_output_dic= []
    acc_test = []
    current_best_acc = 0
    for replica in range(num_replications):
        _cm_resnet_params_dic, _cm_resnet_hyper_params_dic, _cm_resnet_output_dic = util.est_cm_resnet_simultaneous_use_delta(data,
                                                                                                                 penalty_const,
                                                                                                                 MODEL_NAME,
                                                                                                                 restore,
                                                                                                                 n_epoches=n_epoches,
                                                                                                                 K=K,
                                                                                                                 current_best_acc=current_best_acc)
        cm_resnet_params_dic.append(_cm_resnet_params_dic)
        cm_resnet_hyper_params_dic.append(_cm_resnet_hyper_params_dic)
        cm_resnet_output_dic.append(_cm_resnet_output_dic)
        acc_test.append(_cm_resnet_output_dic['accuracy_testing'][-1])
        current_best_acc = _cm_resnet_output_dic['current_best_acc']
    best_index = acc_test.index(max(acc_test))



    # 2. restore
    restore = True
    _, _, cm_resnet_grid_output_dic = util.est_cm_resnet_simultaneous_use_delta(data_grid, penalty_const, MODEL_NAME, restore,
                                                         n_epoches=n_epoches, K=K)

    ### attacks
    # 3. fgsm
    cm_resnet_adv_fgsm_output_dic = {}
    for epsilon in epsilon_list:
        print('fgsm ', epsilon)
        data_len_list = [len(x0_vars), len(x1_vars), len(x2_vars), len(x3_vars), len(x4_vars), len(z_vars)]
        gradient_training = cm_resnet_output_dic[best_index]['gradient_cost_x_training']
        gradient_testing = cm_resnet_output_dic[best_index]['gradient_cost_x_testing']
        data_adv_fgsm = util_helpers.fgsm_adv_examples(data, data_len_list, epsilon, gradient_training,
                                                       gradient_testing)
        restore = True
        _, _, cm_resnet_adv_fgsm_output = util.est_cm_resnet_simultaneous_use_delta(data_adv_fgsm, penalty_const, MODEL_NAME,
                                                             restore, n_epoches=n_epoches, K=K)
        cm_resnet_adv_fgsm_output_dic[epsilon] = {'accuracy_training': cm_resnet_adv_fgsm_output['accuracy_training'],
                                                  'accuracy_testing': cm_resnet_adv_fgsm_output['accuracy_testing'],
                                                  'log_loss_training': cm_resnet_adv_fgsm_output['log_loss_training'],
                                                  'log_loss_testing': cm_resnet_adv_fgsm_output['log_loss_testing']
                                                  }

    # 4. tgsm
    cm_resnet_adv_tgsm_output_dic = {}
    for epsilon in epsilon_list:
        print('tgsm ', epsilon)
        gradient_target_training, gradient_target_testing = util_helpers.tgsm_target_gradients(data, n_class,
                                                                                               MODEL_NAME,
                                                                                               fun=util.est_cm_resnet_simultaneous_use_delta,
                                                                                               penalty_const=penalty_const)
        data_len_list = [len(x0_vars), len(x1_vars), len(x2_vars), len(x3_vars), len(x4_vars), len(z_vars)]
        restore = True
        data_adv_tgsm = util_helpers.tgsm_adv_examples(data, data_len_list, epsilon, gradient_target_training,
                                                       gradient_target_testing)
        _, _, cm_resnet_adv_tgsm_output = util.est_cm_resnet_simultaneous_use_delta(data_adv_tgsm, penalty_const, MODEL_NAME,
                                                             restore, n_epoches=n_epoches, K=K)
        cm_resnet_adv_tgsm_output_dic[epsilon] = {'accuracy_training': cm_resnet_adv_tgsm_output['accuracy_training'],
                                                  'accuracy_testing': cm_resnet_adv_tgsm_output['accuracy_testing'],
                                                  'log_loss_training': cm_resnet_adv_tgsm_output['log_loss_training'],
                                                  'log_loss_testing': cm_resnet_adv_tgsm_output['log_loss_testing']
                                                  }



    # 5. GN
    cm_resnet_adv_GN_output_dic = {}
    for epsilon in epsilon_list:
        print('GN ', epsilon)
        #MODEL_NAME = 'cm_est' # use it to generate the target gradients
        data_len_list = [len(x0_vars), len(x1_vars), len(x2_vars), len(x3_vars), len(x4_vars), len(z_vars)]
        restore = True
        data_adv_GN =  util_helpers.GN_adv_examples(data, data_len_list, epsilon)
        _,_,cm_resnet_adv_GN_output=util.est_cm_resnet_simultaneous_use_delta(data_adv_GN, penalty_const, MODEL_NAME, restore, n_epoches = n_epoches, K = K)
        cm_resnet_adv_GN_output_dic[epsilon] = {'accuracy_training':cm_resnet_adv_GN_output['accuracy_training'],
                                                'accuracy_testing':cm_resnet_adv_GN_output['accuracy_testing'],
                                                'log_loss_training': cm_resnet_adv_GN_output['log_loss_training'],
                                                'log_loss_testing': cm_resnet_adv_GN_output['log_loss_testing']
                                                }
    # save
    cm_info[MODEL_NAME] = {}
    cm_info[MODEL_NAME]['data_output'] = cm_resnet_output_dic
    cm_info[MODEL_NAME]['grid_output'] = cm_resnet_grid_output_dic
    cm_info[MODEL_NAME]['fgsm_output'] = cm_resnet_adv_fgsm_output_dic
    cm_info[MODEL_NAME]['tgsm_output'] = cm_resnet_adv_tgsm_output_dic
    cm_info[MODEL_NAME]['GN_output'] = cm_resnet_adv_GN_output_dic

#
with open('cm_info_simul_use_delta.pickle', 'wb') as f:
    pickle.dump(cm_info, f)





