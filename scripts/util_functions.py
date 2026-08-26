#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 08:38:11 2019

util functions

@author: shenhao
"""


import tensorflow as tf
import numpy as np
import pandas as pd
#import matplotlib as mpl
#mpl.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import copy
import pickle
from sklearn.metrics import f1_score

def obtain_mini_batch_dnn(x_training,y_training,n_mini_batch):
    '''
    Return mini_batch
    '''
    N = x_training.shape[0]
    index = np.random.choice(N, size = n_mini_batch)
    x_training_batch=x_training[index, :]
    y_training_batch=y_training[index]
    return x_training_batch,y_training_batch


def obtain_mini_batch_dnn_alt_specific(X0,X1,X2,X3,X4,Z,Y,n_mini_batch):
    '''
    Return mini_batch
    assume that the row numbers of all input df are the same
    '''
    N, D = X0.shape                     
    index = np.random.choice(N, size = n_mini_batch)     
    X0_batch = X0[index, :]
    X1_batch = X1[index, :]
    X2_batch = X2[index, :]
    X3_batch = X3[index, :]
    X4_batch = X4[index, :]
    Z_batch = Z[index, :]
    Y_batch = Y[index]
    return X0_batch, X1_batch, X2_batch, X3_batch, X4_batch, Z_batch, Y_batch


def standard_hidden_layer(input_, n_hidden, BN, Dropout, Dropout_rate):
    # standard layer, repeated in the following for loop.
    hidden = tf.layers.dense(input_, n_hidden, activation = tf.nn.relu)
    if BN:
        hidden = tf.layers.batch_normalization(inputs = hidden, axis = 1)
    if Dropout:
        hidden = tf.layers.dropout(inputs = hidden, rate = Dropout_rate)
    return hidden


# class dnn_constraints:
#     def __init__(self, dnn_param_dic):
#         self.layer_name = 'test'
#         self.dnn_param_dic = dnn_param_dic
#     def dnn_kernel_equality_constraints(self, input_var):
#         return self.dnn_param_dic[self.layer_name]
#     def dnn_bias_equality_constraints(self, input_var):
#         return self.dnn_param_dic[self.layer_name]



def standard_hidden_layer_constriants(input_, n_hidden, BN, Dropout, Dropout_rate, dnn_param_dic, layer_num):
    # standard layer, repeated in the following for loop.
    if layer_num == 0:
        kernel_name = 'dense/kernel:0'
        bias_name = 'dense/bias:0'
    else:
        kernel_name = 'dense_' + str(layer_num) + '/kernel:0'
        bias_name = 'dense_' + str(layer_num) + '/bias:0'
    hidden = tf.layers.dense(input_, n_hidden, activation = tf.nn.relu, kernel_constraint=lambda x: dnn_param_dic[kernel_name],
                             bias_constraint=lambda x: dnn_param_dic[bias_name])
    if BN:
        hidden = tf.layers.batch_normalization(inputs = hidden, axis = 1)
    if Dropout:
        hidden = tf.layers.dropout(inputs = hidden, rate = Dropout_rate)
    return hidden


def est_dnn(data, penalty_const,  MODEL_NAME, restore = False, n_epoches = 10000, n_mini_batch = 100, M = 3, n_hidden = 100,
            simul_data = None, Dropout = False, Dropout_rate = 0.01, BN = False, K = 2, current_best_acc = 0):
    # 
    x_training, y_training=data['training']
    x_testing, y_testing=data['testing']
    x_dim = x_training.shape[1]
    #     
    tf.reset_default_graph()

    x = tf.placeholder(dtype = tf.float32, shape = (None, x_dim), name = 'x')
    y = tf.placeholder(dtype = tf.int64, shape = (None), name = 'y')
    
    with tf.name_scope("dnn"):
        hidden = x
        for i in range(M):
            hidden = standard_hidden_layer(hidden, n_hidden, BN, Dropout, Dropout_rate)
        u = tf.layers.dense(hidden, K, name = 'output')
        prob = tf.nn.softmax(u)

    # gradient info: only alternative 1 respect to inputs
    gradient_u1_x = tf.gradients(tf.reshape(u[:,1],[-1,1]), x)
    gradient_prob1_x = tf.gradients(tf.reshape(prob[:,1],[-1,1]), x)
    
    # regularization l2
    l2_regularizer = tf.contrib.layers.l2_regularizer(scale=penalty_const, scope=None)
    vars_ = tf.trainable_variables()
    regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, vars_)
        
    with tf.name_scope("cost"):
        log_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = u, labels = y), name = 'log_loss')
        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = u, labels = y), name = 'cost')
        cost += regularization_penalty

    # gradients of costs wrt inputs
    gradient_cost_x = tf.gradients(cost, x)
         
    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(u, y, 1)                  
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    
    optimizer = tf.train.AdamOptimizer() # opt objective
    training_op = optimizer.minimize(cost) # minimize the opt objective
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    #
    cost_training=[]
    cost_testing=[]
    accuracy_training=[]
    accuracy_testing=[]
    log_loss_training=[]
    log_loss_testing=[]

    # 2. model execution
    with tf.Session() as sess:
        # always run this to train the model
        init.run()
        feed_training = {x: x_training, y: y_training}
        feed_testing = {x: x_testing, y: y_testing}
        
        if restore == True:
            # case 1. restore models and evaluate key values
            saver.restore(sess, "tmp/"+MODEL_NAME+".ckpt")
            current_cost_training = cost.eval(feed_dict=feed_training)
            current_cost_testing = cost.eval(feed_dict=feed_testing)
            current_accuracy_training = accuracy.eval(feed_dict=feed_training)
            current_accuracy_testing = accuracy.eval(feed_dict=feed_testing)
            current_log_loss_training=log_loss.eval(feed_dict=feed_training)
            current_log_loss_testing=log_loss.eval(feed_dict=feed_testing)

            cost_training.append(current_cost_training)
            cost_testing.append(current_cost_testing)
            accuracy_training.append(current_accuracy_training)
            accuracy_testing.append(current_accuracy_testing)
            log_loss_training.append(current_log_loss_training)
            log_loss_testing.append(current_log_loss_testing)

            u_training = u.eval(feed_dict = feed_training)
            u_testing = u.eval(feed_dict = feed_testing)
            prob_training = prob.eval(feed_dict = feed_training)
            prob_testing = prob.eval(feed_dict = feed_testing)
            # 
            gradient_u1_x_training=gradient_u1_x[0].eval(feed_dict = feed_training)
            gradient_u1_x_testing=gradient_u1_x[0].eval(feed_dict = feed_testing)
            gradient_prob1_x_training=gradient_prob1_x[0].eval(feed_dict = feed_training)
            gradient_prob1_x_testing=gradient_prob1_x[0].eval(feed_dict = feed_testing)
            gradient_cost_x_training=gradient_cost_x[0].eval(feed_dict = feed_training)          
            gradient_cost_x_testing=gradient_cost_x[0].eval(feed_dict = feed_testing)

            # save
            params_dic = {}
            hyper_params_dic = {}
            output_dic = {}
            #
            vars = tf.trainable_variables()
            vars_vals = sess.run(vars)
            for var, val in zip(vars, vars_vals):
                params_dic[var.name] = val
            #
            output_dic['u_training'] = u_training
            output_dic['u_testing'] = u_testing
            output_dic['prob_training'] = prob_training
            output_dic['prob_testing'] = prob_testing
            
            output_dic['gradient_u1_x_training'] = gradient_u1_x_training
            output_dic['gradient_u1_x_testing'] = gradient_u1_x_testing
            output_dic['gradient_prob1_x_training'] = gradient_prob1_x_training
            output_dic['gradient_prob1_x_testing'] = gradient_prob1_x_testing   
            output_dic['gradient_cost_x_training'] = gradient_cost_x_training
            output_dic['gradient_cost_x_testing'] = gradient_cost_x_testing
            
            output_dic['cost_training'] = cost_training
            output_dic['cost_testing'] = cost_testing
            output_dic['accuracy_training'] = accuracy_training
            output_dic['accuracy_testing'] = accuracy_testing
            output_dic['log_loss_training'] = log_loss_training
            output_dic['log_loss_testing'] = log_loss_testing

        elif restore == False:
            for i in range(n_epoches):
                # mini batch
                x_training_batch,y_training_batch=obtain_mini_batch_dnn(x_training,y_training,n_mini_batch)
                # train
                sess.run(training_op, feed_dict = {x: x_training_batch, y:y_training_batch})
    
                if i % 1000 == 0:
                    print("Iterations", i, "Cost = ", cost.eval(feed_dict = feed_training))
    
                if i % 20 == 0:
                    current_cost_training = cost.eval(feed_dict=feed_training)
                    current_cost_testing = cost.eval(feed_dict=feed_testing)
                    current_accuracy_training = accuracy.eval(feed_dict=feed_training)
                    current_accuracy_testing = accuracy.eval(feed_dict=feed_testing)
                    current_log_loss_training=log_loss.eval(feed_dict=feed_training)
                    current_log_loss_testing=log_loss.eval(feed_dict=feed_testing)

                    cost_training.append(current_cost_training)
                    cost_testing.append(current_cost_testing)
                    accuracy_training.append(current_accuracy_training)
                    accuracy_testing.append(current_accuracy_testing)
                    log_loss_training.append(current_log_loss_training)
                    log_loss_testing.append(current_log_loss_testing)

            # save the model
            if accuracy_testing[-1] >= current_best_acc:
                saver.save(sess, "tmp/"+MODEL_NAME+".ckpt")
                current_best_acc = accuracy_testing[-1]
            #
            u_training = u.eval(feed_dict = feed_training)
            u_testing = u.eval(feed_dict = feed_testing)
            prob_training = prob.eval(feed_dict = feed_training)
            prob_testing = prob.eval(feed_dict = feed_testing)
            # 
            gradient_u1_x_training=gradient_u1_x[0].eval(feed_dict = feed_training)
            gradient_u1_x_testing=gradient_u1_x[0].eval(feed_dict = feed_testing)
            gradient_prob1_x_training=gradient_prob1_x[0].eval(feed_dict = feed_training)
            gradient_prob1_x_testing=gradient_prob1_x[0].eval(feed_dict = feed_testing)
            gradient_cost_x_training=gradient_cost_x[0].eval(feed_dict = feed_training)            
            gradient_cost_x_testing=gradient_cost_x[0].eval(feed_dict = feed_testing)            
            # outputs
            params_dic = {}
            hyper_params_dic = {}
            output_dic = {}
            #
            vars = tf.trainable_variables()
            vars_vals = sess.run(vars)
            for var, val in zip(vars, vars_vals):
                params_dic[var.name] = val
            # 
            output_dic['cost_training'] = cost_training
            output_dic['cost_testing'] = cost_testing
            output_dic['accuracy_training'] = accuracy_training
            output_dic['accuracy_testing'] = accuracy_testing
            output_dic['log_loss_training'] = log_loss_training
            output_dic['log_loss_testing'] = log_loss_testing

            output_dic['u_training'] = u_training
            output_dic['u_testing'] = u_testing
            output_dic['prob_training'] = prob_training
            output_dic['prob_testing'] = prob_testing
            
            output_dic['gradient_u1_x_training'] = gradient_u1_x_training
            output_dic['gradient_u1_x_testing'] = gradient_u1_x_testing
            output_dic['gradient_prob1_x_training'] = gradient_prob1_x_training
            output_dic['gradient_prob1_x_testing'] = gradient_prob1_x_testing
            output_dic['gradient_cost_x_training'] = gradient_cost_x_training
            output_dic['gradient_cost_x_testing'] = gradient_cost_x_testing

            output_dic['current_best_acc'] = current_best_acc
            # 
            hyper_params_dic['penalty_const']=penalty_const

#        if simul_data != None:
#            output_dic['prob_simul'] = prob_simul
#            output_dic['u_simul']=u_simul
        # hyper param
#        hyper_params_dic['penalty_const']=penalty_const                
    return params_dic,hyper_params_dic,output_dic


def est_dnn_elasticity(data, MODEL_NAME, all_elas_var, all_elas_var_id, df_sp_test_nonstand,penalty_const, n_epoches = 10000, n_mini_batch = 100, M = 3, n_hidden = 100,
            simul_data = None, Dropout = False, Dropout_rate = 0.01, BN = False, K = 2):
    #
    x_training, y_training = data['training']
    x_testing, y_testing = data['testing']
    x_dim = x_training.shape[1]
    #
    tf.reset_default_graph()

    x = tf.placeholder(dtype=tf.float32, shape=(None, x_dim), name='x')
    y = tf.placeholder(dtype=tf.int64, shape=(None), name='y')

    with tf.name_scope("dnn"):
        hidden = x
        for i in range(M):
            hidden = standard_hidden_layer(hidden, n_hidden, BN, Dropout, Dropout_rate)
        u = tf.layers.dense(hidden, K, name='output')
        prob = tf.nn.softmax(u)

    # gradient info: only alternative 1 respect to inputs
    gradient_u1_x = tf.gradients(tf.reshape(u[:, 1], [-1, 1]), x)
    gradient_prob1_x = tf.gradients(tf.reshape(prob[:, 1], [-1, 1]), x)

    # regularization l2
    l2_regularizer = tf.contrib.layers.l2_regularizer(scale=penalty_const, scope=None)
    vars_ = tf.trainable_variables()
    regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, vars_)

    with tf.name_scope("cost"):
        log_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=u, labels=y), name='log_loss')
        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=u, labels=y), name='cost')
        cost += regularization_penalty

    # gradients of costs wrt inputs
    gradient_cost_x = tf.gradients(cost, x)

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(u, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    optimizer = tf.train.AdamOptimizer()  # opt objective
    training_op = optimizer.minimize(cost)  # minimize the opt objective
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    #
    cost_training = []
    cost_testing = []
    accuracy_training = []
    accuracy_testing = []
    log_loss_training = []
    log_loss_testing = []

    # 2. model execution
    with tf.Session() as sess:
        # always run this to train the model
        init.run()
        feed_testing_before = {x: x_testing, y: y_testing}

        saver.restore(sess, "tmp/" + MODEL_NAME + ".ckpt")
        prob_before = prob.eval(feed_dict=feed_testing_before)
        delta_increase = 0.001

        num_modes = K
        elast_records = {}
        for key in all_elas_var:
            for i in range(len(all_elas_var[key])):
                var_name = all_elas_var[key][i]
                data_increase = np.copy(x_testing)
                idx = all_elas_var_id[key][i]
                data_increase[:,idx] += delta_increase
                feed_testing_after = {x: data_increase, y: y_testing}
                prob_after = prob.eval(feed_dict=feed_testing_after)


                for mode in range(num_modes):
                    elasticity_individual = (prob_after[:,mode] - prob_before[:,mode]) / prob_before[:,mode] / delta_increase * df_sp_test_nonstand.loc[:, var_name] / df_sp_test_nonstand.loc[:, var_name].std()
                    elasticity = np.mean(elasticity_individual)
                    elast_records[str(mode) + '___' + var_name] = [elasticity]




    return elast_records

def est_cm_elasticity(data, MODEL_NAME, all_elas_var, df_sp_test_nonstand):
    #
    print(len(data['training']))
    x0_training, x1_training, x2_training, x3_training, x4_training, z_training, y_training = data['training']
    x0_testing, x1_testing, x2_testing, x3_testing, x4_testing, z_testing, y_testing = data['testing']

    N, D0 = x0_training.shape
    N, D1 = x1_training.shape
    N, D2 = x2_training.shape
    N, D3 = x3_training.shape
    N, D4 = x4_training.shape
    N, DZ = z_training.shape

    D_dic = {'d0': D0, 'd1': D1, 'd2': D2, 'd3': D3, 'd4': D4, 'dz': DZ}

    # model
    tf.reset_default_graph()

    x0 = tf.placeholder(dtype=tf.float32, shape=(None, D0), name='x0')
    x1 = tf.placeholder(dtype=tf.float32, shape=(None, D1), name='x1')
    x2 = tf.placeholder(dtype=tf.float32, shape=(None, D2), name='x2')
    x3 = tf.placeholder(dtype=tf.float32, shape=(None, D3), name='x3')
    x4 = tf.placeholder(dtype=tf.float32, shape=(None, D4), name='x4')
    z = tf.placeholder(dtype=tf.float32, shape=(None, DZ), name='z')
    y = tf.placeholder(dtype=tf.int64, shape=(None), name='y')

    hidden_x0 = x0
    hidden_x1 = x1
    hidden_x2 = x2
    hidden_x3 = x3
    hidden_x4 = x4
    hidden_z = z

    hidden_dic = {}
    hidden_dic['x0'] = hidden_x0
    hidden_dic['x1'] = hidden_x1
    hidden_dic['x2'] = hidden_x2
    hidden_dic['x3'] = hidden_x3
    hidden_dic['x4'] = hidden_x4
    hidden_dic['z'] = hidden_z

    # initialize parameters
    params_dic = {}
    for j in range(5):
        w_name = 'w' + str(j)
        b_name = 'b' + str(j)
        wz_name = 'wz' + str(j)
        d_name = 'd' + str(j)
        params_dic[w_name] = tf.Variable(tf.random_normal([D_dic[d_name], 1]), dtype=tf.float32, name=w_name)
        params_dic[b_name] = tf.Variable(tf.random_normal([1]), dtype=tf.float32, name=b_name)
        params_dic[wz_name] = tf.Variable(tf.random_normal([D_dic['dz'], 1]), dtype=tf.float32, name=wz_name)
    #
    output_dic = {}
    for j in range(5):
        layer_name = 'x' + str(j)
        w_name = 'w' + str(j)
        b_name = 'b' + str(j)
        wz_name = 'wz' + str(j)
        hidden_j = tf.add(tf.add(tf.matmul(hidden_dic[layer_name], params_dic[w_name]),
                                 tf.matmul(hidden_dic['z'], params_dic[wz_name])), params_dic[b_name])
        output_name = 'u' + str(j)
        output_dic[output_name] = hidden_j

    u = tf.concat([output_dic['u0'], output_dic['u1'], output_dic['u2'], output_dic['u3'], output_dic['u4']], axis=1,
                  name='u')
    prob = tf.nn.softmax(u)

    # gradient info: only alternative 1 respect to inputs
    gradient_u1_x = [tf.concat(
        [tf.gradients(u[:, 1], [x0])[0], tf.gradients(u[:, 1], [x1])[0], tf.gradients(u[:, 1], [x2])[0],
         tf.gradients(u[:, 1], [x3])[0], tf.gradients(u[:, 1], [x4])[0], tf.gradients(u[:, 1], [z])[0]], axis=1)]

    gradient_prob1_x = [tf.concat(
        [tf.gradients(prob[:, 1], [x0])[0], tf.gradients(prob[:, 1], [x1])[0], tf.gradients(prob[:, 1], [x2])[0],
         tf.gradients(prob[:, 1], [x3])[0], tf.gradients(prob[:, 1], [x4])[0], tf.gradients(prob[:, 1], [z])[0]],
        axis=1)]

    # cost
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=u, labels=y), name='cost')
    # cost gradients
    gradient_cost_x = [tf.concat([tf.gradients(cost, [x0])[0], tf.gradients(cost, [x1])[0], tf.gradients(cost, [x2])[0],
                                  tf.gradients(cost, [x3])[0], tf.gradients(cost, [x4])[0], tf.gradients(cost, [z])[0]],
                                 axis=1)]

    log_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=u, labels=y), name='log_loss')
    optimizer = tf.train.AdamOptimizer()  # opt: better than simple gradient descents: always converge to stable numbers.
    #    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate) # opt objective
    training_op = optimizer.minimize(cost)  # minimize the opt objective

    # eval params
    correct = tf.nn.in_top_k(u, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    cost_training = []
    cost_testing = []
    accuracy_training = []
    accuracy_testing = []
    log_loss_training = []
    log_loss_testing = []


    #start calculation
    num_modes = 5
    elast_records = {}
    with tf.Session() as sess:
        # always run this to train the model
        init.run()
        feed_testing_before = {x0: x0_testing, x1: x1_testing, x2: x2_testing, x3: x3_testing, x4: x4_testing, z: z_testing,
                        y: y_testing}

        saver.restore(sess, "tmp/" + MODEL_NAME + ".ckpt")
        prob_before = prob.eval(feed_dict=feed_testing_before)
        delta_increase = 0.001

        for key in all_elas_var:
            for i in range(len(all_elas_var[key])):
                var_name = all_elas_var[key][i]
                if key == 'x0_vars':
                    data_increase = np.copy(x0_testing)
                    data_increase[:,i] += delta_increase
                    feed_testing_after = {x0: data_increase, x1: x1_testing, x2: x2_testing, x3: x3_testing, x4: x4_testing, z: z_testing,
                        y: y_testing}
                    prob_after = prob.eval(feed_dict=feed_testing_after)

                elif key == 'x1_vars':
                    data_increase = np.copy(x1_testing)
                    data_increase[:,i] += delta_increase
                    feed_testing_after = {x0: x0_testing, x1: data_increase, x2: x2_testing, x3: x3_testing, x4: x4_testing, z: z_testing,
                        y: y_testing}
                    prob_after = prob.eval(feed_dict=feed_testing_after)
                elif key == 'x2_vars':
                    data_increase = np.copy(x2_testing)
                    data_increase[:,i] += delta_increase
                    feed_testing_after = {x0: x0_testing, x1: x1_testing, x2: data_increase, x3: x3_testing, x4: x4_testing, z: z_testing,
                        y: y_testing}
                    prob_after = prob.eval(feed_dict=feed_testing_after)
                elif key == 'x3_vars':
                    data_increase = np.copy(x3_testing)
                    data_increase[:,i] += delta_increase
                    feed_testing_after = {x0: x0_testing, x1: x1_testing, x2: x2_testing, x3: data_increase, x4: x4_testing, z: z_testing,
                        y: y_testing}
                    prob_after = prob.eval(feed_dict=feed_testing_after)
                elif key == 'x4_vars':
                    data_increase = np.copy(x4_testing)
                    data_increase[:,i] += delta_increase
                    feed_testing_after = {x0: x0_testing, x1: x1_testing, x2: x2_testing, x3: x3_testing, x4: data_increase, z: z_testing,
                        y: y_testing}
                    prob_after = prob.eval(feed_dict=feed_testing_after)

                for mode in range(num_modes):
                    elasticity_individual = (prob_after[:,mode] - prob_before[:,mode]) / prob_before[:,mode] / delta_increase * df_sp_test_nonstand.loc[:, var_name] / df_sp_test_nonstand.loc[:, var_name].std()
                    elasticity = np.mean(elasticity_individual)
                    elast_records[str(mode) + '___' + var_name] = [elasticity]




    return elast_records


def est_cm(data, MODEL_NAME, restore = False, n_epoches = 10000, current_best_acc = 0, K=5):
    #
    x0_training,x1_training,x2_training,x3_training,x4_training,z_training,y_training = data['training']
    x0_testing,x1_testing,x2_testing,x3_testing,x4_testing,z_testing,y_testing = data['testing']
        
    N, D0 = x0_training.shape
    N, D1 = x1_training.shape
    N, D2 = x2_training.shape
    N, D3 = x3_training.shape
    N, D4 = x4_training.shape
    N, DZ = z_training.shape
    
    D_dic = {'d0':D0, 'd1':D1, 'd2':D2, 'd3':D3, 'd4':D4, 'dz':DZ}
    
    # model 
    tf.reset_default_graph()
    
    x0 = tf.placeholder(dtype = tf.float32, shape = (None, D0), name = 'x0')
    x1 = tf.placeholder(dtype = tf.float32, shape = (None, D1), name = 'x1')
    x2 = tf.placeholder(dtype = tf.float32, shape = (None, D2), name = 'x2')
    x3 = tf.placeholder(dtype = tf.float32, shape = (None, D3), name = 'x3')
    x4 = tf.placeholder(dtype = tf.float32, shape = (None, D4), name = 'x4')
    z = tf.placeholder(dtype = tf.float32, shape = (None, DZ), name = 'z')
    y = tf.placeholder(dtype = tf.int64, shape = (None), name = 'y')
    
    hidden_x0 = x0
    hidden_x1 = x1
    hidden_x2 = x2
    hidden_x3 = x3
    hidden_x4 = x4
    hidden_z = z
    
    hidden_dic = {}
    hidden_dic['x0'] = hidden_x0
    hidden_dic['x1'] = hidden_x1
    hidden_dic['x2'] = hidden_x2
    hidden_dic['x3'] = hidden_x3
    hidden_dic['x4'] = hidden_x4
    hidden_dic['z'] = hidden_z        

    # initialize parameters
    params_dic = {}
    for j in range(K):
        w_name = 'w'+str(j)
        b_name = 'b'+str(j)
        wz_name = 'wz'+str(j)
        d_name = 'd'+str(j)
        params_dic[w_name] = tf.Variable(tf.random_normal([D_dic[d_name], 1]), dtype = tf.float32, name = w_name)
        params_dic[b_name] = tf.Variable(tf.random_normal([1]), dtype = tf.float32, name = b_name)        
        params_dic[wz_name] = tf.Variable(tf.random_normal([D_dic['dz'], 1]), dtype = tf.float32, name = wz_name)
    #
    output_dic = {}
    for j in range(K):
        layer_name = 'x'+str(j)
        w_name = 'w'+str(j)
        b_name = 'b'+str(j)
        wz_name = 'wz'+str(j)        
        hidden_j = tf.add(tf.add(tf.matmul(hidden_dic[layer_name], params_dic[w_name]), tf.matmul(hidden_dic['z'], params_dic[wz_name])), params_dic[b_name])
        output_name = 'u'+str(j)
        output_dic[output_name] = hidden_j
    
    u = tf.concat([output_dic['u0'], output_dic['u1'], output_dic['u2'], output_dic['u3'], output_dic['u4']], axis = 1, name = 'u')
    prob = tf.nn.softmax(u)

    # gradient info: only alternative 1 respect to inputs
    gradient_u1_x = [tf.concat([tf.gradients(u[:,1], [x0])[0], tf.gradients(u[:,1], [x1])[0], tf.gradients(u[:,1], [x2])[0], tf.gradients(u[:,1], [x3])[0], tf.gradients(u[:,1], [x4])[0], tf.gradients(u[:,1], [z])[0]], axis = 1)]
    
    gradient_prob1_x = [tf.concat([tf.gradients(prob[:,1], [x0])[0], tf.gradients(prob[:,1], [x1])[0], tf.gradients(prob[:,1], [x2])[0], tf.gradients(prob[:,1], [x3])[0], tf.gradients(prob[:,1], [x4])[0], tf.gradients(prob[:,1], [z])[0]], axis = 1)]

    # cost
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = u, labels = y), name = 'cost')
    # cost gradients
    gradient_cost_x = [tf.concat([tf.gradients(cost, [x0])[0], tf.gradients(cost, [x1])[0], tf.gradients(cost, [x2])[0], tf.gradients(cost, [x3])[0], tf.gradients(cost, [x4])[0], tf.gradients(cost, [z])[0]], axis = 1)]
    
    log_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = u, labels = y), name = 'log_loss')
    optimizer = tf.train.AdamOptimizer() # opt: better than simple gradient descents: always converge to stable numbers.
    #    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate) # opt objective
    training_op = optimizer.minimize(cost) # minimize the opt objective
    
    # eval params
    pred_y = tf.argmax(prob,1)
    correct = tf.nn.in_top_k(u, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    cost_training=[]
    cost_testing=[]
    accuracy_training=[]
    accuracy_testing=[]
    # f1_score_training = []
    # f1_score_testing = []
    log_loss_training=[]
    log_loss_testing=[]
        
    with tf.Session() as sess:
        # always run this to train the model
        init.run()
        feed_training = {x0: x0_training, x1: x1_training, x2: x2_training, x3: x3_training, x4: x4_training, z: z_training, y: y_training}
        feed_testing = {x0: x0_testing, x1: x1_testing, x2: x2_testing, x3: x3_testing, x4: x4_testing, z: z_testing, y: y_testing}
        
        if restore == True:
            saver.restore(sess, "tmp/"+MODEL_NAME+".ckpt")
            u_training = u.eval(feed_dict = feed_training)
            u_testing = u.eval(feed_dict = feed_testing)
            prob_training = prob.eval(feed_dict = feed_training)
            prob_testing = prob.eval(feed_dict = feed_testing)
            #


            current_cost_training = cost.eval(feed_dict=feed_training)
            current_cost_testing = cost.eval(feed_dict=feed_testing)
            current_accuracy_training = accuracy.eval(feed_dict=feed_training)
            current_accuracy_testing = accuracy.eval(feed_dict=feed_testing)
            current_log_loss_training = log_loss.eval(feed_dict = feed_training)
            current_log_loss_testing = log_loss.eval(feed_dict = feed_testing)
            # current_pred_y_train = pred_y.eval(feed_dict=feed_training)
            # current_pred_y_test = pred_y.eval(feed_dict=feed_testing)
            # current_f1_train = f1_score(y_training, current_pred_y_train, average='micro')
            # current_f1_test = f1_score(y_testing, current_pred_y_test, average='micro')

            #
            log_loss_training.append(current_log_loss_training)
            log_loss_testing.append(current_log_loss_testing)
            cost_training.append(current_cost_training)
            cost_testing.append(current_cost_testing)
            accuracy_training.append(current_accuracy_training)
            accuracy_testing.append(current_accuracy_testing)
            # f1_score_training.append(current_f1_train)
            # f1_score_testing.append(current_f1_test)
            #
            gradient_u1_x_training=gradient_u1_x[0].eval(feed_dict = feed_training)
            gradient_u1_x_testing=gradient_u1_x[0].eval(feed_dict = feed_testing)
            gradient_prob1_x_training=gradient_prob1_x[0].eval(feed_dict = feed_training)
            gradient_prob1_x_testing=gradient_prob1_x[0].eval(feed_dict = feed_testing)
            gradient_cost_x_training=gradient_cost_x[0].eval(feed_dict = feed_training)          
            gradient_cost_x_testing=gradient_cost_x[0].eval(feed_dict = feed_testing)
            # save
            params_values = {}
            hyper_params_values = {}
            output_values = {}
            
            output_values['u_training'] = u_training
            output_values['u_testing'] = u_testing
            output_values['prob_training'] = prob_training
            output_values['prob_testing'] = prob_testing

            output_values['cost_training'] = cost_training
            output_values['cost_testing'] = cost_testing
            output_values['accuracy_training'] = accuracy_training
            output_values['accuracy_testing'] = accuracy_testing
            output_values['log_loss_training'] = log_loss_training
            output_values['log_loss_testing'] = log_loss_testing
            # output_values['f1_score_training'] = f1_score_training
            # output_values['f1_score_testing'] = f1_score_testing

            output_values['gradient_u1_x_training'] = gradient_u1_x_training
            output_values['gradient_u1_x_testing'] = gradient_u1_x_testing
            output_values['gradient_prob1_x_training'] = gradient_prob1_x_training
            output_values['gradient_prob1_x_testing'] = gradient_prob1_x_testing
            output_values['gradient_cost_x_training'] = gradient_cost_x_training
            output_values['gradient_cost_x_testing'] = gradient_cost_x_testing
                        
        if restore == False:
            for i in range(n_epoches):
                X0_batch, X1_batch, X2_batch, X3_batch, X4_batch, Z_batch, Y_batch = obtain_mini_batch_dnn_alt_specific(x0_training, x1_training, x2_training, 
                    x3_training, x4_training, z_training, y_training, n_mini_batch = 100)
                sess.run(training_op, feed_dict = {x0:X0_batch, x1:X1_batch, x2:X2_batch, x3:X3_batch, x4:X4_batch, z:Z_batch, y:Y_batch})
                if i % 1000 == 0:
                    print("Iterations", i, "Cost = ", cost.eval(feed_dict = feed_training))
                if i % 20 == 0:
                    current_cost_training = cost.eval(feed_dict=feed_training)
                    current_cost_testing = cost.eval(feed_dict=feed_testing)

                    # current_pred_y_train = pred_y.eval(feed_dict=feed_training)
                    # current_pred_y_test = pred_y.eval(feed_dict=feed_testing)
                    # current_f1_train = f1_score(y_training, current_pred_y_train, average='micro')
                    # current_f1_test = f1_score(y_testing, current_pred_y_test, average='micro')
                    current_accuracy_training = accuracy.eval(feed_dict=feed_training)
                    current_accuracy_testing = accuracy.eval(feed_dict=feed_testing)
                    current_log_loss_training = log_loss.eval(feed_dict = feed_training)
                    current_log_loss_testing = log_loss.eval(feed_dict = feed_testing)


                    log_loss_training.append(current_log_loss_training)
                    log_loss_testing.append(current_log_loss_testing)
                    cost_training.append(current_cost_training)
                    cost_testing.append(current_cost_testing)
                    accuracy_training.append(current_accuracy_training)
                    accuracy_testing.append(current_accuracy_testing)
                    # f1_score_training.append(current_f1_train)
                    # f1_score_testing.append(current_f1_test)


            # save the model
            if accuracy_testing[-1] >= current_best_acc:
                saver.save(sess, "tmp/"+MODEL_NAME+".ckpt")
                current_best_acc = accuracy_testing[-1]


            # 
            u_training = u.eval(feed_dict = feed_training)
            u_testing = u.eval(feed_dict = feed_testing)
            prob_training = prob.eval(feed_dict = feed_training)
            prob_testing = prob.eval(feed_dict = feed_testing)
            #
            gradient_u1_x_training=gradient_u1_x[0].eval(feed_dict = feed_training)
            gradient_u1_x_testing=gradient_u1_x[0].eval(feed_dict = feed_testing)
            gradient_prob1_x_training=gradient_prob1_x[0].eval(feed_dict = feed_training)
            gradient_prob1_x_testing=gradient_prob1_x[0].eval(feed_dict = feed_testing)
            gradient_cost_x_training=gradient_cost_x[0].eval(feed_dict = feed_training)            
            gradient_cost_x_testing=gradient_cost_x[0].eval(feed_dict = feed_testing)            

            params_values = {}
            hyper_params_values = {}
            output_values = {}
                        
            for j in range(5):
                w_name = 'w'+str(j)
                b_name = 'b'+str(j)
                wz_name = 'wz'+str(j)
                params_values[w_name] = params_dic[w_name].eval()
                params_values[b_name] = params_dic[b_name].eval()
                params_values[wz_name] = params_dic[wz_name].eval()

            ''' evlauate the model by testing data'''
            print("Final Training Accuracy: ", accuracy.eval(feed_dict = feed_training))
            
            output_values['cost_training'] = cost_training
            output_values['cost_testing'] = cost_testing
            output_values['accuracy_training'] = accuracy_training
            output_values['accuracy_testing'] = accuracy_testing
            output_values['log_loss_training'] = log_loss_training
            output_values['log_loss_testing'] = log_loss_testing
            # output_values['f1_score_training'] = f1_score_training
            # output_values['f1_score_testing'] = f1_score_testing


            output_values['u_training'] = u_training
            output_values['u_testing'] = u_testing
            output_values['prob_training'] = prob_training
            output_values['prob_testing'] = prob_testing
            
            output_values['gradient_u1_x_training'] = gradient_u1_x_training
            output_values['gradient_u1_x_testing'] = gradient_u1_x_testing
            output_values['gradient_prob1_x_training'] = gradient_prob1_x_training
            output_values['gradient_prob1_x_testing'] = gradient_prob1_x_testing
            output_values['gradient_cost_x_training'] = gradient_cost_x_training
            output_values['gradient_cost_x_testing'] = gradient_cost_x_testing

            output_values['current_best_acc'] = current_best_acc

    return params_values,hyper_params_values,output_values


def est_cm_resnet_elasticity(data, cm_param_dic, penalty_const, MODEL_NAME, all_elas_var, df_sp_test_nonstand,
                             n_epoches = 10000, n_mini_batch=100, M=3,n_hidden=100,
                             simul_data = None, Dropout = False, Dropout_rate = 0.01, BN = False, x_dim = 15, K = 2, D = 7):
    #
    x0_training, x1_training, x2_training, x3_training, x4_training, z_training, y_training = data['training']
    x0_testing, x1_testing, x2_testing, x3_testing, x4_testing, z_testing, y_testing = data['testing']
    #
    N, D0 = x0_training.shape
    N, D1 = x1_training.shape
    N, D2 = x2_training.shape
    N, D3 = x3_training.shape
    N, D4 = x4_training.shape
    N, DZ = z_training.shape

    D_dic = {'d0': D0, 'd1': D1, 'd2': D2, 'd3': D3, 'd4': D4, 'dz': DZ}

    # model
    tf.reset_default_graph()

    x0 = tf.placeholder(dtype=tf.float32, shape=(None, D0), name='x0')
    x1 = tf.placeholder(dtype=tf.float32, shape=(None, D1), name='x1')
    x2 = tf.placeholder(dtype=tf.float32, shape=(None, D2), name='x2')
    x3 = tf.placeholder(dtype=tf.float32, shape=(None, D3), name='x3')
    x4 = tf.placeholder(dtype=tf.float32, shape=(None, D4), name='x4')
    z = tf.placeholder(dtype=tf.float32, shape=(None, DZ), name='z')
    y = tf.placeholder(dtype=tf.int64, shape=(None), name='y')
    x = tf.concat([x0, x1, x2, x3, x4, z], axis=1, name='x')

    hidden_x0 = x0
    hidden_x1 = x1
    hidden_x2 = x2
    hidden_x3 = x3
    hidden_x4 = x4
    hidden_z = z

    hidden_dic = {}
    hidden_dic['x0'] = hidden_x0
    hidden_dic['x1'] = hidden_x1
    hidden_dic['x2'] = hidden_x2
    hidden_dic['x3'] = hidden_x3
    hidden_dic['x4'] = hidden_x4
    hidden_dic['z'] = hidden_z

    # initialize parameters
    params_dic = {}
    for j in range(5):
        w_name = 'w' + str(j)
        b_name = 'b' + str(j)
        wz_name = 'wz' + str(j)
        d_name = 'd' + str(j)
        params_dic[w_name] = cm_param_dic[w_name]
        params_dic[b_name] = cm_param_dic[b_name]
        params_dic[wz_name] = cm_param_dic[wz_name]
        #
    output_dic = {}
    for j in range(5):
        layer_name = 'x' + str(j)
        w_name = 'w' + str(j)
        b_name = 'b' + str(j)
        wz_name = 'wz' + str(j)
        hidden_j = tf.add(tf.add(tf.matmul(hidden_dic[layer_name], params_dic[w_name]),
                                 tf.matmul(hidden_dic['z'], params_dic[wz_name])), params_dic[b_name])
        output_name = 'u' + str(j)
        output_dic[output_name] = hidden_j
    u_cm = tf.concat([output_dic['u0'], output_dic['u1'], output_dic['u2'], output_dic['u3'], output_dic['u4']], axis=1,
                     name='u')

    # train dnn part
    with tf.name_scope("dnn"):
        hidden = x
        for i in range(M):
            hidden = standard_hidden_layer(hidden, n_hidden, BN, Dropout, Dropout_rate)
        u_dnn = tf.layers.dense(hidden, K, name='output')

    # u and prob
    u = u_cm + u_dnn
    prob = tf.nn.softmax(u)

    # gradient info: only alternative 1 (bus) respect to inputs
    gradient_u1_x = [tf.concat(
        [tf.gradients(u[:, 1], [x0])[0], tf.gradients(u[:, 1], [x1])[0], tf.gradients(u[:, 1], [x2])[0],
         tf.gradients(u[:, 1], [x3])[0],
         tf.gradients(u[:, 1], [x4])[0], tf.gradients(u[:, 1], [z])[0]], axis=1)]

    gradient_prob1_x = [tf.concat(
        [tf.gradients(prob[:, 1], [x0])[0], tf.gradients(prob[:, 1], [x1])[0], tf.gradients(prob[:, 1], [x2])[0],
         tf.gradients(prob[:, 1], [x3])[0], tf.gradients(prob[:, 1], [x4])[0], tf.gradients(prob[:, 1], [z])[0]],
        axis=1)]

    # regularization l2
    l2_regularizer = tf.contrib.layers.l2_regularizer(scale=penalty_const, scope=None)
    vars_ = tf.trainable_variables()
    regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, vars_)

    # cost
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=u, labels=y), name='cost')
    cost += regularization_penalty
    log_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=u, labels=y), name='log_loss')
    optimizer = tf.train.AdamOptimizer()  # opt: better than simple gradient descents: always converge to stable numbers.
    #    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate) # opt objective
    training_op = optimizer.minimize(cost)  # minimize the opt objective

    #
    gradient_cost_x = [tf.concat([tf.gradients(cost, [x0])[0], tf.gradients(cost, [x1])[0], tf.gradients(cost, [x2])[0],
                                  tf.gradients(cost, [x3])[0], tf.gradients(cost, [x4])[0], tf.gradients(cost, [z])[0]],
                                 axis=1)]

    # eval params
    correct = tf.nn.in_top_k(u, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    cost_training = []
    cost_testing = []
    accuracy_training = []
    accuracy_testing = []
    log_loss_training = []
    log_loss_testing = []

    num_modes = K
    elast_records = {}
    with tf.Session() as sess:
        # always run this to train the model
        init.run()
        feed_testing_before = {x0: x0_testing, x1: x1_testing, x2: x2_testing, x3: x3_testing, x4: x4_testing, z: z_testing,
                        y: y_testing}

        saver.restore(sess, "tmp/" + MODEL_NAME + ".ckpt")
        prob_before = prob.eval(feed_dict=feed_testing_before)
        delta_increase = 0.001

        for key in all_elas_var:
            for i in range(len(all_elas_var[key])):
                var_name = all_elas_var[key][i]
                if key == 'x0_vars':
                    data_increase = np.copy(x0_testing)
                    data_increase[:,i] += delta_increase
                    feed_testing_after = {x0: data_increase, x1: x1_testing, x2: x2_testing, x3: x3_testing, x4: x4_testing, z: z_testing,
                        y: y_testing}
                    prob_after = prob.eval(feed_dict=feed_testing_after)

                elif key == 'x1_vars':
                    data_increase = np.copy(x1_testing)
                    data_increase[:,i] += delta_increase
                    feed_testing_after = {x0: x0_testing, x1: data_increase, x2: x2_testing, x3: x3_testing, x4: x4_testing, z: z_testing,
                        y: y_testing}
                    prob_after = prob.eval(feed_dict=feed_testing_after)
                elif key == 'x2_vars':
                    data_increase = np.copy(x2_testing)
                    data_increase[:,i] += delta_increase
                    feed_testing_after = {x0: x0_testing, x1: x1_testing, x2: data_increase, x3: x3_testing, x4: x4_testing, z: z_testing,
                        y: y_testing}
                    prob_after = prob.eval(feed_dict=feed_testing_after)
                elif key == 'x3_vars':
                    data_increase = np.copy(x3_testing)
                    data_increase[:,i] += delta_increase
                    feed_testing_after = {x0: x0_testing, x1: x1_testing, x2: x2_testing, x3: data_increase, x4: x4_testing, z: z_testing,
                        y: y_testing}
                    prob_after = prob.eval(feed_dict=feed_testing_after)
                elif key == 'x4_vars':
                    data_increase = np.copy(x4_testing)
                    data_increase[:,i] += delta_increase
                    feed_testing_after = {x0: x0_testing, x1: x1_testing, x2: x2_testing, x3: x3_testing, x4: data_increase, z: z_testing,
                        y: y_testing}
                    prob_after = prob.eval(feed_dict=feed_testing_after)

                for mode in range(num_modes):
                    elasticity_individual = (prob_after[:,mode] - prob_before[:,mode]) / prob_before[:,mode] / delta_increase * df_sp_test_nonstand.loc[:, var_name] / df_sp_test_nonstand.loc[:, var_name].std()
                    elasticity = np.mean(elasticity_individual)
                    elast_records[str(mode) + '___' + var_name] = [elasticity]

    return elast_records







def est_cm_resnet_elasticity_use_delta(data, cm_param_dic, penalty_const, MODEL_NAME, all_elas_var, df_sp_test_nonstand,
                             n_epoches = 10000, n_mini_batch=100, M=3,n_hidden=100, l2_regu = 1e-50,
                             simul_data = None, Dropout = False, Dropout_rate = 0.01, BN = False, x_dim = 15, K = 2, D = 7):
    #
    x0_training, x1_training, x2_training, x3_training, x4_training, z_training, y_training = data['training']
    x0_testing, x1_testing, x2_testing, x3_testing, x4_testing, z_testing, y_testing = data['testing']
    #
    N, D0 = x0_training.shape
    N, D1 = x1_training.shape
    N, D2 = x2_training.shape
    N, D3 = x3_training.shape
    N, D4 = x4_training.shape
    N, DZ = z_training.shape

    D_dic = {'d0': D0, 'd1': D1, 'd2': D2, 'd3': D3, 'd4': D4, 'dz': DZ}

    # model
    tf.reset_default_graph()

    x0 = tf.placeholder(dtype=tf.float32, shape=(None, D0), name='x0')
    x1 = tf.placeholder(dtype=tf.float32, shape=(None, D1), name='x1')
    x2 = tf.placeholder(dtype=tf.float32, shape=(None, D2), name='x2')
    x3 = tf.placeholder(dtype=tf.float32, shape=(None, D3), name='x3')
    x4 = tf.placeholder(dtype=tf.float32, shape=(None, D4), name='x4')
    z = tf.placeholder(dtype=tf.float32, shape=(None, DZ), name='z')
    y = tf.placeholder(dtype=tf.int64, shape=(None), name='y')
    x = tf.concat([x0, x1, x2, x3, x4, z], axis=1, name='x')

    hidden_x0 = x0
    hidden_x1 = x1
    hidden_x2 = x2
    hidden_x3 = x3
    hidden_x4 = x4
    hidden_z = z

    hidden_dic = {}
    hidden_dic['x0'] = hidden_x0
    hidden_dic['x1'] = hidden_x1
    hidden_dic['x2'] = hidden_x2
    hidden_dic['x3'] = hidden_x3
    hidden_dic['x4'] = hidden_x4
    hidden_dic['z'] = hidden_z

    # initialize parameters
    params_dic = {}
    for j in range(5):
        w_name = 'w' + str(j)
        b_name = 'b' + str(j)
        wz_name = 'wz' + str(j)
        d_name = 'd' + str(j)
        params_dic[w_name] = cm_param_dic[w_name]
        params_dic[b_name] = cm_param_dic[b_name]
        params_dic[wz_name] = cm_param_dic[wz_name]
        #
    output_dic = {}
    for j in range(5):
        layer_name = 'x' + str(j)
        w_name = 'w' + str(j)
        b_name = 'b' + str(j)
        wz_name = 'wz' + str(j)
        hidden_j = tf.add(tf.add(tf.matmul(hidden_dic[layer_name], params_dic[w_name]),
                                 tf.matmul(hidden_dic['z'], params_dic[wz_name])), params_dic[b_name])
        output_name = 'u' + str(j)
        output_dic[output_name] = hidden_j
    u_cm = tf.concat([output_dic['u0'], output_dic['u1'], output_dic['u2'], output_dic['u3'], output_dic['u4']], axis=1,
                     name='u')

    # train dnn part
    with tf.name_scope("dnn"):
        hidden = x
        for i in range(M):
            hidden = standard_hidden_layer(hidden, n_hidden, BN, Dropout, Dropout_rate)
        u_dnn = tf.layers.dense(hidden, K, name='output')

    # u and prob
    u = (1-penalty_const) * u_cm + penalty_const * u_dnn
    prob = tf.nn.softmax(u)

    # gradient info: only alternative 1 (bus) respect to inputs
    gradient_u1_x = [tf.concat(
        [tf.gradients(u[:, 1], [x0])[0], tf.gradients(u[:, 1], [x1])[0], tf.gradients(u[:, 1], [x2])[0],
         tf.gradients(u[:, 1], [x3])[0],
         tf.gradients(u[:, 1], [x4])[0], tf.gradients(u[:, 1], [z])[0]], axis=1)]

    gradient_prob1_x = [tf.concat(
        [tf.gradients(prob[:, 1], [x0])[0], tf.gradients(prob[:, 1], [x1])[0], tf.gradients(prob[:, 1], [x2])[0],
         tf.gradients(prob[:, 1], [x3])[0], tf.gradients(prob[:, 1], [x4])[0], tf.gradients(prob[:, 1], [z])[0]],
        axis=1)]

    # regularization l2
    l2_regularizer = tf.contrib.layers.l2_regularizer(scale=l2_regu, scope=None)
    vars_ = tf.trainable_variables()
    regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, vars_)

    # cost
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=u, labels=y), name='cost')
    cost += regularization_penalty
    log_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=u, labels=y), name='log_loss')
    optimizer = tf.train.AdamOptimizer()  # opt: better than simple gradient descents: always converge to stable numbers.
    #    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate) # opt objective
    training_op = optimizer.minimize(cost)  # minimize the opt objective

    #
    gradient_cost_x = [tf.concat([tf.gradients(cost, [x0])[0], tf.gradients(cost, [x1])[0], tf.gradients(cost, [x2])[0],
                                  tf.gradients(cost, [x3])[0], tf.gradients(cost, [x4])[0], tf.gradients(cost, [z])[0]],
                                 axis=1)]

    # eval params
    correct = tf.nn.in_top_k(u, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    cost_training = []
    cost_testing = []
    accuracy_training = []
    accuracy_testing = []
    log_loss_training = []
    log_loss_testing = []

    num_modes = K
    elast_records = {}
    with tf.Session() as sess:
        # always run this to train the model
        init.run()
        feed_testing_before = {x0: x0_testing, x1: x1_testing, x2: x2_testing, x3: x3_testing, x4: x4_testing, z: z_testing,
                        y: y_testing}

        saver.restore(sess, "tmp/" + MODEL_NAME + ".ckpt")
        prob_before = prob.eval(feed_dict=feed_testing_before)
        delta_increase = 0.001

        for key in all_elas_var:
            for i in range(len(all_elas_var[key])):
                var_name = all_elas_var[key][i]
                if key == 'x0_vars':
                    data_increase = np.copy(x0_testing)
                    data_increase[:,i] += delta_increase
                    feed_testing_after = {x0: data_increase, x1: x1_testing, x2: x2_testing, x3: x3_testing, x4: x4_testing, z: z_testing,
                        y: y_testing}
                    prob_after = prob.eval(feed_dict=feed_testing_after)

                elif key == 'x1_vars':
                    data_increase = np.copy(x1_testing)
                    data_increase[:,i] += delta_increase
                    feed_testing_after = {x0: x0_testing, x1: data_increase, x2: x2_testing, x3: x3_testing, x4: x4_testing, z: z_testing,
                        y: y_testing}
                    prob_after = prob.eval(feed_dict=feed_testing_after)
                elif key == 'x2_vars':
                    data_increase = np.copy(x2_testing)
                    data_increase[:,i] += delta_increase
                    feed_testing_after = {x0: x0_testing, x1: x1_testing, x2: data_increase, x3: x3_testing, x4: x4_testing, z: z_testing,
                        y: y_testing}
                    prob_after = prob.eval(feed_dict=feed_testing_after)
                elif key == 'x3_vars':
                    data_increase = np.copy(x3_testing)
                    data_increase[:,i] += delta_increase
                    feed_testing_after = {x0: x0_testing, x1: x1_testing, x2: x2_testing, x3: data_increase, x4: x4_testing, z: z_testing,
                        y: y_testing}
                    prob_after = prob.eval(feed_dict=feed_testing_after)
                elif key == 'x4_vars':
                    data_increase = np.copy(x4_testing)
                    data_increase[:,i] += delta_increase
                    feed_testing_after = {x0: x0_testing, x1: x1_testing, x2: x2_testing, x3: x3_testing, x4: data_increase, z: z_testing,
                        y: y_testing}
                    prob_after = prob.eval(feed_dict=feed_testing_after)

                for mode in range(num_modes):
                    elasticity_individual = (prob_after[:,mode] - prob_before[:,mode]) / prob_before[:,mode] / delta_increase * df_sp_test_nonstand.loc[:, var_name] / df_sp_test_nonstand.loc[:, var_name].std()
                    elasticity = np.mean(elasticity_individual)
                    elast_records[str(mode) + '___' + var_name] = [elasticity]

    return elast_records


def est_cm_resnet(data, cm_param_dic, penalty_const, MODEL_NAME, restore = False, n_epoches = 10000, n_mini_batch=100, M=3,
                  n_hidden=100, simul_data = None,Dropout = False, Dropout_rate = 0.01, BN = False, x_dim = 15, K = 2, D = 7, current_best_acc = 0):
    #
    x0_training,x1_training,x2_training,x3_training,x4_training,z_training,y_training = data['training']
    x0_testing,x1_testing,x2_testing,x3_testing,x4_testing,z_testing,y_testing = data['testing']
    #             
    N, D0 = x0_training.shape
    N, D1 = x1_training.shape
    N, D2 = x2_training.shape
    N, D3 = x3_training.shape
    N, D4 = x4_training.shape
    N, DZ = z_training.shape
    
    D_dic = {'d0':D0, 'd1':D1, 'd2':D2, 'd3':D3, 'd4':D4, 'dz':DZ}
    
    # model 
    tf.reset_default_graph()
    
    x0 = tf.placeholder(dtype = tf.float32, shape = (None, D0), name = 'x0')
    x1 = tf.placeholder(dtype = tf.float32, shape = (None, D1), name = 'x1')
    x2 = tf.placeholder(dtype = tf.float32, shape = (None, D2), name = 'x2')
    x3 = tf.placeholder(dtype = tf.float32, shape = (None, D3), name = 'x3')
    x4 = tf.placeholder(dtype = tf.float32, shape = (None, D4), name = 'x4')
    z = tf.placeholder(dtype = tf.float32, shape = (None, DZ), name = 'z')
    y = tf.placeholder(dtype = tf.int64, shape = (None), name = 'y')
    x = tf.concat([x0,x1,x2,x3,x4,z], axis = 1, name = 'x')
    
    hidden_x0 = x0
    hidden_x1 = x1
    hidden_x2 = x2
    hidden_x3 = x3
    hidden_x4 = x4
    hidden_z = z
    
    hidden_dic = {}
    hidden_dic['x0'] = hidden_x0
    hidden_dic['x1'] = hidden_x1
    hidden_dic['x2'] = hidden_x2
    hidden_dic['x3'] = hidden_x3
    hidden_dic['x4'] = hidden_x4
    hidden_dic['z'] = hidden_z        

    # initialize parameters
    params_dic = {}
    for j in range(5):
        w_name = 'w'+str(j)
        b_name = 'b'+str(j)
        wz_name = 'wz'+str(j)
        d_name = 'd'+str(j)
        params_dic[w_name] = cm_param_dic[w_name]
        params_dic[b_name] = cm_param_dic[b_name]    
        params_dic[wz_name] = cm_param_dic[wz_name] 
    #
    output_dic = {}
    for j in range(5):
        layer_name = 'x'+str(j)
        w_name = 'w'+str(j)
        b_name = 'b'+str(j)
        wz_name = 'wz'+str(j)        
        hidden_j = tf.add(tf.add(tf.matmul(hidden_dic[layer_name], params_dic[w_name]), tf.matmul(hidden_dic['z'], params_dic[wz_name])), params_dic[b_name])
        output_name = 'u'+str(j)
        output_dic[output_name] = hidden_j    
    u_cm = tf.concat([output_dic['u0'], output_dic['u1'], output_dic['u2'], output_dic['u3'], output_dic['u4']], axis = 1, name = 'u')

    # train dnn part
    with tf.name_scope("dnn"):
        hidden = x
        for i in range(M):
            hidden = standard_hidden_layer(hidden, n_hidden, BN, Dropout, Dropout_rate)
        u_dnn = tf.layers.dense(hidden, K, name = 'output')

    # u and prob
    u = u_cm + u_dnn
    prob = tf.nn.softmax(u)

    # gradient info: only alternative 1 (bus) respect to inputs
    gradient_u1_x = [tf.concat([tf.gradients(u[:,1], [x0])[0], tf.gradients(u[:,1], [x1])[0], tf.gradients(u[:,1], [x2])[0], tf.gradients(u[:,1], [x3])[0],
                                tf.gradients(u[:,1], [x4])[0], tf.gradients(u[:,1], [z])[0]], axis = 1)]
    
    gradient_prob1_x = [tf.concat([tf.gradients(prob[:,1], [x0])[0], tf.gradients(prob[:,1], [x1])[0], tf.gradients(prob[:,1], [x2])[0], 
                                   tf.gradients(prob[:,1], [x3])[0], tf.gradients(prob[:,1], [x4])[0], tf.gradients(prob[:,1], [z])[0]], axis = 1)]

    # regularization l2
    l2_regularizer = tf.contrib.layers.l2_regularizer(scale=penalty_const, scope=None)
    vars_ = tf.trainable_variables()
    regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, vars_)
    
    # cost
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = u, labels = y), name = 'cost')
    cost += regularization_penalty
    log_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = u, labels = y), name = 'log_loss')
    optimizer = tf.train.AdamOptimizer() # opt: better than simple gradient descents: always converge to stable numbers.
    #    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate) # opt objective
    training_op = optimizer.minimize(cost) # minimize the opt objective

    # 
    gradient_cost_x = [tf.concat([tf.gradients(cost, [x0])[0], tf.gradients(cost, [x1])[0], tf.gradients(cost, [x2])[0], tf.gradients(cost, [x3])[0], tf.gradients(cost, [x4])[0], tf.gradients(cost, [z])[0]], axis = 1)]
    
    # eval params
    pred_y = tf.argmax(prob,1)

    correct = tf.nn.in_top_k(u, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    cost_training=[]
    cost_testing=[]
    accuracy_training=[]
    accuracy_testing=[]
    log_loss_training=[]
    log_loss_testing=[]
    # f1_score_training = []
    # f1_score_testing = []


    with tf.Session() as sess:
        # always run this to train the model
        init.run()
        feed_training = {x0: x0_training, x1: x1_training, x2: x2_training, x3: x3_training, x4: x4_training, z: z_training, y: y_training}
        feed_testing = {x0: x0_testing, x1: x1_testing, x2: x2_testing, x3: x3_testing, x4: x4_testing, z: z_testing, y: y_testing}
        
        if restore == True:
            saver.restore(sess, "tmp/"+MODEL_NAME+".ckpt")
            u_training = u.eval(feed_dict = feed_training)
            u_testing = u.eval(feed_dict = feed_testing)
            prob_training = prob.eval(feed_dict = feed_training)
            prob_testing = prob.eval(feed_dict = feed_testing)
            u_cm_training = u_cm.eval(feed_dict = feed_training)
            u_dnn_training = u_dnn.eval(feed_dict=feed_training)
            u_cm_testing = u_cm.eval(feed_dict = feed_testing)
            u_dnn_testing = u_dnn.eval(feed_dict=feed_testing)
            # 
            gradient_u1_x_training=gradient_u1_x[0].eval(feed_dict = feed_training)
            gradient_u1_x_testing=gradient_u1_x[0].eval(feed_dict = feed_testing)
            gradient_prob1_x_training=gradient_prob1_x[0].eval(feed_dict = feed_training)
            gradient_prob1_x_testing=gradient_prob1_x[0].eval(feed_dict = feed_testing)
            gradient_cost_x_training=gradient_cost_x[0].eval(feed_dict = feed_training)          
            gradient_cost_x_testing=gradient_cost_x[0].eval(feed_dict = feed_testing)
            #
            current_cost_training = cost.eval(feed_dict=feed_training)
            current_cost_testing = cost.eval(feed_dict=feed_testing)
            current_accuracy_training = accuracy.eval(feed_dict=feed_training)
            current_accuracy_testing = accuracy.eval(feed_dict=feed_testing)
            current_log_loss_training = log_loss.eval(feed_dict = feed_training)
            current_log_loss_testing = log_loss.eval(feed_dict = feed_testing)

            # current_pred_y_train = pred_y.eval(feed_dict=feed_training)
            # current_pred_y_test = pred_y.eval(feed_dict=feed_testing)
            # current_f1_train = f1_score(y_training, current_pred_y_train, average='micro')
            # current_f1_test = f1_score(y_testing, current_pred_y_test, average='micro')

            #
            log_loss_training.append(current_log_loss_training)
            log_loss_testing.append(current_log_loss_testing)
            cost_training.append(current_cost_training)
            cost_testing.append(current_cost_testing)
            accuracy_training.append(current_accuracy_training)
            accuracy_testing.append(current_accuracy_testing)
            # f1_score_training.append(current_f1_train)
            # f1_score_testing.append(current_f1_test)

            # save
            params_values = {}
            hyper_params_values = {}
            output_values = {}
            
            output_values['u_training'] = u_training
            output_values['u_testing'] = u_testing
            output_values['prob_training'] = prob_training
            output_values['prob_testing'] = prob_testing
            output_values['u_cm_training'] = u_cm_training
            output_values['u_dnn_training'] = u_dnn_training
            output_values['u_cm_testing'] = u_cm_testing
            output_values['u_dnn_testing'] = u_dnn_testing

            output_values['cost_training'] = cost_training
            output_values['cost_testing'] = cost_testing
            output_values['accuracy_training'] = accuracy_training
            output_values['accuracy_testing'] = accuracy_testing
            output_values['log_loss_training'] = log_loss_training
            output_values['log_loss_testing'] = log_loss_testing
            # output_values['f1_score_training'] = f1_score_training
            # output_values['f1_score_testing'] = f1_score_testing


            output_values['gradient_u1_x_training'] = gradient_u1_x_training
            output_values['gradient_u1_x_testing'] = gradient_u1_x_testing
            output_values['gradient_prob1_x_training'] = gradient_prob1_x_training
            output_values['gradient_prob1_x_testing'] = gradient_prob1_x_testing
            output_values['gradient_cost_x_training'] = gradient_cost_x_training
            output_values['gradient_cost_x_testing'] = gradient_cost_x_testing
            
        if restore == False:
            for i in range(n_epoches):
                # 
                X0_batch, X1_batch, X2_batch, X3_batch, X4_batch, Z_batch, Y_batch = obtain_mini_batch_dnn_alt_specific(x0_training, x1_training, x2_training, 
                    x3_training, x4_training, z_training, y_training, n_mini_batch = n_mini_batch)
                sess.run(training_op, feed_dict = {x0:X0_batch, x1:X1_batch, x2:X2_batch, x3:X3_batch, x4:X4_batch, z:Z_batch, y:Y_batch})
                # 
                if i % 1000 == 0:
                    print("Iterations", i, "Cost = ", cost.eval(feed_dict = feed_training))
                if i % 20 == 0:
                    current_cost_training = cost.eval(feed_dict=feed_training)
                    current_cost_testing = cost.eval(feed_dict=feed_testing)
                    current_accuracy_training = accuracy.eval(feed_dict=feed_training)
                    current_accuracy_testing = accuracy.eval(feed_dict=feed_testing)
                    current_log_loss_training = log_loss.eval(feed_dict = feed_training)
                    current_log_loss_testing = log_loss.eval(feed_dict = feed_testing)
                    # current_pred_y_train = pred_y.eval(feed_dict=feed_training)
                    # current_pred_y_test = pred_y.eval(feed_dict=feed_testing)
                    # current_f1_train = f1_score(y_training, current_pred_y_train, average='micro')
                    # current_f1_test = f1_score(y_testing, current_pred_y_test, average='micro')

                    log_loss_training.append(current_log_loss_training)
                    log_loss_testing.append(current_log_loss_testing)
                    cost_training.append(current_cost_training)
                    cost_testing.append(current_cost_testing)
                    accuracy_training.append(current_accuracy_training)
                    accuracy_testing.append(current_accuracy_testing)
                    # f1_score_training.append(current_f1_train)
                    # f1_score_testing.append(current_f1_test)

            # save the model
            if accuracy_testing[-1] >= current_best_acc:
                saver.save(sess, "tmp/"+MODEL_NAME+".ckpt")
                current_best_acc = accuracy_testing[-1]

            # 
            u_training = u.eval(feed_dict = feed_training)
            u_testing = u.eval(feed_dict = feed_testing)
            prob_training = prob.eval(feed_dict = feed_training)
            prob_testing = prob.eval(feed_dict = feed_testing)
            u_cm_training = u_cm.eval(feed_dict = feed_training)
            u_dnn_training = u_dnn.eval(feed_dict=feed_training)
            u_cm_testing = u_cm.eval(feed_dict = feed_testing)
            u_dnn_testing = u_dnn.eval(feed_dict=feed_testing)

            #
            gradient_u1_x_training=gradient_u1_x[0].eval(feed_dict = feed_training)
            gradient_u1_x_testing=gradient_u1_x[0].eval(feed_dict = feed_testing)
            gradient_prob1_x_training=gradient_prob1_x[0].eval(feed_dict = feed_training)
            gradient_prob1_x_testing=gradient_prob1_x[0].eval(feed_dict = feed_testing)
            gradient_cost_x_training=gradient_cost_x[0].eval(feed_dict = feed_training)            
            gradient_cost_x_testing=gradient_cost_x[0].eval(feed_dict = feed_testing)            

            params_values = {}
            hyper_params_values = {}
            output_values = {}
                        
            ''' evlauate the model by testing data'''
            print("Final Training Accuracy: ", accuracy.eval(feed_dict = feed_training))
            
            output_values['cost_training'] = cost_training
            output_values['cost_testing'] = cost_testing
            output_values['accuracy_training'] = accuracy_training
            output_values['accuracy_testing'] = accuracy_testing
            output_values['log_loss_training'] = log_loss_training
            output_values['log_loss_testing'] = log_loss_testing
            # output_values['f1_score_training'] = f1_score_training
            # output_values['f1_score_testing'] = f1_score_testing
            output_values['u_cm_training'] = u_cm_training
            output_values['u_dnn_training'] = u_dnn_training
            output_values['u_cm_testing'] = u_cm_testing
            output_values['u_dnn_testing'] = u_dnn_testing


            output_values['u_training'] = u_training
            output_values['u_testing'] = u_testing
            output_values['prob_training'] = prob_training
            output_values['prob_testing'] = prob_testing
            
            output_values['gradient_u1_x_training'] = gradient_u1_x_training
            output_values['gradient_u1_x_testing'] = gradient_u1_x_testing
            output_values['gradient_prob1_x_training'] = gradient_prob1_x_training
            output_values['gradient_prob1_x_testing'] = gradient_prob1_x_testing
            output_values['gradient_cost_x_training'] = gradient_cost_x_training
            output_values['gradient_cost_x_testing'] = gradient_cost_x_testing

            output_values['current_best_acc'] = current_best_acc
            
    return params_values,hyper_params_values,output_values


def est_cm_resnet_use_delta(data, cm_param_dic, penalty_const, MODEL_NAME, restore=False, n_epoches=10000, l2_regu = 1e-50, n_mini_batch=100, M=3,
                  n_hidden=100, simul_data=None, Dropout=False, Dropout_rate=0.01, BN=False, x_dim=15, K=2, D=7,
                  current_best_acc=0):
    #
    x0_training, x1_training, x2_training, x3_training, x4_training, z_training, y_training = data['training']
    x0_testing, x1_testing, x2_testing, x3_testing, x4_testing, z_testing, y_testing = data['testing']
    #
    N, D0 = x0_training.shape
    N, D1 = x1_training.shape
    N, D2 = x2_training.shape
    N, D3 = x3_training.shape
    N, D4 = x4_training.shape
    N, DZ = z_training.shape

    D_dic = {'d0': D0, 'd1': D1, 'd2': D2, 'd3': D3, 'd4': D4, 'dz': DZ}

    # model
    tf.reset_default_graph()

    x0 = tf.placeholder(dtype=tf.float32, shape=(None, D0), name='x0')
    x1 = tf.placeholder(dtype=tf.float32, shape=(None, D1), name='x1')
    x2 = tf.placeholder(dtype=tf.float32, shape=(None, D2), name='x2')
    x3 = tf.placeholder(dtype=tf.float32, shape=(None, D3), name='x3')
    x4 = tf.placeholder(dtype=tf.float32, shape=(None, D4), name='x4')
    z = tf.placeholder(dtype=tf.float32, shape=(None, DZ), name='z')
    y = tf.placeholder(dtype=tf.int64, shape=(None), name='y')
    x = tf.concat([x0, x1, x2, x3, x4, z], axis=1, name='x')

    hidden_x0 = x0
    hidden_x1 = x1
    hidden_x2 = x2
    hidden_x3 = x3
    hidden_x4 = x4
    hidden_z = z

    hidden_dic = {}
    hidden_dic['x0'] = hidden_x0
    hidden_dic['x1'] = hidden_x1
    hidden_dic['x2'] = hidden_x2
    hidden_dic['x3'] = hidden_x3
    hidden_dic['x4'] = hidden_x4
    hidden_dic['z'] = hidden_z

    # initialize parameters
    params_dic = {}
    for j in range(5):
        w_name = 'w' + str(j)
        b_name = 'b' + str(j)
        wz_name = 'wz' + str(j)
        d_name = 'd' + str(j)
        params_dic[w_name] = cm_param_dic[w_name]
        params_dic[b_name] = cm_param_dic[b_name]
        params_dic[wz_name] = cm_param_dic[wz_name]
        #
    output_dic = {}
    for j in range(5):
        layer_name = 'x' + str(j)
        w_name = 'w' + str(j)
        b_name = 'b' + str(j)
        wz_name = 'wz' + str(j)
        hidden_j = tf.add(tf.add(tf.matmul(hidden_dic[layer_name], params_dic[w_name]),
                                 tf.matmul(hidden_dic['z'], params_dic[wz_name])), params_dic[b_name])
        output_name = 'u' + str(j)
        output_dic[output_name] = hidden_j
    u_cm = tf.concat([output_dic['u0'], output_dic['u1'], output_dic['u2'], output_dic['u3'], output_dic['u4']], axis=1,
                     name='u')

    # train dnn part
    with tf.name_scope("dnn"):
        hidden = x
        for i in range(M):
            hidden = standard_hidden_layer(hidden, n_hidden, BN, Dropout, Dropout_rate)
        u_dnn = tf.layers.dense(hidden, K, name='output')

    # u and prob
    u = (1 - penalty_const) * u_cm + penalty_const * u_dnn
    prob = tf.nn.softmax(u)

    # gradient info: only alternative 1 (bus) respect to inputs
    gradient_u1_x = [tf.concat(
        [tf.gradients(u[:, 1], [x0])[0], tf.gradients(u[:, 1], [x1])[0], tf.gradients(u[:, 1], [x2])[0],
         tf.gradients(u[:, 1], [x3])[0],
         tf.gradients(u[:, 1], [x4])[0], tf.gradients(u[:, 1], [z])[0]], axis=1)]

    gradient_prob1_x = [tf.concat(
        [tf.gradients(prob[:, 1], [x0])[0], tf.gradients(prob[:, 1], [x1])[0], tf.gradients(prob[:, 1], [x2])[0],
         tf.gradients(prob[:, 1], [x3])[0], tf.gradients(prob[:, 1], [x4])[0], tf.gradients(prob[:, 1], [z])[0]],
        axis=1)]

    # regularization l2
    l2_regularizer = tf.contrib.layers.l2_regularizer(scale=l2_regu, scope=None)
    vars_ = tf.trainable_variables()
    regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, vars_)

    # cost
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=u, labels=y), name='cost')
    cost += regularization_penalty
    log_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=u, labels=y), name='log_loss')
    optimizer = tf.train.AdamOptimizer()  # opt: better than simple gradient descents: always converge to stable numbers.
    #    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate) # opt objective
    training_op = optimizer.minimize(cost)  # minimize the opt objective

    #
    gradient_cost_x = [tf.concat([tf.gradients(cost, [x0])[0], tf.gradients(cost, [x1])[0], tf.gradients(cost, [x2])[0],
                                  tf.gradients(cost, [x3])[0], tf.gradients(cost, [x4])[0], tf.gradients(cost, [z])[0]],
                                 axis=1)]

    # eval params
    pred_y = tf.argmax(prob, 1)

    correct = tf.nn.in_top_k(u, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    cost_training = []
    cost_testing = []
    accuracy_training = []
    accuracy_testing = []
    log_loss_training = []
    log_loss_testing = []
    # f1_score_training = []
    # f1_score_testing = []

    with tf.Session() as sess:
        # always run this to train the model
        init.run()
        feed_training = {x0: x0_training, x1: x1_training, x2: x2_training, x3: x3_training, x4: x4_training,
                         z: z_training, y: y_training}
        feed_testing = {x0: x0_testing, x1: x1_testing, x2: x2_testing, x3: x3_testing, x4: x4_testing, z: z_testing,
                        y: y_testing}

        if restore == True:
            saver.restore(sess, "tmp/" + MODEL_NAME + ".ckpt")
            u_training = u.eval(feed_dict=feed_training)
            u_testing = u.eval(feed_dict=feed_testing)
            prob_training = prob.eval(feed_dict=feed_training)
            prob_testing = prob.eval(feed_dict=feed_testing)
            u_cm_training = u_cm.eval(feed_dict=feed_training)
            u_dnn_training = u_dnn.eval(feed_dict=feed_training)
            u_cm_testing = u_cm.eval(feed_dict=feed_testing)
            u_dnn_testing = u_dnn.eval(feed_dict=feed_testing)
            #
            gradient_u1_x_training = gradient_u1_x[0].eval(feed_dict=feed_training)
            gradient_u1_x_testing = gradient_u1_x[0].eval(feed_dict=feed_testing)
            gradient_prob1_x_training = gradient_prob1_x[0].eval(feed_dict=feed_training)
            gradient_prob1_x_testing = gradient_prob1_x[0].eval(feed_dict=feed_testing)
            gradient_cost_x_training = gradient_cost_x[0].eval(feed_dict=feed_training)
            gradient_cost_x_testing = gradient_cost_x[0].eval(feed_dict=feed_testing)
            #
            current_cost_training = cost.eval(feed_dict=feed_training)
            current_cost_testing = cost.eval(feed_dict=feed_testing)
            current_accuracy_training = accuracy.eval(feed_dict=feed_training)
            current_accuracy_testing = accuracy.eval(feed_dict=feed_testing)
            current_log_loss_training = log_loss.eval(feed_dict=feed_training)
            current_log_loss_testing = log_loss.eval(feed_dict=feed_testing)

            # current_pred_y_train = pred_y.eval(feed_dict=feed_training)
            # current_pred_y_test = pred_y.eval(feed_dict=feed_testing)
            # current_f1_train = f1_score(y_training, current_pred_y_train, average='micro')
            # current_f1_test = f1_score(y_testing, current_pred_y_test, average='micro')

            #
            log_loss_training.append(current_log_loss_training)
            log_loss_testing.append(current_log_loss_testing)
            cost_training.append(current_cost_training)
            cost_testing.append(current_cost_testing)
            accuracy_training.append(current_accuracy_training)
            accuracy_testing.append(current_accuracy_testing)
            # f1_score_training.append(current_f1_train)
            # f1_score_testing.append(current_f1_test)

            # save
            params_values = {}
            hyper_params_values = {}
            output_values = {}

            output_values['u_training'] = u_training
            output_values['u_testing'] = u_testing
            output_values['prob_training'] = prob_training
            output_values['prob_testing'] = prob_testing
            output_values['u_cm_training'] = u_cm_training
            output_values['u_dnn_training'] = u_dnn_training
            output_values['u_cm_testing'] = u_cm_testing
            output_values['u_dnn_testing'] = u_dnn_testing

            output_values['cost_training'] = cost_training
            output_values['cost_testing'] = cost_testing
            output_values['accuracy_training'] = accuracy_training
            output_values['accuracy_testing'] = accuracy_testing
            output_values['log_loss_training'] = log_loss_training
            output_values['log_loss_testing'] = log_loss_testing
            # output_values['f1_score_training'] = f1_score_training
            # output_values['f1_score_testing'] = f1_score_testing

            output_values['gradient_u1_x_training'] = gradient_u1_x_training
            output_values['gradient_u1_x_testing'] = gradient_u1_x_testing
            output_values['gradient_prob1_x_training'] = gradient_prob1_x_training
            output_values['gradient_prob1_x_testing'] = gradient_prob1_x_testing
            output_values['gradient_cost_x_training'] = gradient_cost_x_training
            output_values['gradient_cost_x_testing'] = gradient_cost_x_testing

        if restore == False:
            for i in range(n_epoches):
                #
                X0_batch, X1_batch, X2_batch, X3_batch, X4_batch, Z_batch, Y_batch = obtain_mini_batch_dnn_alt_specific(
                    x0_training, x1_training, x2_training,
                    x3_training, x4_training, z_training, y_training, n_mini_batch=n_mini_batch)
                sess.run(training_op,
                         feed_dict={x0: X0_batch, x1: X1_batch, x2: X2_batch, x3: X3_batch, x4: X4_batch, z: Z_batch,
                                    y: Y_batch})
                #
                if i % 1000 == 0:
                    print("Iterations", i, "Cost = ", cost.eval(feed_dict=feed_training))
                if i % 20 == 0:
                    current_cost_training = cost.eval(feed_dict=feed_training)
                    current_cost_testing = cost.eval(feed_dict=feed_testing)
                    current_accuracy_training = accuracy.eval(feed_dict=feed_training)
                    current_accuracy_testing = accuracy.eval(feed_dict=feed_testing)
                    current_log_loss_training = log_loss.eval(feed_dict=feed_training)
                    current_log_loss_testing = log_loss.eval(feed_dict=feed_testing)
                    # current_pred_y_train = pred_y.eval(feed_dict=feed_training)
                    # current_pred_y_test = pred_y.eval(feed_dict=feed_testing)
                    # current_f1_train = f1_score(y_training, current_pred_y_train, average='micro')
                    # current_f1_test = f1_score(y_testing, current_pred_y_test, average='micro')

                    log_loss_training.append(current_log_loss_training)
                    log_loss_testing.append(current_log_loss_testing)
                    cost_training.append(current_cost_training)
                    cost_testing.append(current_cost_testing)
                    accuracy_training.append(current_accuracy_training)
                    accuracy_testing.append(current_accuracy_testing)
                    # f1_score_training.append(current_f1_train)
                    # f1_score_testing.append(current_f1_test)

            # save the model
            if accuracy_testing[-1] >= current_best_acc:
                saver.save(sess, "tmp/" + MODEL_NAME + ".ckpt")
                current_best_acc = accuracy_testing[-1]

            #
            u_training = u.eval(feed_dict=feed_training)
            u_testing = u.eval(feed_dict=feed_testing)
            prob_training = prob.eval(feed_dict=feed_training)
            prob_testing = prob.eval(feed_dict=feed_testing)
            u_cm_training = u_cm.eval(feed_dict=feed_training)
            u_dnn_training = u_dnn.eval(feed_dict=feed_training)
            u_cm_testing = u_cm.eval(feed_dict=feed_testing)
            u_dnn_testing = u_dnn.eval(feed_dict=feed_testing)

            #
            gradient_u1_x_training = gradient_u1_x[0].eval(feed_dict=feed_training)
            gradient_u1_x_testing = gradient_u1_x[0].eval(feed_dict=feed_testing)
            gradient_prob1_x_training = gradient_prob1_x[0].eval(feed_dict=feed_training)
            gradient_prob1_x_testing = gradient_prob1_x[0].eval(feed_dict=feed_testing)
            gradient_cost_x_training = gradient_cost_x[0].eval(feed_dict=feed_training)
            gradient_cost_x_testing = gradient_cost_x[0].eval(feed_dict=feed_testing)

            params_values = {}
            hyper_params_values = {}
            output_values = {}

            ''' evlauate the model by testing data'''
            print("Final Training Accuracy: ", accuracy.eval(feed_dict=feed_training))

            output_values['cost_training'] = cost_training
            output_values['cost_testing'] = cost_testing
            output_values['accuracy_training'] = accuracy_training
            output_values['accuracy_testing'] = accuracy_testing
            output_values['log_loss_training'] = log_loss_training
            output_values['log_loss_testing'] = log_loss_testing
            # output_values['f1_score_training'] = f1_score_training
            # output_values['f1_score_testing'] = f1_score_testing
            output_values['u_cm_training'] = u_cm_training
            output_values['u_dnn_training'] = u_dnn_training
            output_values['u_cm_testing'] = u_cm_testing
            output_values['u_dnn_testing'] = u_dnn_testing

            output_values['u_training'] = u_training
            output_values['u_testing'] = u_testing
            output_values['prob_training'] = prob_training
            output_values['prob_testing'] = prob_testing

            output_values['gradient_u1_x_training'] = gradient_u1_x_training
            output_values['gradient_u1_x_testing'] = gradient_u1_x_testing
            output_values['gradient_prob1_x_training'] = gradient_prob1_x_training
            output_values['gradient_prob1_x_testing'] = gradient_prob1_x_testing
            output_values['gradient_cost_x_training'] = gradient_cost_x_training
            output_values['gradient_cost_x_testing'] = gradient_cost_x_testing

            output_values['current_best_acc'] = current_best_acc

    return params_values, hyper_params_values, output_values




def est_cm_resnet_use_delta_save_train(data, cm_param_dic, penalty_const, MODEL_NAME, restore=False, n_epoches=10000, l2_regu = 1e-50, n_mini_batch=100, M=3,
                  n_hidden=100, simul_data=None, Dropout=False, Dropout_rate=0.01, BN=False, x_dim=15, K=2, D=7,
                  current_best_acc=0):
    #
    x0_training, x1_training, x2_training, x3_training, x4_training, z_training, y_training = data['training']
    x0_testing, x1_testing, x2_testing, x3_testing, x4_testing, z_testing, y_testing = data['testing']
    #
    N, D0 = x0_training.shape
    N, D1 = x1_training.shape
    N, D2 = x2_training.shape
    N, D3 = x3_training.shape
    N, D4 = x4_training.shape
    N, DZ = z_training.shape

    D_dic = {'d0': D0, 'd1': D1, 'd2': D2, 'd3': D3, 'd4': D4, 'dz': DZ}

    # model
    tf.reset_default_graph()

    x0 = tf.placeholder(dtype=tf.float32, shape=(None, D0), name='x0')
    x1 = tf.placeholder(dtype=tf.float32, shape=(None, D1), name='x1')
    x2 = tf.placeholder(dtype=tf.float32, shape=(None, D2), name='x2')
    x3 = tf.placeholder(dtype=tf.float32, shape=(None, D3), name='x3')
    x4 = tf.placeholder(dtype=tf.float32, shape=(None, D4), name='x4')
    z = tf.placeholder(dtype=tf.float32, shape=(None, DZ), name='z')
    y = tf.placeholder(dtype=tf.int64, shape=(None), name='y')
    x = tf.concat([x0, x1, x2, x3, x4, z], axis=1, name='x')

    hidden_x0 = x0
    hidden_x1 = x1
    hidden_x2 = x2
    hidden_x3 = x3
    hidden_x4 = x4
    hidden_z = z

    hidden_dic = {}
    hidden_dic['x0'] = hidden_x0
    hidden_dic['x1'] = hidden_x1
    hidden_dic['x2'] = hidden_x2
    hidden_dic['x3'] = hidden_x3
    hidden_dic['x4'] = hidden_x4
    hidden_dic['z'] = hidden_z

    # initialize parameters
    params_dic = {}
    for j in range(5):
        w_name = 'w' + str(j)
        b_name = 'b' + str(j)
        wz_name = 'wz' + str(j)
        d_name = 'd' + str(j)
        params_dic[w_name] = cm_param_dic[w_name]
        params_dic[b_name] = cm_param_dic[b_name]
        params_dic[wz_name] = cm_param_dic[wz_name]
        #
    output_dic = {}
    for j in range(5):
        layer_name = 'x' + str(j)
        w_name = 'w' + str(j)
        b_name = 'b' + str(j)
        wz_name = 'wz' + str(j)
        hidden_j = tf.add(tf.add(tf.matmul(hidden_dic[layer_name], params_dic[w_name]),
                                 tf.matmul(hidden_dic['z'], params_dic[wz_name])), params_dic[b_name])
        output_name = 'u' + str(j)
        output_dic[output_name] = hidden_j
    u_cm = tf.concat([output_dic['u0'], output_dic['u1'], output_dic['u2'], output_dic['u3'], output_dic['u4']], axis=1,
                     name='u')

    # train dnn part
    with tf.name_scope("dnn"):
        hidden = x
        for i in range(M):
            hidden = standard_hidden_layer(hidden, n_hidden, BN, Dropout, Dropout_rate)
        u_dnn = tf.layers.dense(hidden, K, name='output')

    # u and prob
    u = (1 - penalty_const) * u_cm + penalty_const * u_dnn
    prob = tf.nn.softmax(u)

    # gradient info: only alternative 1 (bus) respect to inputs
    gradient_u1_x = [tf.concat(
        [tf.gradients(u[:, 1], [x0])[0], tf.gradients(u[:, 1], [x1])[0], tf.gradients(u[:, 1], [x2])[0],
         tf.gradients(u[:, 1], [x3])[0],
         tf.gradients(u[:, 1], [x4])[0], tf.gradients(u[:, 1], [z])[0]], axis=1)]

    gradient_prob1_x = [tf.concat(
        [tf.gradients(prob[:, 1], [x0])[0], tf.gradients(prob[:, 1], [x1])[0], tf.gradients(prob[:, 1], [x2])[0],
         tf.gradients(prob[:, 1], [x3])[0], tf.gradients(prob[:, 1], [x4])[0], tf.gradients(prob[:, 1], [z])[0]],
        axis=1)]

    # regularization l2
    l2_regularizer = tf.contrib.layers.l2_regularizer(scale=l2_regu, scope=None)
    vars_ = tf.trainable_variables()
    regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, vars_)

    # cost
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=u, labels=y), name='cost')
    cost += regularization_penalty
    log_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=u, labels=y), name='log_loss')
    optimizer = tf.train.AdamOptimizer()  # opt: better than simple gradient descents: always converge to stable numbers.
    #    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate) # opt objective
    training_op = optimizer.minimize(cost)  # minimize the opt objective

    #
    gradient_cost_x = [tf.concat([tf.gradients(cost, [x0])[0], tf.gradients(cost, [x1])[0], tf.gradients(cost, [x2])[0],
                                  tf.gradients(cost, [x3])[0], tf.gradients(cost, [x4])[0], tf.gradients(cost, [z])[0]],
                                 axis=1)]

    # eval params
    pred_y = tf.argmax(prob, 1)

    correct = tf.nn.in_top_k(u, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    cost_training = []
    cost_testing = []
    accuracy_training = []
    accuracy_testing = []
    log_loss_training = []
    log_loss_testing = []
    # f1_score_training = []
    # f1_score_testing = []

    with tf.Session() as sess:
        # always run this to train the model
        init.run()
        feed_training = {x0: x0_training, x1: x1_training, x2: x2_training, x3: x3_training, x4: x4_training,
                         z: z_training, y: y_training}
        feed_testing = {x0: x0_testing, x1: x1_testing, x2: x2_testing, x3: x3_testing, x4: x4_testing, z: z_testing,
                        y: y_testing}

        if restore == True:
            saver.restore(sess, "tmp/" + MODEL_NAME + ".ckpt")
            u_training = u.eval(feed_dict=feed_training)
            u_testing = u.eval(feed_dict=feed_testing)
            prob_training = prob.eval(feed_dict=feed_training)
            prob_testing = prob.eval(feed_dict=feed_testing)
            u_cm_training = u_cm.eval(feed_dict=feed_training)
            u_dnn_training = u_dnn.eval(feed_dict=feed_training)
            u_cm_testing = u_cm.eval(feed_dict=feed_testing)
            u_dnn_testing = u_dnn.eval(feed_dict=feed_testing)
            #
            gradient_u1_x_training = gradient_u1_x[0].eval(feed_dict=feed_training)
            gradient_u1_x_testing = gradient_u1_x[0].eval(feed_dict=feed_testing)
            gradient_prob1_x_training = gradient_prob1_x[0].eval(feed_dict=feed_training)
            gradient_prob1_x_testing = gradient_prob1_x[0].eval(feed_dict=feed_testing)
            gradient_cost_x_training = gradient_cost_x[0].eval(feed_dict=feed_training)
            gradient_cost_x_testing = gradient_cost_x[0].eval(feed_dict=feed_testing)
            #
            current_cost_training = cost.eval(feed_dict=feed_training)
            current_cost_testing = cost.eval(feed_dict=feed_testing)
            current_accuracy_training = accuracy.eval(feed_dict=feed_training)
            current_accuracy_testing = accuracy.eval(feed_dict=feed_testing)
            current_log_loss_training = log_loss.eval(feed_dict=feed_training)
            current_log_loss_testing = log_loss.eval(feed_dict=feed_testing)

            # current_pred_y_train = pred_y.eval(feed_dict=feed_training)
            # current_pred_y_test = pred_y.eval(feed_dict=feed_testing)
            # current_f1_train = f1_score(y_training, current_pred_y_train, average='micro')
            # current_f1_test = f1_score(y_testing, current_pred_y_test, average='micro')

            #
            log_loss_training.append(current_log_loss_training)
            log_loss_testing.append(current_log_loss_testing)
            cost_training.append(current_cost_training)
            cost_testing.append(current_cost_testing)
            accuracy_training.append(current_accuracy_training)
            accuracy_testing.append(current_accuracy_testing)
            # f1_score_training.append(current_f1_train)
            # f1_score_testing.append(current_f1_test)

            # save
            params_values = {}
            hyper_params_values = {}
            output_values = {}

            output_values['u_training'] = u_training
            output_values['u_testing'] = u_testing
            output_values['prob_training'] = prob_training
            output_values['prob_testing'] = prob_testing
            output_values['u_cm_training'] = u_cm_training
            output_values['u_dnn_training'] = u_dnn_training
            output_values['u_cm_testing'] = u_cm_testing
            output_values['u_dnn_testing'] = u_dnn_testing

            output_values['cost_training'] = cost_training
            output_values['cost_testing'] = cost_testing
            output_values['accuracy_training'] = accuracy_training
            output_values['accuracy_testing'] = accuracy_testing
            output_values['log_loss_training'] = log_loss_training
            output_values['log_loss_testing'] = log_loss_testing
            # output_values['f1_score_training'] = f1_score_training
            # output_values['f1_score_testing'] = f1_score_testing

            output_values['gradient_u1_x_training'] = gradient_u1_x_training
            output_values['gradient_u1_x_testing'] = gradient_u1_x_testing
            output_values['gradient_prob1_x_training'] = gradient_prob1_x_training
            output_values['gradient_prob1_x_testing'] = gradient_prob1_x_testing
            output_values['gradient_cost_x_training'] = gradient_cost_x_training
            output_values['gradient_cost_x_testing'] = gradient_cost_x_testing

        if restore == False:
            for i in range(n_epoches):
                #
                X0_batch, X1_batch, X2_batch, X3_batch, X4_batch, Z_batch, Y_batch = obtain_mini_batch_dnn_alt_specific(
                    x0_training, x1_training, x2_training,
                    x3_training, x4_training, z_training, y_training, n_mini_batch=n_mini_batch)
                sess.run(training_op,
                         feed_dict={x0: X0_batch, x1: X1_batch, x2: X2_batch, x3: X3_batch, x4: X4_batch, z: Z_batch,
                                    y: Y_batch})
                #
                if i % 1000 == 0:
                    print("Iterations", i, "Cost = ", cost.eval(feed_dict=feed_training))

                current_cost_training = cost.eval(feed_dict=feed_training)
                current_cost_testing = cost.eval(feed_dict=feed_testing)
                current_accuracy_training = accuracy.eval(feed_dict=feed_training)
                current_accuracy_testing = accuracy.eval(feed_dict=feed_testing)
                current_log_loss_training = log_loss.eval(feed_dict=feed_training)
                current_log_loss_testing = log_loss.eval(feed_dict=feed_testing)
                # current_pred_y_train = pred_y.eval(feed_dict=feed_training)
                # current_pred_y_test = pred_y.eval(feed_dict=feed_testing)
                # current_f1_train = f1_score(y_training, current_pred_y_train, average='micro')
                # current_f1_test = f1_score(y_testing, current_pred_y_test, average='micro')

                log_loss_training.append(current_log_loss_training)
                log_loss_testing.append(current_log_loss_testing)
                cost_training.append(current_cost_training)
                cost_testing.append(current_cost_testing)
                accuracy_training.append(current_accuracy_training)
                accuracy_testing.append(current_accuracy_testing)
                # f1_score_training.append(current_f1_train)
                # f1_score_testing.append(current_f1_test)

            # save the model
            if accuracy_testing[-1] >= current_best_acc:
                saver.save(sess, "tmp/" + MODEL_NAME + ".ckpt")
                current_best_acc = accuracy_testing[-1]

            #
            u_training = u.eval(feed_dict=feed_training)
            u_testing = u.eval(feed_dict=feed_testing)
            prob_training = prob.eval(feed_dict=feed_training)
            prob_testing = prob.eval(feed_dict=feed_testing)
            u_cm_training = u_cm.eval(feed_dict=feed_training)
            u_dnn_training = u_dnn.eval(feed_dict=feed_training)
            u_cm_testing = u_cm.eval(feed_dict=feed_testing)
            u_dnn_testing = u_dnn.eval(feed_dict=feed_testing)

            #
            gradient_u1_x_training = gradient_u1_x[0].eval(feed_dict=feed_training)
            gradient_u1_x_testing = gradient_u1_x[0].eval(feed_dict=feed_testing)
            gradient_prob1_x_training = gradient_prob1_x[0].eval(feed_dict=feed_training)
            gradient_prob1_x_testing = gradient_prob1_x[0].eval(feed_dict=feed_testing)
            gradient_cost_x_training = gradient_cost_x[0].eval(feed_dict=feed_training)
            gradient_cost_x_testing = gradient_cost_x[0].eval(feed_dict=feed_testing)

            params_values = {}
            hyper_params_values = {}
            output_values = {}

            ''' evlauate the model by testing data'''
            print("Final Training Accuracy: ", accuracy.eval(feed_dict=feed_training))

            output_values['cost_training'] = cost_training
            output_values['cost_testing'] = cost_testing
            output_values['accuracy_training'] = accuracy_training
            output_values['accuracy_testing'] = accuracy_testing
            output_values['log_loss_training'] = log_loss_training
            output_values['log_loss_testing'] = log_loss_testing
            # output_values['f1_score_training'] = f1_score_training
            # output_values['f1_score_testing'] = f1_score_testing
            output_values['u_cm_training'] = u_cm_training
            output_values['u_dnn_training'] = u_dnn_training
            output_values['u_cm_testing'] = u_cm_testing
            output_values['u_dnn_testing'] = u_dnn_testing

            output_values['u_training'] = u_training
            output_values['u_testing'] = u_testing
            output_values['prob_training'] = prob_training
            output_values['prob_testing'] = prob_testing

            output_values['gradient_u1_x_training'] = gradient_u1_x_training
            output_values['gradient_u1_x_testing'] = gradient_u1_x_testing
            output_values['gradient_prob1_x_training'] = gradient_prob1_x_training
            output_values['gradient_prob1_x_testing'] = gradient_prob1_x_testing
            output_values['gradient_cost_x_training'] = gradient_cost_x_training
            output_values['gradient_cost_x_testing'] = gradient_cost_x_testing

            output_values['current_best_acc'] = current_best_acc

    return params_values, hyper_params_values, output_values
















def est_cm_resnet_reverse(data, dnn_param_dic, penalty_const, MODEL_NAME, restore=False, n_epoches=10000, n_mini_batch=100, M=3,
                  n_hidden=100, simul_data=None, Dropout=False, Dropout_rate=0.01, BN=False, x_dim=15, K=2, D=7):
    #
    x0_training, x1_training, x2_training, x3_training, x4_training, z_training, y_training = data['training']
    x0_testing, x1_testing, x2_testing, x3_testing, x4_testing, z_testing, y_testing = data['testing']
    #
    N, D0 = x0_training.shape
    N, D1 = x1_training.shape
    N, D2 = x2_training.shape
    N, D3 = x3_training.shape
    N, D4 = x4_training.shape
    N, DZ = z_training.shape

    D_dic = {'d0': D0, 'd1': D1, 'd2': D2, 'd3': D3, 'd4': D4, 'dz': DZ}

    # model
    tf.reset_default_graph()

    x0 = tf.placeholder(dtype=tf.float32, shape=(None, D0), name='x0')
    x1 = tf.placeholder(dtype=tf.float32, shape=(None, D1), name='x1')
    x2 = tf.placeholder(dtype=tf.float32, shape=(None, D2), name='x2')
    x3 = tf.placeholder(dtype=tf.float32, shape=(None, D3), name='x3')
    x4 = tf.placeholder(dtype=tf.float32, shape=(None, D4), name='x4')
    z = tf.placeholder(dtype=tf.float32, shape=(None, DZ), name='z')
    y = tf.placeholder(dtype=tf.int64, shape=(None), name='y')
    x = tf.concat([x0, x1, x2, x3, x4, z], axis=1, name='x')

    hidden_x0 = x0
    hidden_x1 = x1
    hidden_x2 = x2
    hidden_x3 = x3
    hidden_x4 = x4
    hidden_z = z

    hidden_dic = {}
    hidden_dic['x0'] = hidden_x0
    hidden_dic['x1'] = hidden_x1
    hidden_dic['x2'] = hidden_x2
    hidden_dic['x3'] = hidden_x3
    hidden_dic['x4'] = hidden_x4
    hidden_dic['z'] = hidden_z

    # initialize parameters
    params_dic = {}
    for j in range(5):
        w_name = 'w' + str(j)
        b_name = 'b' + str(j)
        wz_name = 'wz' + str(j)
        d_name = 'd' + str(j)
        params_dic[w_name] = tf.Variable(tf.random_normal([D_dic[d_name], 1]), dtype = tf.float32, name = w_name)
        params_dic[b_name] = tf.Variable(tf.random_normal([1]), dtype = tf.float32, name = b_name)
        params_dic[wz_name] = tf.Variable(tf.random_normal([D_dic['dz'], 1]), dtype = tf.float32, name = wz_name)
        #
    output_dic = {}
    for j in range(5):
        layer_name = 'x' + str(j)
        w_name = 'w' + str(j)
        b_name = 'b' + str(j)
        wz_name = 'wz' + str(j)
        hidden_j = tf.add(tf.add(tf.matmul(hidden_dic[layer_name], params_dic[w_name]),
                                 tf.matmul(hidden_dic['z'], params_dic[wz_name])), params_dic[b_name])
        output_name = 'u' + str(j)
        output_dic[output_name] = hidden_j
    u_cm = tf.concat([output_dic['u0'], output_dic['u1'], output_dic['u2'], output_dic['u3'], output_dic['u4']], axis=1,
                     name='u')

    # train dnn part
    with tf.name_scope("dnn"):
        hidden = x
        for i in range(M):
            hidden = standard_hidden_layer_constriants(hidden, n_hidden, BN, Dropout, Dropout_rate, dnn_param_dic, i)
        kernel_name = 'output/kernel:0'
        bias_name = 'output/bias:0'
        u_dnn = tf.layers.dense(hidden, K, name='output',kernel_constraint=lambda x: dnn_param_dic[kernel_name],
                                bias_constraint=lambda x: dnn_param_dic[bias_name])

    # u and prob
    u = u_cm + u_dnn
    prob = tf.nn.softmax(u)

    # gradient info: only alternative 1 (bus) respect to inputs
    gradient_u1_x = [tf.concat(
        [tf.gradients(u[:, 1], [x0])[0], tf.gradients(u[:, 1], [x1])[0], tf.gradients(u[:, 1], [x2])[0],
         tf.gradients(u[:, 1], [x3])[0],
         tf.gradients(u[:, 1], [x4])[0], tf.gradients(u[:, 1], [z])[0]], axis=1)]

    gradient_prob1_x = [tf.concat(
        [tf.gradients(prob[:, 1], [x0])[0], tf.gradients(prob[:, 1], [x1])[0], tf.gradients(prob[:, 1], [x2])[0],
         tf.gradients(prob[:, 1], [x3])[0], tf.gradients(prob[:, 1], [x4])[0], tf.gradients(prob[:, 1], [z])[0]],
        axis=1)]

    # regularization l2
    l2_regularizer = tf.contrib.layers.l2_regularizer(scale=penalty_const, scope=None)
    vars_ = tf.trainable_variables()
    regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, vars_)

    # cost
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=u, labels=y), name='cost')
    cost += regularization_penalty
    log_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=u, labels=y), name='log_loss')
    optimizer = tf.train.AdamOptimizer()  # opt: better than simple gradient descents: always converge to stable numbers.
    #    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate) # opt objective
    training_op = optimizer.minimize(cost)  # minimize the opt objective

    #
    gradient_cost_x = [tf.concat([tf.gradients(cost, [x0])[0], tf.gradients(cost, [x1])[0], tf.gradients(cost, [x2])[0],
                                  tf.gradients(cost, [x3])[0], tf.gradients(cost, [x4])[0], tf.gradients(cost, [z])[0]],
                                 axis=1)]

    # eval params
    correct = tf.nn.in_top_k(u, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    cost_training = []
    cost_testing = []
    accuracy_training = []
    accuracy_testing = []
    log_loss_training = []
    log_loss_testing = []

    with tf.Session() as sess:
        # always run this to train the model
        init.run()
        feed_training = {x0: x0_training, x1: x1_training, x2: x2_training, x3: x3_training, x4: x4_training,
                         z: z_training, y: y_training}
        feed_testing = {x0: x0_testing, x1: x1_testing, x2: x2_testing, x3: x3_testing, x4: x4_testing, z: z_testing,
                        y: y_testing}

        if restore == True:
            saver.restore(sess, "tmp/" + MODEL_NAME + ".ckpt")
            u_training = u.eval(feed_dict=feed_training)
            u_testing = u.eval(feed_dict=feed_testing)
            prob_training = prob.eval(feed_dict=feed_training)
            prob_testing = prob.eval(feed_dict=feed_testing)
            #
            gradient_u1_x_training = gradient_u1_x[0].eval(feed_dict=feed_training)
            gradient_u1_x_testing = gradient_u1_x[0].eval(feed_dict=feed_testing)
            gradient_prob1_x_training = gradient_prob1_x[0].eval(feed_dict=feed_training)
            gradient_prob1_x_testing = gradient_prob1_x[0].eval(feed_dict=feed_testing)
            gradient_cost_x_training = gradient_cost_x[0].eval(feed_dict=feed_training)
            gradient_cost_x_testing = gradient_cost_x[0].eval(feed_dict=feed_testing)
            #
            current_cost_training = cost.eval(feed_dict=feed_training)
            current_cost_testing = cost.eval(feed_dict=feed_testing)
            current_accuracy_training = accuracy.eval(feed_dict=feed_training)
            current_accuracy_testing = accuracy.eval(feed_dict=feed_testing)
            current_log_loss_training = log_loss.eval(feed_dict=feed_training)
            current_log_loss_testing = log_loss.eval(feed_dict=feed_testing)
            #
            log_loss_training.append(current_log_loss_training)
            log_loss_testing.append(current_log_loss_testing)
            cost_training.append(current_cost_training)
            cost_testing.append(current_cost_testing)
            accuracy_training.append(current_accuracy_training)
            accuracy_testing.append(current_accuracy_testing)

            # save
            params_values = {}
            hyper_params_values = {}
            output_values = {}

            output_values['u_training'] = u_training
            output_values['u_testing'] = u_testing
            output_values['prob_training'] = prob_training
            output_values['prob_testing'] = prob_testing

            output_values['cost_training'] = cost_training
            output_values['cost_testing'] = cost_testing
            output_values['accuracy_training'] = accuracy_training
            output_values['accuracy_testing'] = accuracy_testing
            output_values['log_loss_training'] = log_loss_training
            output_values['log_loss_testing'] = log_loss_testing

            output_values['gradient_u1_x_training'] = gradient_u1_x_training
            output_values['gradient_u1_x_testing'] = gradient_u1_x_testing
            output_values['gradient_prob1_x_training'] = gradient_prob1_x_training
            output_values['gradient_prob1_x_testing'] = gradient_prob1_x_testing
            output_values['gradient_cost_x_training'] = gradient_cost_x_training
            output_values['gradient_cost_x_testing'] = gradient_cost_x_testing

        if restore == False:
            for i in range(n_epoches):
                #
                X0_batch, X1_batch, X2_batch, X3_batch, X4_batch, Z_batch, Y_batch = obtain_mini_batch_dnn_alt_specific(
                    x0_training, x1_training, x2_training,
                    x3_training, x4_training, z_training, y_training, n_mini_batch=n_mini_batch)
                sess.run(training_op,
                         feed_dict={x0: X0_batch, x1: X1_batch, x2: X2_batch, x3: X3_batch, x4: X4_batch, z: Z_batch,
                                    y: Y_batch})
                #
                if i % 1000 == 0:
                    print("Iterations", i, "Cost = ", cost.eval(feed_dict=feed_training))
                if i % 20 == 0:
                    current_cost_training = cost.eval(feed_dict=feed_training)
                    current_cost_testing = cost.eval(feed_dict=feed_testing)
                    current_accuracy_training = accuracy.eval(feed_dict=feed_training)
                    current_accuracy_testing = accuracy.eval(feed_dict=feed_testing)
                    current_log_loss_training = log_loss.eval(feed_dict=feed_training)
                    current_log_loss_testing = log_loss.eval(feed_dict=feed_testing)
                    log_loss_training.append(current_log_loss_training)
                    log_loss_testing.append(current_log_loss_testing)
                    cost_training.append(current_cost_training)
                    cost_testing.append(current_cost_testing)
                    accuracy_training.append(current_accuracy_training)
                    accuracy_testing.append(current_accuracy_testing)

            # save the model
            saver.save(sess, "tmp/" + MODEL_NAME + ".ckpt")

            #
            u_training = u.eval(feed_dict=feed_training)
            u_testing = u.eval(feed_dict=feed_testing)
            prob_training = prob.eval(feed_dict=feed_training)
            prob_testing = prob.eval(feed_dict=feed_testing)
            #
            gradient_u1_x_training = gradient_u1_x[0].eval(feed_dict=feed_training)
            gradient_u1_x_testing = gradient_u1_x[0].eval(feed_dict=feed_testing)
            gradient_prob1_x_training = gradient_prob1_x[0].eval(feed_dict=feed_training)
            gradient_prob1_x_testing = gradient_prob1_x[0].eval(feed_dict=feed_testing)
            gradient_cost_x_training = gradient_cost_x[0].eval(feed_dict=feed_training)
            gradient_cost_x_testing = gradient_cost_x[0].eval(feed_dict=feed_testing)

            params_values = {}
            hyper_params_values = {}
            output_values = {}

            ''' evlauate the model by testing data'''
            print("Final Training Accuracy: ", accuracy.eval(feed_dict=feed_training))

            output_values['cost_training'] = cost_training
            output_values['cost_testing'] = cost_testing
            output_values['accuracy_training'] = accuracy_training
            output_values['accuracy_testing'] = accuracy_testing
            output_values['log_loss_training'] = log_loss_training
            output_values['log_loss_testing'] = log_loss_testing

            output_values['u_training'] = u_training
            output_values['u_testing'] = u_testing
            output_values['prob_training'] = prob_training
            output_values['prob_testing'] = prob_testing

            output_values['gradient_u1_x_training'] = gradient_u1_x_training
            output_values['gradient_u1_x_testing'] = gradient_u1_x_testing
            output_values['gradient_prob1_x_training'] = gradient_prob1_x_training
            output_values['gradient_prob1_x_testing'] = gradient_prob1_x_testing
            output_values['gradient_cost_x_training'] = gradient_cost_x_training
            output_values['gradient_cost_x_testing'] = gradient_cost_x_testing

    return params_values, hyper_params_values, output_values

def est_cm_resnet_simultaneous(data, penalty_const, MODEL_NAME, restore=False, n_epoches=10000, n_mini_batch=100, M=3,
                  n_hidden=100, simul_data=None, Dropout=False, Dropout_rate=0.01, BN=False, x_dim=15, K=2, D=7, current_best_acc = 0):
    #
    x0_training, x1_training, x2_training, x3_training, x4_training, z_training, y_training = data['training']
    x0_testing, x1_testing, x2_testing, x3_testing, x4_testing, z_testing, y_testing = data['testing']
    #
    N, D0 = x0_training.shape
    N, D1 = x1_training.shape
    N, D2 = x2_training.shape
    N, D3 = x3_training.shape
    N, D4 = x4_training.shape
    N, DZ = z_training.shape

    D_dic = {'d0': D0, 'd1': D1, 'd2': D2, 'd3': D3, 'd4': D4, 'dz': DZ}

    # model
    tf.reset_default_graph()

    x0 = tf.placeholder(dtype=tf.float32, shape=(None, D0), name='x0')
    x1 = tf.placeholder(dtype=tf.float32, shape=(None, D1), name='x1')
    x2 = tf.placeholder(dtype=tf.float32, shape=(None, D2), name='x2')
    x3 = tf.placeholder(dtype=tf.float32, shape=(None, D3), name='x3')
    x4 = tf.placeholder(dtype=tf.float32, shape=(None, D4), name='x4')
    z = tf.placeholder(dtype=tf.float32, shape=(None, DZ), name='z')
    y = tf.placeholder(dtype=tf.int64, shape=(None), name='y')
    x = tf.concat([x0, x1, x2, x3, x4, z], axis=1, name='x')

    hidden_x0 = x0
    hidden_x1 = x1
    hidden_x2 = x2
    hidden_x3 = x3
    hidden_x4 = x4
    hidden_z = z

    hidden_dic = {}
    hidden_dic['x0'] = hidden_x0
    hidden_dic['x1'] = hidden_x1
    hidden_dic['x2'] = hidden_x2
    hidden_dic['x3'] = hidden_x3
    hidden_dic['x4'] = hidden_x4
    hidden_dic['z'] = hidden_z

    # initialize parameters
    params_dic = {}
    for j in range(5):
        w_name = 'w' + str(j)
        b_name = 'b' + str(j)
        wz_name = 'wz' + str(j)
        d_name = 'd' + str(j)

        params_dic[w_name] = tf.Variable(tf.random_normal([D_dic[d_name], 1]), dtype = tf.float32, name = w_name)
        params_dic[b_name] = tf.Variable(tf.random_normal([1]), dtype = tf.float32, name = b_name)
        params_dic[wz_name] = tf.Variable(tf.random_normal([D_dic['dz'], 1]), dtype = tf.float32, name = wz_name)
        #
    output_dic = {}
    for j in range(5):
        layer_name = 'x' + str(j)
        w_name = 'w' + str(j)
        b_name = 'b' + str(j)
        wz_name = 'wz' + str(j)
        hidden_j = tf.add(tf.add(tf.matmul(hidden_dic[layer_name], params_dic[w_name]),
                                 tf.matmul(hidden_dic['z'], params_dic[wz_name])), params_dic[b_name])
        output_name = 'u' + str(j)
        output_dic[output_name] = hidden_j
    u_cm = tf.concat([output_dic['u0'], output_dic['u1'], output_dic['u2'], output_dic['u3'], output_dic['u4']], axis=1,
                     name='u')

    # train dnn part
    with tf.name_scope("dnn"):
        hidden = x
        for i in range(M):
            hidden = standard_hidden_layer(hidden, n_hidden, BN, Dropout, Dropout_rate)
        u_dnn = tf.layers.dense(hidden, K, name='output')

    # u and prob
    u = u_cm + u_dnn
    prob = tf.nn.softmax(u)

    # gradient info: only alternative 1 (bus) respect to inputs
    gradient_u1_x = [tf.concat(
        [tf.gradients(u[:, 1], [x0])[0], tf.gradients(u[:, 1], [x1])[0], tf.gradients(u[:, 1], [x2])[0],
         tf.gradients(u[:, 1], [x3])[0],
         tf.gradients(u[:, 1], [x4])[0], tf.gradients(u[:, 1], [z])[0]], axis=1)]

    gradient_prob1_x = [tf.concat(
        [tf.gradients(prob[:, 1], [x0])[0], tf.gradients(prob[:, 1], [x1])[0], tf.gradients(prob[:, 1], [x2])[0],
         tf.gradients(prob[:, 1], [x3])[0], tf.gradients(prob[:, 1], [x4])[0], tf.gradients(prob[:, 1], [z])[0]],
        axis=1)]

    # regularization l2
    l2_regularizer = tf.contrib.layers.l2_regularizer(scale=penalty_const, scope=None)
    vars_ = tf.trainable_variables()
    regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, vars_)

    # cost
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=u, labels=y), name='cost')
    cost += regularization_penalty
    log_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=u, labels=y), name='log_loss')
    optimizer = tf.train.AdamOptimizer()  # opt: better than simple gradient descents: always converge to stable numbers.
    #    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate) # opt objective
    training_op = optimizer.minimize(cost)  # minimize the opt objective

    #
    gradient_cost_x = [tf.concat([tf.gradients(cost, [x0])[0], tf.gradients(cost, [x1])[0], tf.gradients(cost, [x2])[0],
                                  tf.gradients(cost, [x3])[0], tf.gradients(cost, [x4])[0], tf.gradients(cost, [z])[0]],
                                 axis=1)]

    # eval params
    pred_y = tf.argmax(prob,1)

    correct = tf.nn.in_top_k(u, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    cost_training = []
    cost_testing = []
    accuracy_training = []
    accuracy_testing = []
    log_loss_training = []
    log_loss_testing = []
    # f1_score_training = []
    # f1_score_testing = []
    with tf.Session() as sess:
        # always run this to train the model
        init.run()
        feed_training = {x0: x0_training, x1: x1_training, x2: x2_training, x3: x3_training, x4: x4_training,
                         z: z_training, y: y_training}
        feed_testing = {x0: x0_testing, x1: x1_testing, x2: x2_testing, x3: x3_testing, x4: x4_testing, z: z_testing,
                        y: y_testing}

        if restore == True:
            saver.restore(sess, "tmp/" + MODEL_NAME + ".ckpt")
            u_training = u.eval(feed_dict=feed_training)
            u_testing = u.eval(feed_dict=feed_testing)
            prob_training = prob.eval(feed_dict=feed_training)
            prob_testing = prob.eval(feed_dict=feed_testing)
            #
            gradient_u1_x_training = gradient_u1_x[0].eval(feed_dict=feed_training)
            gradient_u1_x_testing = gradient_u1_x[0].eval(feed_dict=feed_testing)
            gradient_prob1_x_training = gradient_prob1_x[0].eval(feed_dict=feed_training)
            gradient_prob1_x_testing = gradient_prob1_x[0].eval(feed_dict=feed_testing)
            gradient_cost_x_training = gradient_cost_x[0].eval(feed_dict=feed_training)
            gradient_cost_x_testing = gradient_cost_x[0].eval(feed_dict=feed_testing)
            #
            current_cost_training = cost.eval(feed_dict=feed_training)
            current_cost_testing = cost.eval(feed_dict=feed_testing)
            current_accuracy_training = accuracy.eval(feed_dict=feed_training)
            current_accuracy_testing = accuracy.eval(feed_dict=feed_testing)
            current_log_loss_training = log_loss.eval(feed_dict=feed_training)
            current_log_loss_testing = log_loss.eval(feed_dict=feed_testing)
            #
            log_loss_training.append(current_log_loss_training)
            log_loss_testing.append(current_log_loss_testing)
            cost_training.append(current_cost_training)
            cost_testing.append(current_cost_testing)
            accuracy_training.append(current_accuracy_training)
            accuracy_testing.append(current_accuracy_testing)

            # save
            params_values = {}
            hyper_params_values = {}
            output_values = {}

            output_values['u_training'] = u_training
            output_values['u_testing'] = u_testing
            output_values['prob_training'] = prob_training
            output_values['prob_testing'] = prob_testing

            output_values['cost_training'] = cost_training
            output_values['cost_testing'] = cost_testing
            output_values['accuracy_training'] = accuracy_training
            output_values['accuracy_testing'] = accuracy_testing
            output_values['log_loss_training'] = log_loss_training
            output_values['log_loss_testing'] = log_loss_testing

            output_values['gradient_u1_x_training'] = gradient_u1_x_training
            output_values['gradient_u1_x_testing'] = gradient_u1_x_testing
            output_values['gradient_prob1_x_training'] = gradient_prob1_x_training
            output_values['gradient_prob1_x_testing'] = gradient_prob1_x_testing
            output_values['gradient_cost_x_training'] = gradient_cost_x_training
            output_values['gradient_cost_x_testing'] = gradient_cost_x_testing

        if restore == False:
            for i in range(n_epoches):
                #
                X0_batch, X1_batch, X2_batch, X3_batch, X4_batch, Z_batch, Y_batch = obtain_mini_batch_dnn_alt_specific(
                    x0_training, x1_training, x2_training,
                    x3_training, x4_training, z_training, y_training, n_mini_batch=n_mini_batch)
                sess.run(training_op,
                         feed_dict={x0: X0_batch, x1: X1_batch, x2: X2_batch, x3: X3_batch, x4: X4_batch, z: Z_batch,
                                    y: Y_batch})
                #
                if i % 1000 == 0:
                    print("Iterations", i, "Cost = ", cost.eval(feed_dict=feed_training))
                if i % 20 == 0:
                    current_cost_training = cost.eval(feed_dict=feed_training)
                    current_cost_testing = cost.eval(feed_dict=feed_testing)
                    current_accuracy_training = accuracy.eval(feed_dict=feed_training)
                    current_accuracy_testing = accuracy.eval(feed_dict=feed_testing)
                    current_log_loss_training = log_loss.eval(feed_dict=feed_training)
                    current_log_loss_testing = log_loss.eval(feed_dict=feed_testing)
                    log_loss_training.append(current_log_loss_training)
                    log_loss_testing.append(current_log_loss_testing)
                    cost_training.append(current_cost_training)
                    cost_testing.append(current_cost_testing)
                    accuracy_training.append(current_accuracy_training)
                    accuracy_testing.append(current_accuracy_testing)

            # save the model
            if accuracy_testing[-1] >= current_best_acc:
                saver.save(sess, "tmp/"+MODEL_NAME+".ckpt")
                current_best_acc = accuracy_testing[-1]
            #
            u_training = u.eval(feed_dict=feed_training)
            u_testing = u.eval(feed_dict=feed_testing)
            prob_training = prob.eval(feed_dict=feed_training)
            prob_testing = prob.eval(feed_dict=feed_testing)
            #
            gradient_u1_x_training = gradient_u1_x[0].eval(feed_dict=feed_training)
            gradient_u1_x_testing = gradient_u1_x[0].eval(feed_dict=feed_testing)
            gradient_prob1_x_training = gradient_prob1_x[0].eval(feed_dict=feed_training)
            gradient_prob1_x_testing = gradient_prob1_x[0].eval(feed_dict=feed_testing)
            gradient_cost_x_training = gradient_cost_x[0].eval(feed_dict=feed_training)
            gradient_cost_x_testing = gradient_cost_x[0].eval(feed_dict=feed_testing)

            params_values = {}
            hyper_params_values = {}
            output_values = {}

            ''' evlauate the model by testing data'''
            print("Final Training Accuracy: ", accuracy.eval(feed_dict=feed_training))

            output_values['cost_training'] = cost_training
            output_values['cost_testing'] = cost_testing
            output_values['accuracy_training'] = accuracy_training
            output_values['accuracy_testing'] = accuracy_testing
            output_values['log_loss_training'] = log_loss_training
            output_values['log_loss_testing'] = log_loss_testing

            output_values['u_training'] = u_training
            output_values['u_testing'] = u_testing
            output_values['prob_training'] = prob_training
            output_values['prob_testing'] = prob_testing

            output_values['gradient_u1_x_training'] = gradient_u1_x_training
            output_values['gradient_u1_x_testing'] = gradient_u1_x_testing
            output_values['gradient_prob1_x_training'] = gradient_prob1_x_training
            output_values['gradient_prob1_x_testing'] = gradient_prob1_x_testing
            output_values['gradient_cost_x_training'] = gradient_cost_x_training
            output_values['gradient_cost_x_testing'] = gradient_cost_x_testing

            output_values['current_best_acc'] = current_best_acc

    return params_values, hyper_params_values, output_values




def est_cm_resnet_simultaneous_use_delta(data, penalty_const, MODEL_NAME, restore=False, n_epoches=10000, n_mini_batch=100, M=3,
                                         n_hidden=100, simul_data=None, Dropout=False, Dropout_rate=0.01, l2_regu = 1e-50,
                                         BN=False, x_dim=15, K=2, D=7, current_best_acc = 0):
    #
    x0_training, x1_training, x2_training, x3_training, x4_training, z_training, y_training = data['training']
    x0_testing, x1_testing, x2_testing, x3_testing, x4_testing, z_testing, y_testing = data['testing']
    #
    N, D0 = x0_training.shape
    N, D1 = x1_training.shape
    N, D2 = x2_training.shape
    N, D3 = x3_training.shape
    N, D4 = x4_training.shape
    N, DZ = z_training.shape

    D_dic = {'d0': D0, 'd1': D1, 'd2': D2, 'd3': D3, 'd4': D4, 'dz': DZ}

    # model
    tf.reset_default_graph()

    x0 = tf.placeholder(dtype=tf.float32, shape=(None, D0), name='x0')
    x1 = tf.placeholder(dtype=tf.float32, shape=(None, D1), name='x1')
    x2 = tf.placeholder(dtype=tf.float32, shape=(None, D2), name='x2')
    x3 = tf.placeholder(dtype=tf.float32, shape=(None, D3), name='x3')
    x4 = tf.placeholder(dtype=tf.float32, shape=(None, D4), name='x4')
    z = tf.placeholder(dtype=tf.float32, shape=(None, DZ), name='z')
    y = tf.placeholder(dtype=tf.int64, shape=(None), name='y')
    x = tf.concat([x0, x1, x2, x3, x4, z], axis=1, name='x')

    hidden_x0 = x0
    hidden_x1 = x1
    hidden_x2 = x2
    hidden_x3 = x3
    hidden_x4 = x4
    hidden_z = z

    hidden_dic = {}
    hidden_dic['x0'] = hidden_x0
    hidden_dic['x1'] = hidden_x1
    hidden_dic['x2'] = hidden_x2
    hidden_dic['x3'] = hidden_x3
    hidden_dic['x4'] = hidden_x4
    hidden_dic['z'] = hidden_z

    # initialize parameters
    params_dic = {}
    for j in range(5):
        w_name = 'w' + str(j)
        b_name = 'b' + str(j)
        wz_name = 'wz' + str(j)
        d_name = 'd' + str(j)

        params_dic[w_name] = tf.Variable(tf.random_normal([D_dic[d_name], 1]), dtype = tf.float32, name = w_name)
        params_dic[b_name] = tf.Variable(tf.random_normal([1]), dtype = tf.float32, name = b_name)
        params_dic[wz_name] = tf.Variable(tf.random_normal([D_dic['dz'], 1]), dtype = tf.float32, name = wz_name)
        #
    output_dic = {}
    for j in range(5):
        layer_name = 'x' + str(j)
        w_name = 'w' + str(j)
        b_name = 'b' + str(j)
        wz_name = 'wz' + str(j)
        hidden_j = tf.add(tf.add(tf.matmul(hidden_dic[layer_name], params_dic[w_name]),
                                 tf.matmul(hidden_dic['z'], params_dic[wz_name])), params_dic[b_name])
        output_name = 'u' + str(j)
        output_dic[output_name] = hidden_j
    u_cm = tf.concat([output_dic['u0'], output_dic['u1'], output_dic['u2'], output_dic['u3'], output_dic['u4']], axis=1,
                     name='u')

    # train dnn part
    with tf.name_scope("dnn"):
        hidden = x
        for i in range(M):
            hidden = standard_hidden_layer(hidden, n_hidden, BN, Dropout, Dropout_rate)
        u_dnn = tf.layers.dense(hidden, K, name='output')

    # u and prob
    u = (1 - penalty_const) * u_cm + penalty_const * u_dnn
    prob = tf.nn.softmax(u)

    # gradient info: only alternative 1 (bus) respect to inputs
    gradient_u1_x = [tf.concat(
        [tf.gradients(u[:, 1], [x0])[0], tf.gradients(u[:, 1], [x1])[0], tf.gradients(u[:, 1], [x2])[0],
         tf.gradients(u[:, 1], [x3])[0],
         tf.gradients(u[:, 1], [x4])[0], tf.gradients(u[:, 1], [z])[0]], axis=1)]

    gradient_prob1_x = [tf.concat(
        [tf.gradients(prob[:, 1], [x0])[0], tf.gradients(prob[:, 1], [x1])[0], tf.gradients(prob[:, 1], [x2])[0],
         tf.gradients(prob[:, 1], [x3])[0], tf.gradients(prob[:, 1], [x4])[0], tf.gradients(prob[:, 1], [z])[0]],
        axis=1)]

    # regularization l2
    l2_regularizer = tf.contrib.layers.l2_regularizer(scale=l2_regu, scope=None)
    vars_ = tf.trainable_variables()
    regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, vars_)

    # cost
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=u, labels=y), name='cost')
    cost += regularization_penalty
    log_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=u, labels=y), name='log_loss')
    optimizer = tf.train.AdamOptimizer()  # opt: better than simple gradient descents: always converge to stable numbers.
    #    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate) # opt objective
    training_op = optimizer.minimize(cost)  # minimize the opt objective

    #
    gradient_cost_x = [tf.concat([tf.gradients(cost, [x0])[0], tf.gradients(cost, [x1])[0], tf.gradients(cost, [x2])[0],
                                  tf.gradients(cost, [x3])[0], tf.gradients(cost, [x4])[0], tf.gradients(cost, [z])[0]],
                                 axis=1)]

    # eval params
    pred_y = tf.argmax(prob,1)

    correct = tf.nn.in_top_k(u, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    cost_training = []
    cost_testing = []
    accuracy_training = []
    accuracy_testing = []
    log_loss_training = []
    log_loss_testing = []
    # f1_score_training = []
    # f1_score_testing = []
    with tf.Session() as sess:
        # always run this to train the model
        init.run()
        feed_training = {x0: x0_training, x1: x1_training, x2: x2_training, x3: x3_training, x4: x4_training,
                         z: z_training, y: y_training}
        feed_testing = {x0: x0_testing, x1: x1_testing, x2: x2_testing, x3: x3_testing, x4: x4_testing, z: z_testing,
                        y: y_testing}

        if restore == True:
            saver.restore(sess, "tmp/" + MODEL_NAME + ".ckpt")
            u_training = u.eval(feed_dict=feed_training)
            u_testing = u.eval(feed_dict=feed_testing)
            prob_training = prob.eval(feed_dict=feed_training)
            prob_testing = prob.eval(feed_dict=feed_testing)
            #
            gradient_u1_x_training = gradient_u1_x[0].eval(feed_dict=feed_training)
            gradient_u1_x_testing = gradient_u1_x[0].eval(feed_dict=feed_testing)
            gradient_prob1_x_training = gradient_prob1_x[0].eval(feed_dict=feed_training)
            gradient_prob1_x_testing = gradient_prob1_x[0].eval(feed_dict=feed_testing)
            gradient_cost_x_training = gradient_cost_x[0].eval(feed_dict=feed_training)
            gradient_cost_x_testing = gradient_cost_x[0].eval(feed_dict=feed_testing)
            #
            current_cost_training = cost.eval(feed_dict=feed_training)
            current_cost_testing = cost.eval(feed_dict=feed_testing)
            current_accuracy_training = accuracy.eval(feed_dict=feed_training)
            current_accuracy_testing = accuracy.eval(feed_dict=feed_testing)
            current_log_loss_training = log_loss.eval(feed_dict=feed_training)
            current_log_loss_testing = log_loss.eval(feed_dict=feed_testing)
            #
            log_loss_training.append(current_log_loss_training)
            log_loss_testing.append(current_log_loss_testing)
            cost_training.append(current_cost_training)
            cost_testing.append(current_cost_testing)
            accuracy_training.append(current_accuracy_training)
            accuracy_testing.append(current_accuracy_testing)

            # save
            params_values = {}
            hyper_params_values = {}
            output_values = {}

            output_values['u_training'] = u_training
            output_values['u_testing'] = u_testing
            output_values['prob_training'] = prob_training
            output_values['prob_testing'] = prob_testing

            output_values['cost_training'] = cost_training
            output_values['cost_testing'] = cost_testing
            output_values['accuracy_training'] = accuracy_training
            output_values['accuracy_testing'] = accuracy_testing
            output_values['log_loss_training'] = log_loss_training
            output_values['log_loss_testing'] = log_loss_testing

            output_values['gradient_u1_x_training'] = gradient_u1_x_training
            output_values['gradient_u1_x_testing'] = gradient_u1_x_testing
            output_values['gradient_prob1_x_training'] = gradient_prob1_x_training
            output_values['gradient_prob1_x_testing'] = gradient_prob1_x_testing
            output_values['gradient_cost_x_training'] = gradient_cost_x_training
            output_values['gradient_cost_x_testing'] = gradient_cost_x_testing

        if restore == False:
            for i in range(n_epoches):
                #
                X0_batch, X1_batch, X2_batch, X3_batch, X4_batch, Z_batch, Y_batch = obtain_mini_batch_dnn_alt_specific(
                    x0_training, x1_training, x2_training,
                    x3_training, x4_training, z_training, y_training, n_mini_batch=n_mini_batch)
                sess.run(training_op,
                         feed_dict={x0: X0_batch, x1: X1_batch, x2: X2_batch, x3: X3_batch, x4: X4_batch, z: Z_batch,
                                    y: Y_batch})
                #
                if i % 1000 == 0:
                    print("Iterations", i, "Cost = ", cost.eval(feed_dict=feed_training))
                if i % 20 == 0:
                    current_cost_training = cost.eval(feed_dict=feed_training)
                    current_cost_testing = cost.eval(feed_dict=feed_testing)
                    current_accuracy_training = accuracy.eval(feed_dict=feed_training)
                    current_accuracy_testing = accuracy.eval(feed_dict=feed_testing)
                    current_log_loss_training = log_loss.eval(feed_dict=feed_training)
                    current_log_loss_testing = log_loss.eval(feed_dict=feed_testing)
                    log_loss_training.append(current_log_loss_training)
                    log_loss_testing.append(current_log_loss_testing)
                    cost_training.append(current_cost_training)
                    cost_testing.append(current_cost_testing)
                    accuracy_training.append(current_accuracy_training)
                    accuracy_testing.append(current_accuracy_testing)

            # save the model
            if accuracy_testing[-1] >= current_best_acc:
                saver.save(sess, "tmp/"+MODEL_NAME+".ckpt")
                current_best_acc = accuracy_testing[-1]
            #
            u_training = u.eval(feed_dict=feed_training)
            u_testing = u.eval(feed_dict=feed_testing)
            prob_training = prob.eval(feed_dict=feed_training)
            prob_testing = prob.eval(feed_dict=feed_testing)
            #
            gradient_u1_x_training = gradient_u1_x[0].eval(feed_dict=feed_training)
            gradient_u1_x_testing = gradient_u1_x[0].eval(feed_dict=feed_testing)
            gradient_prob1_x_training = gradient_prob1_x[0].eval(feed_dict=feed_training)
            gradient_prob1_x_testing = gradient_prob1_x[0].eval(feed_dict=feed_testing)
            gradient_cost_x_training = gradient_cost_x[0].eval(feed_dict=feed_training)
            gradient_cost_x_testing = gradient_cost_x[0].eval(feed_dict=feed_testing)

            params_values = {}
            hyper_params_values = {}
            output_values = {}

            ''' evlauate the model by testing data'''
            print("Final Training Accuracy: ", accuracy.eval(feed_dict=feed_training))

            output_values['cost_training'] = cost_training
            output_values['cost_testing'] = cost_testing
            output_values['accuracy_training'] = accuracy_training
            output_values['accuracy_testing'] = accuracy_testing
            output_values['log_loss_training'] = log_loss_training
            output_values['log_loss_testing'] = log_loss_testing

            output_values['u_training'] = u_training
            output_values['u_testing'] = u_testing
            output_values['prob_training'] = prob_training
            output_values['prob_testing'] = prob_testing

            output_values['gradient_u1_x_training'] = gradient_u1_x_training
            output_values['gradient_u1_x_testing'] = gradient_u1_x_testing
            output_values['gradient_prob1_x_training'] = gradient_prob1_x_training
            output_values['gradient_prob1_x_testing'] = gradient_prob1_x_testing
            output_values['gradient_cost_x_training'] = gradient_cost_x_training
            output_values['gradient_cost_x_testing'] = gradient_cost_x_testing

            output_values['current_best_acc'] = current_best_acc

    return params_values, hyper_params_values, output_values



def obtain_mini_batch_pt(x0_training,x1_training,p0_training,p1_training,z_training,y_training,n_mini_batch):
    '''
    Return mini_batch
    '''
    N = x0_training.shape[0]
    index = np.random.choice(N, size = n_mini_batch)
    x0_training_batch=x0_training[index, :]
    x1_training_batch=x1_training[index, :]
    p0_training_batch=p0_training[index, :]
    p1_training_batch=p1_training[index, :]
    z_training_batch=z_training[index, :]
    y_training_batch=y_training[index]
    return x0_training_batch,x1_training_batch,p0_training_batch,p1_training_batch,z_training_batch,y_training_batch


def est_pt(data, MODEL_NAME, restore = False, n_epoches = 10000, current_best_acc = 0):
    '''
    estimate pt model... 
    '''
    
    x0_training,x1_training,p0_training,p1_training,z_training,y_training = data['training']
    x0_testing,x1_testing,p0_testing,p1_testing,z_testing,y_testing = data['testing']
    #
    Dz = z_training.shape[1]    
    # build models
    tf.reset_default_graph()
    z = tf.placeholder(dtype = tf.float32, shape = (None, Dz), name = 'z')
    x0 = tf.placeholder(dtype = tf.float32, shape = (None, 2), name = 'x0')
    x1 = tf.placeholder(dtype = tf.float32, shape = (None, 2), name = 'x1')
    p0 = tf.placeholder(dtype = tf.float32, shape = (None, 2), name = 'p0')
    p1 = tf.placeholder(dtype = tf.float32, shape = (None, 2), name = 'p1')
    y = tf.placeholder(dtype = tf.int64, shape = (None), name = 'y')

    gamma_param_0 = tf.Variable(tf.random_normal([1]), dtype = tf.float32)
    beta_param_0 = tf.Variable(tf.random_normal([1]), dtype = tf.float32)
    lambda_param_0 = tf.Variable(tf.random_normal([1]), dtype = tf.float32)
    ref_param_0 = tf.Variable(tf.random_normal([1]), dtype = tf.float32)
    gamma_param = tf.Variable(tf.random_normal([Dz, 1]), dtype = tf.float32)
    beta_param = tf.Variable(tf.random_normal([Dz, 1]), dtype = tf.float32)
    lambda_param = tf.Variable(tf.random_normal([Dz, 1]), dtype = tf.float32)
    ref_param = tf.Variable(tf.random_normal([Dz, 1]), dtype = tf.float32)
    
    gamma_initial = tf.add(tf.matmul(z, gamma_param), gamma_param_0)
    beta_initial = tf.add(tf.matmul(z, beta_param), beta_param_0)
    lambda_initial = tf.add(tf.matmul(z, lambda_param), lambda_param_0)
    ref_initial = tf.add(tf.matmul(z, ref_param), ref_param_0)

    gamma = tf.nn.sigmoid(gamma_initial)
    beta = tf.nn.sigmoid(beta_initial)
    lambda_ = tf.nn.sigmoid(lambda_initial)*4 + 1 # hence constrain lambda_ between 0 and 5
    ref = ref_initial*1.0

    # update x1 and x0 by ref pt
    x1_ref_dep = tf.subtract(x1, ref)
    x0_ref_dep = tf.subtract(x0, ref)    
    # create zeros
    zeros = tf.zeros_like(x0, dtype = tf.float32)
    pi_1 = tf.exp(- tf.pow((-tf.log(p1)), beta))
    pi_0 = tf.exp(- tf.pow((-tf.log(p0)), beta))
    # compute u_0
    # u_0 postive
    v_0_pos = tf.pow(tf.where(x0_ref_dep>0, x0_ref_dep, zeros), gamma)
    u_0_pos = tf.multiply(v_0_pos, pi_0)
    # u_0 negative        
    v_0_neg = tf.multiply(-lambda_, tf.pow(-tf.where(x0_ref_dep<0, x0_ref_dep, zeros), gamma))
    u_0_neg = tf.multiply(v_0_neg, pi_0)    
    u_0 = tf.reduce_sum(tf.add(u_0_pos, u_0_neg), axis = 1, keepdims=True)
    # compute u_1
    # u_1 postive
    v_1_pos = tf.pow(tf.where(x1_ref_dep>0, x1_ref_dep, zeros), gamma)
    u_1_pos = tf.multiply(v_1_pos, pi_1)
    # u_1 negative
    v_1_neg = tf.multiply(-lambda_, tf.pow(-tf.where(x1_ref_dep<0, x1_ref_dep, zeros), gamma))
    u_1_neg = tf.multiply(v_1_neg, pi_1)
    u_1 = tf.reduce_sum(tf.add(u_1_pos, u_1_neg), axis = 1, keepdims=True)
    #     
    u = tf.concat([u_0, u_1], axis = 1)
    prob = tf.nn.softmax(u)

    # gradient info: only alternative 1 respect to inputs
    gradient_u1_x = [tf.concat([tf.gradients(u[:,1], [x0])[0], tf.gradients(u[:,1], [x1])[0], tf.gradients(u[:,1], [p0])[0], tf.gradients(u[:,1], [p1])[0],  tf.gradients(u[:,1], [z])[0]], axis = 1)]
    gradient_prob1_x = [tf.concat([tf.gradients(prob[:,1], [x0])[0], tf.gradients(prob[:,1], [x1])[0], tf.gradients(prob[:,1], [p0])[0], tf.gradients(prob[:,1], [p1])[0],  tf.gradients(prob[:,1], [z])[0]], axis = 1)]

    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = u, labels = y), name = 'cost')
    log_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = u, labels = y), name = 'log_loss')
    optimizer = tf.train.AdamOptimizer() # opt: better than simple gradient descents: always converge to stable numbers.
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate) # opt objective
    training_op = optimizer.minimize(cost) # minimize the opt objective

    # 
    gradient_cost_x = [tf.concat([tf.gradients(cost, [x0])[0], tf.gradients(cost, [x1])[0], tf.gradients(cost, [p0])[0], tf.gradients(cost, [p1])[0], tf.gradients(cost, [z])[0]], axis = 1)]
    
    # eval params
    correct = tf.nn.in_top_k(u, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    cost_training=[]
    cost_testing=[]
    accuracy_training=[]
    accuracy_testing=[]
    log_loss_training=[]
    log_loss_testing=[]
    
    with tf.Session() as sess:
        # always run this to train the model
        init.run()
        feed_training = {x0: x0_training, x1: x1_training, p0:p0_training, p1: p1_training, z: z_training, y:y_training}
        feed_testing = {x0: x0_testing, x1: x1_testing, p0:p0_testing, p1: p1_testing, z: z_testing, y:y_testing}

        if restore == True:
            # case 1. restore models and evaluate key values
            saver.restore(sess, "tmp/"+MODEL_NAME+".ckpt")
            u_training = u.eval(feed_dict = feed_training)
            u_testing = u.eval(feed_dict = feed_testing)
            prob_training = prob.eval(feed_dict = feed_training)
            prob_testing = prob.eval(feed_dict = feed_testing)
            # 
            current_cost_training = cost.eval(feed_dict=feed_training)
            current_cost_testing = cost.eval(feed_dict=feed_testing)
            current_accuracy_training = accuracy.eval(feed_dict=feed_training)
            current_accuracy_testing = accuracy.eval(feed_dict=feed_testing)
            current_log_loss_training = log_loss.eval(feed_dict = feed_training)
            current_log_loss_testing = log_loss.eval(feed_dict = feed_testing)
            #
            log_loss_training.append(current_log_loss_training)
            log_loss_testing.append(current_log_loss_testing)
            cost_training.append(current_cost_training)
            cost_testing.append(current_cost_testing)
            accuracy_training.append(current_accuracy_training)
            accuracy_testing.append(current_accuracy_testing)
            #            
            gradient_u1_x_training=gradient_u1_x[0].eval(feed_dict = feed_training)
            gradient_u1_x_testing=gradient_u1_x[0].eval(feed_dict = feed_testing)
            gradient_prob1_x_training=gradient_prob1_x[0].eval(feed_dict = feed_training)
            gradient_prob1_x_testing=gradient_prob1_x[0].eval(feed_dict = feed_testing)
            gradient_cost_x_training=gradient_cost_x[0].eval(feed_dict = feed_training)          
            gradient_cost_x_testing=gradient_cost_x[0].eval(feed_dict = feed_testing)

            # save
            params_dic = {}
            hyper_params_dic = {}
            output_dic = {}
            output_dic['u_training'] = u_training
            output_dic['u_testing'] = u_testing
            output_dic['prob_training'] = prob_training
            output_dic['prob_testing'] = prob_testing

            output_dic['cost_training'] = cost_training
            output_dic['cost_testing'] = cost_testing
            output_dic['accuracy_training'] = accuracy_training
            output_dic['accuracy_testing'] = accuracy_testing
            output_dic['log_loss_training'] = log_loss_training
            output_dic['log_loss_testing'] = log_loss_testing
            
            output_dic['gradient_u1_x_training'] = gradient_u1_x_training
            output_dic['gradient_u1_x_testing'] = gradient_u1_x_testing
            output_dic['gradient_prob1_x_training'] = gradient_prob1_x_training
            output_dic['gradient_prob1_x_testing'] = gradient_prob1_x_testing
            output_dic['gradient_cost_x_training'] = gradient_cost_x_training
            output_dic['gradient_cost_x_testing'] = gradient_cost_x_testing

        elif restore == False:
            # case 3. train and save the estimated model
            for i in range(n_epoches):
                x0_training_batch,x1_training_batch,p0_training_batch,p1_training_batch,z_training_batch,y_training_batch = \
                     obtain_mini_batch_pt(x0_training,x1_training,p0_training,p1_training,z_training,y_training,n_mini_batch=100)               
                
                sess.run(training_op, feed_dict = {x0: x0_training_batch, x1: x1_training_batch, p0:p0_training_batch,
                    p1:p1_training_batch, z:z_training_batch, y:y_training_batch})
                if i % 1000 == 0:
                    print("Iterations", i, "Cost = ", cost.eval(feed_dict = feed_training))
                if i % 20 == 0:
                    current_cost_training = cost.eval(feed_dict=feed_training)
                    current_cost_testing = cost.eval(feed_dict=feed_testing)
                    current_accuracy_training = accuracy.eval(feed_dict=feed_training)
                    current_accuracy_testing = accuracy.eval(feed_dict=feed_testing)
                    current_log_loss_training=log_loss.eval(feed_dict=feed_training)
                    current_log_loss_testing=log_loss.eval(feed_dict=feed_testing)

                    cost_training.append(current_cost_training)
                    cost_testing.append(current_cost_testing)
                    accuracy_training.append(current_accuracy_training)
                    accuracy_testing.append(current_accuracy_testing)
                    log_loss_training.append(current_log_loss_training)
                    log_loss_testing.append(current_log_loss_testing)
    
            # save the model
            if accuracy_testing[-1] >= current_best_acc:
                saver.save(sess, "tmp/"+MODEL_NAME+".ckpt")
                current_best_acc = accuracy_testing[-1]
            #                
            u_training = u.eval(feed_dict = feed_training)
            u_testing = u.eval(feed_dict = feed_testing)
            prob_training = prob.eval(feed_dict = feed_training)
            prob_testing = prob.eval(feed_dict = feed_testing)
            # 
            gradient_u1_x_training=gradient_u1_x[0].eval(feed_dict = feed_training)
            gradient_u1_x_testing=gradient_u1_x[0].eval(feed_dict = feed_testing)
            gradient_prob1_x_training=gradient_prob1_x[0].eval(feed_dict = feed_training)
            gradient_prob1_x_testing=gradient_prob1_x[0].eval(feed_dict = feed_testing)
            gradient_cost_x_training=gradient_cost_x[0].eval(feed_dict = feed_training)            
            gradient_cost_x_testing=gradient_cost_x[0].eval(feed_dict = feed_testing)            
            # 
            gamma_param_0_value = gamma_param_0.eval()
            beta_param_0_value = beta_param_0.eval()
            lambda_param_0_value = lambda_param_0.eval()
            ref_param_0_value = ref_param_0.eval()
            gamma_param_value = gamma_param.eval()
            beta_param_value = beta_param.eval()
            lambda_param_value = lambda_param.eval()
            ref_param_value = ref_param.eval()
            # 
            gamma_value = gamma.eval(feed_dict = feed_training)
            beta_value = beta.eval(feed_dict = feed_training)
            lambda_value = lambda_.eval(feed_dict = feed_training)
            ref_value = ref.eval(feed_dict = feed_training)

            # outputs
            params_dic = {}
            hyper_params_dic = {}
            output_dic = {}
            
            output_dic['cost_training'] = cost_training
            output_dic['cost_testing'] = cost_testing
            output_dic['accuracy_training'] = accuracy_training
            output_dic['accuracy_testing'] = accuracy_testing
            output_dic['log_loss_training'] = log_loss_training
            output_dic['log_loss_testing'] = log_loss_testing

            output_dic['u_training'] = u_training
            output_dic['u_testing'] = u_testing
            output_dic['prob_training'] = prob_training
            output_dic['prob_testing'] = prob_testing
            
            output_dic['gradient_u1_x_training'] = gradient_u1_x_training
            output_dic['gradient_u1_x_testing'] = gradient_u1_x_testing
            output_dic['gradient_prob1_x_training'] = gradient_prob1_x_training
            output_dic['gradient_prob1_x_testing'] = gradient_prob1_x_testing
            output_dic['gradient_cost_x_training'] = gradient_cost_x_training
            output_dic['gradient_cost_x_testing'] = gradient_cost_x_testing

            output_dic['current_best_acc'] = current_best_acc

            params_dic['gamma']=gamma_value
            params_dic['beta']=beta_value
            params_dic['lambda']=lambda_value
            params_dic['ref']=ref_value
            params_dic['gamma_param_0']=gamma_param_0_value
            params_dic['beta_param_0']=beta_param_0_value
            params_dic['lambda_param_0']=lambda_param_0_value
            params_dic['ref_param_0']=ref_param_0_value
            params_dic['gamma_param']=gamma_param_value
            params_dic['beta_param']=beta_param_value
            params_dic['lambda_param']=lambda_param_value
            params_dic['ref_param']=ref_param_value

    return params_dic,hyper_params_dic,output_dic
    

def est_pt_resnet(data, pt_param_dic, penalty_const, MODEL_NAME, restore = False, n_epoches = 10000,
                  n_mini_batch=100, M=3, n_hidden=100, simul_data = None,Dropout = False, Dropout_rate = 0.01,
                  BN = False, x_dim = 15, K = 2, D = 7, current_best_acc = 0):
    '''
    estimate pt resnet model
    '''
    x0_training,x1_training,p0_training,p1_training,z_training,y_training = data['training']
    x0_testing,x1_testing,p0_testing,p1_testing,z_testing,y_testing = data['testing']
    #
    Dz = z_training.shape[1]
    # build models
    tf.reset_default_graph()
    z = tf.placeholder(dtype = tf.float32, shape = (None, Dz), name = 'z')
    x0 = tf.placeholder(dtype = tf.float32, shape = (None, 2), name = 'x0')
    x1 = tf.placeholder(dtype = tf.float32, shape = (None, 2), name = 'x1')
    p0 = tf.placeholder(dtype = tf.float32, shape = (None, 2), name = 'p0')
    p1 = tf.placeholder(dtype = tf.float32, shape = (None, 2), name = 'p1')
    x = tf.concat([x0, x1, p0, p1, z], axis = 1, name = 'x')
    y = tf.placeholder(dtype = tf.int64, shape = (None), name = 'y')

    gamma_param_0 = pt_param_dic['gamma_param_0']
    beta_param_0 = pt_param_dic['beta_param_0']
    lambda_param_0 = pt_param_dic['lambda_param_0']
    ref_param_0 = pt_param_dic['ref_param_0']
    gamma_param = pt_param_dic['gamma_param']
    beta_param = pt_param_dic['beta_param']
    lambda_param = pt_param_dic['lambda_param']
    ref_param = pt_param_dic['ref_param']
    
    gamma_initial = tf.add(tf.matmul(z, gamma_param), gamma_param_0)
    beta_initial = tf.add(tf.matmul(z, beta_param), beta_param_0)
    lambda_initial = tf.add(tf.matmul(z, lambda_param), lambda_param_0)
    ref_initial = tf.add(tf.matmul(z, ref_param), ref_param_0)

    gamma = tf.nn.sigmoid(gamma_initial)
    beta = tf.nn.sigmoid(beta_initial)
    lambda_ = tf.nn.sigmoid(lambda_initial)*4 + 1 # hence constrain lambda_ between 0 and 5
    ref = ref_initial*1.0

    ### PT part
    # update x1 and x0 by ref pt
    x1_ref_dep = tf.subtract(x1, ref)
    x0_ref_dep = tf.subtract(x0, ref)    
    # create zeros
    zeros = tf.zeros_like(x0, dtype = tf.float32)
    pi_1 = tf.exp(- tf.pow((-tf.log(p1)), beta))
    pi_0 = tf.exp(- tf.pow((-tf.log(p0)), beta))
    # compute u_0
    # u_0 postive
    v_0_pos = tf.pow(tf.where(x0_ref_dep>0, x0_ref_dep, zeros), gamma)
    u_0_pos = tf.multiply(v_0_pos, pi_0)
    # u_0 negative        
    v_0_neg = tf.multiply(-lambda_, tf.pow(-tf.where(x0_ref_dep<0, x0_ref_dep, zeros), gamma))
    u_0_neg = tf.multiply(v_0_neg, pi_0)    
    u_0 = tf.reduce_sum(tf.add(u_0_pos, u_0_neg), axis = 1, keepdims=True)
    # compute u_1
    # u_1 postive
    v_1_pos = tf.pow(tf.where(x1_ref_dep>0, x1_ref_dep, zeros), gamma)
    u_1_pos = tf.multiply(v_1_pos, pi_1)
    # u_1 negative
    v_1_neg = tf.multiply(-lambda_, tf.pow(-tf.where(x1_ref_dep<0, x1_ref_dep, zeros), gamma))
    u_1_neg = tf.multiply(v_1_neg, pi_1)
    u_1 = tf.reduce_sum(tf.add(u_1_pos, u_1_neg), axis = 1, keepdims=True)
    #     
    u_pt = tf.concat([u_0, u_1], axis = 1)
    
    ### DNN part
    hidden = x
    for i in range(M):
        hidden = standard_hidden_layer(hidden, n_hidden, BN, Dropout, Dropout_rate)
    u_dnn = tf.layers.dense(hidden, K, name = 'u_dnn')
    
    # combine the PT and DNN parts
    u = tf.add(u_pt, u_dnn)
    prob = tf.nn.softmax(u)

    # gradient info: only alternative 1 respect to inputs
    gradient_u1_x = [tf.concat([tf.gradients(u[:,1], [x0])[0], tf.gradients(u[:,1], [x1])[0], tf.gradients(u[:,1], [p0])[0], tf.gradients(u[:,1], [p1])[0],  tf.gradients(u[:,1], [z])[0]], axis = 1)]
    gradient_prob1_x = [tf.concat([tf.gradients(prob[:,1], [x0])[0], tf.gradients(prob[:,1], [x1])[0], tf.gradients(prob[:,1], [p0])[0], tf.gradients(prob[:,1], [p1])[0],  tf.gradients(prob[:,1], [z])[0]], axis = 1)]

    # regularization l2
    l2_regularizer = tf.contrib.layers.l2_regularizer(scale=penalty_const, scope=None)
    vars_ = tf.trainable_variables()
    regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, vars_)
    
    # 
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = u, labels = y), name = 'cost')
    cost += regularization_penalty
    log_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = u, labels = y), name = 'log_loss')
    optimizer = tf.train.AdamOptimizer() # opt: better than simple gradient descents: always converge to stable numbers.
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate) # opt objective
    training_op = optimizer.minimize(cost) # minimize the opt objective

    #    
    gradient_cost_x = [tf.concat([tf.gradients(cost, [x0])[0], tf.gradients(cost, [x1])[0], tf.gradients(cost, [p0])[0], tf.gradients(cost, [p1])[0], tf.gradients(cost, [z])[0]], axis = 1)]
    
    # eval params
    correct = tf.nn.in_top_k(u, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    cost_training=[]
    cost_testing=[]
    accuracy_training=[]
    accuracy_testing=[]
    log_loss_training=[]
    log_loss_testing=[]
    
    with tf.Session() as sess:
        # always run this to train the model
        init.run()
        feed_training = {x0: x0_training, x1: x1_training, p0:p0_training, p1: p1_training, z: z_training, y:y_training}
        feed_testing = {x0: x0_testing, x1: x1_testing, p0:p0_testing, p1: p1_testing, z: z_testing, y:y_testing}

        if restore == True:
            # case 1. restore models and evaluate key values
            saver.restore(sess, "tmp/"+MODEL_NAME+".ckpt")
            u_training = u.eval(feed_dict = feed_training)
            u_testing = u.eval(feed_dict = feed_testing)
            prob_training = prob.eval(feed_dict = feed_training)
            prob_testing = prob.eval(feed_dict = feed_testing)
            # 
            current_cost_training = cost.eval(feed_dict=feed_training)
            current_cost_testing = cost.eval(feed_dict=feed_testing)
            current_accuracy_training = accuracy.eval(feed_dict=feed_training)
            current_accuracy_testing = accuracy.eval(feed_dict=feed_testing)
            current_log_loss_training = log_loss.eval(feed_dict = feed_training)
            current_log_loss_testing = log_loss.eval(feed_dict = feed_testing)
            #
            log_loss_training.append(current_log_loss_training)
            log_loss_testing.append(current_log_loss_testing)
            cost_training.append(current_cost_training)
            cost_testing.append(current_cost_testing)
            accuracy_training.append(current_accuracy_training)
            accuracy_testing.append(current_accuracy_testing)
            #            
            gradient_u1_x_training=gradient_u1_x[0].eval(feed_dict = feed_training)
            gradient_u1_x_testing=gradient_u1_x[0].eval(feed_dict = feed_testing)
            gradient_prob1_x_training=gradient_prob1_x[0].eval(feed_dict = feed_training)
            gradient_prob1_x_testing=gradient_prob1_x[0].eval(feed_dict = feed_testing)
            gradient_cost_x_training=gradient_cost_x[0].eval(feed_dict = feed_training)          
            gradient_cost_x_testing=gradient_cost_x[0].eval(feed_dict = feed_testing)

            # save
            params_dic = {}
            hyper_params_dic = {}
            output_dic = {}
            output_dic['u_training'] = u_training
            output_dic['u_testing'] = u_testing
            output_dic['prob_training'] = prob_training
            output_dic['prob_testing'] = prob_testing

            output_dic['cost_training'] = cost_training
            output_dic['cost_testing'] = cost_testing
            output_dic['accuracy_training'] = accuracy_training
            output_dic['accuracy_testing'] = accuracy_testing
            output_dic['log_loss_training'] = log_loss_training
            output_dic['log_loss_testing'] = log_loss_testing
            
            output_dic['gradient_u1_x_training'] = gradient_u1_x_training
            output_dic['gradient_u1_x_testing'] = gradient_u1_x_testing
            output_dic['gradient_prob1_x_training'] = gradient_prob1_x_training
            output_dic['gradient_prob1_x_testing'] = gradient_prob1_x_testing
            output_dic['gradient_cost_x_training'] = gradient_cost_x_training
            output_dic['gradient_cost_x_testing'] = gradient_cost_x_testing

        elif restore == False:
            # case 3. train and save the estimated model
            for i in range(n_epoches):
                x0_training_batch,x1_training_batch,p0_training_batch,p1_training_batch,z_training_batch,y_training_batch = \
                     obtain_mini_batch_pt(x0_training,x1_training,p0_training,p1_training,z_training,y_training,n_mini_batch=100)               
                
                sess.run(training_op, feed_dict = {x0: x0_training_batch, x1: x1_training_batch, p0:p0_training_batch,
                    p1:p1_training_batch, z:z_training_batch, y:y_training_batch})

                if i % 1000 == 0:
                    print("Iterations", i, "Cost = ", cost.eval(feed_dict = feed_training))
                if i % 20 == 0:
                    current_cost_training = cost.eval(feed_dict=feed_training)
                    current_cost_testing = cost.eval(feed_dict=feed_testing)
                    current_accuracy_training = accuracy.eval(feed_dict=feed_training)
                    current_accuracy_testing = accuracy.eval(feed_dict=feed_testing)
                    current_log_loss_training=log_loss.eval(feed_dict=feed_training)
                    current_log_loss_testing=log_loss.eval(feed_dict=feed_testing)

                    cost_training.append(current_cost_training)
                    cost_testing.append(current_cost_testing)
                    accuracy_training.append(current_accuracy_training)
                    accuracy_testing.append(current_accuracy_testing)
                    log_loss_training.append(current_log_loss_training)
                    log_loss_testing.append(current_log_loss_testing)
    
            # save the model
            if accuracy_testing[-1] >= current_best_acc:
                saver.save(sess, "tmp/"+MODEL_NAME+".ckpt")
                current_best_acc = accuracy_testing[-1]
            # 
            u_training = u.eval(feed_dict = feed_training)
            u_testing = u.eval(feed_dict = feed_testing)
            prob_training = prob.eval(feed_dict = feed_training)
            prob_testing = prob.eval(feed_dict = feed_testing)
            # 
            gradient_u1_x_training=gradient_u1_x[0].eval(feed_dict = feed_training)
            gradient_u1_x_testing=gradient_u1_x[0].eval(feed_dict = feed_testing)
            gradient_prob1_x_training=gradient_prob1_x[0].eval(feed_dict = feed_training)
            gradient_prob1_x_testing=gradient_prob1_x[0].eval(feed_dict = feed_testing)
            gradient_cost_x_training=gradient_cost_x[0].eval(feed_dict = feed_training)            
            gradient_cost_x_testing=gradient_cost_x[0].eval(feed_dict = feed_testing)            

            # outputs
            params_dic = {}
            hyper_params_dic = {}
            output_dic = {}
            
            output_dic['cost_training'] = cost_training
            output_dic['cost_testing'] = cost_testing
            output_dic['accuracy_training'] = accuracy_training
            output_dic['accuracy_testing'] = accuracy_testing
            output_dic['log_loss_training'] = log_loss_training
            output_dic['log_loss_testing'] = log_loss_testing

            output_dic['u_training'] = u_training
            output_dic['u_testing'] = u_testing
            output_dic['prob_training'] = prob_training
            output_dic['prob_testing'] = prob_testing
            
            output_dic['gradient_u1_x_training'] = gradient_u1_x_training
            output_dic['gradient_u1_x_testing'] = gradient_u1_x_testing
            output_dic['gradient_prob1_x_training'] = gradient_prob1_x_training
            output_dic['gradient_prob1_x_testing'] = gradient_prob1_x_testing
            output_dic['gradient_cost_x_training'] = gradient_cost_x_training
            output_dic['gradient_cost_x_testing'] = gradient_cost_x_testing

            output_dic['current_best_acc'] = current_best_acc
            
    return params_dic,hyper_params_dic,output_dic


def est_pt_resnet_use_delta(data, pt_param_dic, penalty_const, MODEL_NAME, restore=False, n_epoches=10000,
                  n_mini_batch=100, l2_regu = 1e-50, M=3, n_hidden=100, simul_data=None, Dropout=False, Dropout_rate=0.01,
                  BN=False, x_dim=15, K=2, D=7, current_best_acc=0):
    '''
    estimate pt resnet model
    '''
    x0_training, x1_training, p0_training, p1_training, z_training, y_training = data['training']
    x0_testing, x1_testing, p0_testing, p1_testing, z_testing, y_testing = data['testing']
    #
    Dz = z_training.shape[1]
    # build models
    tf.reset_default_graph()
    z = tf.placeholder(dtype=tf.float32, shape=(None, Dz), name='z')
    x0 = tf.placeholder(dtype=tf.float32, shape=(None, 2), name='x0')
    x1 = tf.placeholder(dtype=tf.float32, shape=(None, 2), name='x1')
    p0 = tf.placeholder(dtype=tf.float32, shape=(None, 2), name='p0')
    p1 = tf.placeholder(dtype=tf.float32, shape=(None, 2), name='p1')
    x = tf.concat([x0, x1, p0, p1, z], axis=1, name='x')
    y = tf.placeholder(dtype=tf.int64, shape=(None), name='y')

    gamma_param_0 = pt_param_dic['gamma_param_0']
    beta_param_0 = pt_param_dic['beta_param_0']
    lambda_param_0 = pt_param_dic['lambda_param_0']
    ref_param_0 = pt_param_dic['ref_param_0']
    gamma_param = pt_param_dic['gamma_param']
    beta_param = pt_param_dic['beta_param']
    lambda_param = pt_param_dic['lambda_param']
    ref_param = pt_param_dic['ref_param']

    gamma_initial = tf.add(tf.matmul(z, gamma_param), gamma_param_0)
    beta_initial = tf.add(tf.matmul(z, beta_param), beta_param_0)
    lambda_initial = tf.add(tf.matmul(z, lambda_param), lambda_param_0)
    ref_initial = tf.add(tf.matmul(z, ref_param), ref_param_0)

    gamma = tf.nn.sigmoid(gamma_initial)
    beta = tf.nn.sigmoid(beta_initial)
    lambda_ = tf.nn.sigmoid(lambda_initial) * 4 + 1  # hence constrain lambda_ between 0 and 5
    ref = ref_initial * 1.0

    ### PT part
    # update x1 and x0 by ref pt
    x1_ref_dep = tf.subtract(x1, ref)
    x0_ref_dep = tf.subtract(x0, ref)
    # create zeros
    zeros = tf.zeros_like(x0, dtype=tf.float32)
    pi_1 = tf.exp(- tf.pow((-tf.log(p1)), beta))
    pi_0 = tf.exp(- tf.pow((-tf.log(p0)), beta))
    # compute u_0
    # u_0 postive
    v_0_pos = tf.pow(tf.where(x0_ref_dep > 0, x0_ref_dep, zeros), gamma)
    u_0_pos = tf.multiply(v_0_pos, pi_0)
    # u_0 negative
    v_0_neg = tf.multiply(-lambda_, tf.pow(-tf.where(x0_ref_dep < 0, x0_ref_dep, zeros), gamma))
    u_0_neg = tf.multiply(v_0_neg, pi_0)
    u_0 = tf.reduce_sum(tf.add(u_0_pos, u_0_neg), axis=1, keepdims=True)
    # compute u_1
    # u_1 postive
    v_1_pos = tf.pow(tf.where(x1_ref_dep > 0, x1_ref_dep, zeros), gamma)
    u_1_pos = tf.multiply(v_1_pos, pi_1)
    # u_1 negative
    v_1_neg = tf.multiply(-lambda_, tf.pow(-tf.where(x1_ref_dep < 0, x1_ref_dep, zeros), gamma))
    u_1_neg = tf.multiply(v_1_neg, pi_1)
    u_1 = tf.reduce_sum(tf.add(u_1_pos, u_1_neg), axis=1, keepdims=True)
    #
    u_pt = tf.concat([u_0, u_1], axis=1)

    ### DNN part
    hidden = x
    for i in range(M):
        hidden = standard_hidden_layer(hidden, n_hidden, BN, Dropout, Dropout_rate)
    u_dnn = tf.layers.dense(hidden, K, name='u_dnn')

    # combine the PT and DNN parts
    u = tf.add( (1 - penalty_const) * u_pt, penalty_const * u_dnn)
    prob = tf.nn.softmax(u)

    # gradient info: only alternative 1 respect to inputs
    gradient_u1_x = [tf.concat(
        [tf.gradients(u[:, 1], [x0])[0], tf.gradients(u[:, 1], [x1])[0], tf.gradients(u[:, 1], [p0])[0],
         tf.gradients(u[:, 1], [p1])[0], tf.gradients(u[:, 1], [z])[0]], axis=1)]
    gradient_prob1_x = [tf.concat(
        [tf.gradients(prob[:, 1], [x0])[0], tf.gradients(prob[:, 1], [x1])[0], tf.gradients(prob[:, 1], [p0])[0],
         tf.gradients(prob[:, 1], [p1])[0], tf.gradients(prob[:, 1], [z])[0]], axis=1)]

    # regularization l2
    l2_regularizer = tf.contrib.layers.l2_regularizer(scale=l2_regu, scope=None)
    vars_ = tf.trainable_variables()
    regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, vars_)

    #
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=u, labels=y), name='cost')
    cost += regularization_penalty
    log_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=u, labels=y), name='log_loss')
    optimizer = tf.train.AdamOptimizer()  # opt: better than simple gradient descents: always converge to stable numbers.
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate) # opt objective
    training_op = optimizer.minimize(cost)  # minimize the opt objective

    #
    gradient_cost_x = [tf.concat([tf.gradients(cost, [x0])[0], tf.gradients(cost, [x1])[0], tf.gradients(cost, [p0])[0],
                                  tf.gradients(cost, [p1])[0], tf.gradients(cost, [z])[0]], axis=1)]

    # eval params
    correct = tf.nn.in_top_k(u, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    cost_training = []
    cost_testing = []
    accuracy_training = []
    accuracy_testing = []
    log_loss_training = []
    log_loss_testing = []

    with tf.Session() as sess:
        # always run this to train the model
        init.run()
        feed_training = {x0: x0_training, x1: x1_training, p0: p0_training, p1: p1_training, z: z_training,
                         y: y_training}
        feed_testing = {x0: x0_testing, x1: x1_testing, p0: p0_testing, p1: p1_testing, z: z_testing, y: y_testing}

        if restore == True:
            # case 1. restore models and evaluate key values
            saver.restore(sess, "tmp/" + MODEL_NAME + ".ckpt")
            u_training = u.eval(feed_dict=feed_training)
            u_testing = u.eval(feed_dict=feed_testing)
            prob_training = prob.eval(feed_dict=feed_training)
            prob_testing = prob.eval(feed_dict=feed_testing)
            #
            current_cost_training = cost.eval(feed_dict=feed_training)
            current_cost_testing = cost.eval(feed_dict=feed_testing)
            current_accuracy_training = accuracy.eval(feed_dict=feed_training)
            current_accuracy_testing = accuracy.eval(feed_dict=feed_testing)
            current_log_loss_training = log_loss.eval(feed_dict=feed_training)
            current_log_loss_testing = log_loss.eval(feed_dict=feed_testing)
            #
            log_loss_training.append(current_log_loss_training)
            log_loss_testing.append(current_log_loss_testing)
            cost_training.append(current_cost_training)
            cost_testing.append(current_cost_testing)
            accuracy_training.append(current_accuracy_training)
            accuracy_testing.append(current_accuracy_testing)
            #
            gradient_u1_x_training = gradient_u1_x[0].eval(feed_dict=feed_training)
            gradient_u1_x_testing = gradient_u1_x[0].eval(feed_dict=feed_testing)
            gradient_prob1_x_training = gradient_prob1_x[0].eval(feed_dict=feed_training)
            gradient_prob1_x_testing = gradient_prob1_x[0].eval(feed_dict=feed_testing)
            gradient_cost_x_training = gradient_cost_x[0].eval(feed_dict=feed_training)
            gradient_cost_x_testing = gradient_cost_x[0].eval(feed_dict=feed_testing)

            # save
            params_dic = {}
            hyper_params_dic = {}
            output_dic = {}
            output_dic['u_training'] = u_training
            output_dic['u_testing'] = u_testing
            output_dic['prob_training'] = prob_training
            output_dic['prob_testing'] = prob_testing

            output_dic['cost_training'] = cost_training
            output_dic['cost_testing'] = cost_testing
            output_dic['accuracy_training'] = accuracy_training
            output_dic['accuracy_testing'] = accuracy_testing
            output_dic['log_loss_training'] = log_loss_training
            output_dic['log_loss_testing'] = log_loss_testing

            output_dic['gradient_u1_x_training'] = gradient_u1_x_training
            output_dic['gradient_u1_x_testing'] = gradient_u1_x_testing
            output_dic['gradient_prob1_x_training'] = gradient_prob1_x_training
            output_dic['gradient_prob1_x_testing'] = gradient_prob1_x_testing
            output_dic['gradient_cost_x_training'] = gradient_cost_x_training
            output_dic['gradient_cost_x_testing'] = gradient_cost_x_testing

        elif restore == False:
            # case 3. train and save the estimated model
            for i in range(n_epoches):
                x0_training_batch, x1_training_batch, p0_training_batch, p1_training_batch, z_training_batch, y_training_batch = \
                    obtain_mini_batch_pt(x0_training, x1_training, p0_training, p1_training, z_training, y_training,
                                         n_mini_batch=100)

                sess.run(training_op, feed_dict={x0: x0_training_batch, x1: x1_training_batch, p0: p0_training_batch,
                                                 p1: p1_training_batch, z: z_training_batch, y: y_training_batch})

                if i % 1000 == 0:
                    print("Iterations", i, "Cost = ", cost.eval(feed_dict=feed_training))
                if i % 20 == 0:
                    current_cost_training = cost.eval(feed_dict=feed_training)
                    current_cost_testing = cost.eval(feed_dict=feed_testing)
                    current_accuracy_training = accuracy.eval(feed_dict=feed_training)
                    current_accuracy_testing = accuracy.eval(feed_dict=feed_testing)
                    current_log_loss_training = log_loss.eval(feed_dict=feed_training)
                    current_log_loss_testing = log_loss.eval(feed_dict=feed_testing)

                    cost_training.append(current_cost_training)
                    cost_testing.append(current_cost_testing)
                    accuracy_training.append(current_accuracy_training)
                    accuracy_testing.append(current_accuracy_testing)
                    log_loss_training.append(current_log_loss_training)
                    log_loss_testing.append(current_log_loss_testing)

            # save the model
            if accuracy_testing[-1] >= current_best_acc:
                saver.save(sess, "tmp/" + MODEL_NAME + ".ckpt")
                current_best_acc = accuracy_testing[-1]
            #
            u_training = u.eval(feed_dict=feed_training)
            u_testing = u.eval(feed_dict=feed_testing)
            prob_training = prob.eval(feed_dict=feed_training)
            prob_testing = prob.eval(feed_dict=feed_testing)
            #
            gradient_u1_x_training = gradient_u1_x[0].eval(feed_dict=feed_training)
            gradient_u1_x_testing = gradient_u1_x[0].eval(feed_dict=feed_testing)
            gradient_prob1_x_training = gradient_prob1_x[0].eval(feed_dict=feed_training)
            gradient_prob1_x_testing = gradient_prob1_x[0].eval(feed_dict=feed_testing)
            gradient_cost_x_training = gradient_cost_x[0].eval(feed_dict=feed_training)
            gradient_cost_x_testing = gradient_cost_x[0].eval(feed_dict=feed_testing)

            # outputs
            params_dic = {}
            hyper_params_dic = {}
            output_dic = {}

            output_dic['cost_training'] = cost_training
            output_dic['cost_testing'] = cost_testing
            output_dic['accuracy_training'] = accuracy_training
            output_dic['accuracy_testing'] = accuracy_testing
            output_dic['log_loss_training'] = log_loss_training
            output_dic['log_loss_testing'] = log_loss_testing

            output_dic['u_training'] = u_training
            output_dic['u_testing'] = u_testing
            output_dic['prob_training'] = prob_training
            output_dic['prob_testing'] = prob_testing

            output_dic['gradient_u1_x_training'] = gradient_u1_x_training
            output_dic['gradient_u1_x_testing'] = gradient_u1_x_testing
            output_dic['gradient_prob1_x_training'] = gradient_prob1_x_training
            output_dic['gradient_prob1_x_testing'] = gradient_prob1_x_testing
            output_dic['gradient_cost_x_training'] = gradient_cost_x_training
            output_dic['gradient_cost_x_testing'] = gradient_cost_x_testing

            output_dic['current_best_acc'] = current_best_acc

    return params_dic, hyper_params_dic, output_dic




def est_pt_resnet_use_delta_save_train(data, pt_param_dic, penalty_const, MODEL_NAME, restore=False, n_epoches=10000,
                  n_mini_batch=100, l2_regu = 1e-50, M=3, n_hidden=100, simul_data=None, Dropout=False, Dropout_rate=0.01,
                  BN=False, x_dim=15, K=2, D=7, current_best_acc=0):
    '''
    estimate pt resnet model
    '''
    x0_training, x1_training, p0_training, p1_training, z_training, y_training = data['training']
    x0_testing, x1_testing, p0_testing, p1_testing, z_testing, y_testing = data['testing']
    #
    Dz = z_training.shape[1]
    # build models
    tf.reset_default_graph()
    z = tf.placeholder(dtype=tf.float32, shape=(None, Dz), name='z')
    x0 = tf.placeholder(dtype=tf.float32, shape=(None, 2), name='x0')
    x1 = tf.placeholder(dtype=tf.float32, shape=(None, 2), name='x1')
    p0 = tf.placeholder(dtype=tf.float32, shape=(None, 2), name='p0')
    p1 = tf.placeholder(dtype=tf.float32, shape=(None, 2), name='p1')
    x = tf.concat([x0, x1, p0, p1, z], axis=1, name='x')
    y = tf.placeholder(dtype=tf.int64, shape=(None), name='y')

    gamma_param_0 = pt_param_dic['gamma_param_0']
    beta_param_0 = pt_param_dic['beta_param_0']
    lambda_param_0 = pt_param_dic['lambda_param_0']
    ref_param_0 = pt_param_dic['ref_param_0']
    gamma_param = pt_param_dic['gamma_param']
    beta_param = pt_param_dic['beta_param']
    lambda_param = pt_param_dic['lambda_param']
    ref_param = pt_param_dic['ref_param']

    gamma_initial = tf.add(tf.matmul(z, gamma_param), gamma_param_0)
    beta_initial = tf.add(tf.matmul(z, beta_param), beta_param_0)
    lambda_initial = tf.add(tf.matmul(z, lambda_param), lambda_param_0)
    ref_initial = tf.add(tf.matmul(z, ref_param), ref_param_0)

    gamma = tf.nn.sigmoid(gamma_initial)
    beta = tf.nn.sigmoid(beta_initial)
    lambda_ = tf.nn.sigmoid(lambda_initial) * 4 + 1  # hence constrain lambda_ between 0 and 5
    ref = ref_initial * 1.0

    ### PT part
    # update x1 and x0 by ref pt
    x1_ref_dep = tf.subtract(x1, ref)
    x0_ref_dep = tf.subtract(x0, ref)
    # create zeros
    zeros = tf.zeros_like(x0, dtype=tf.float32)
    pi_1 = tf.exp(- tf.pow((-tf.log(p1)), beta))
    pi_0 = tf.exp(- tf.pow((-tf.log(p0)), beta))
    # compute u_0
    # u_0 postive
    v_0_pos = tf.pow(tf.where(x0_ref_dep > 0, x0_ref_dep, zeros), gamma)
    u_0_pos = tf.multiply(v_0_pos, pi_0)
    # u_0 negative
    v_0_neg = tf.multiply(-lambda_, tf.pow(-tf.where(x0_ref_dep < 0, x0_ref_dep, zeros), gamma))
    u_0_neg = tf.multiply(v_0_neg, pi_0)
    u_0 = tf.reduce_sum(tf.add(u_0_pos, u_0_neg), axis=1, keepdims=True)
    # compute u_1
    # u_1 postive
    v_1_pos = tf.pow(tf.where(x1_ref_dep > 0, x1_ref_dep, zeros), gamma)
    u_1_pos = tf.multiply(v_1_pos, pi_1)
    # u_1 negative
    v_1_neg = tf.multiply(-lambda_, tf.pow(-tf.where(x1_ref_dep < 0, x1_ref_dep, zeros), gamma))
    u_1_neg = tf.multiply(v_1_neg, pi_1)
    u_1 = tf.reduce_sum(tf.add(u_1_pos, u_1_neg), axis=1, keepdims=True)
    #
    u_pt = tf.concat([u_0, u_1], axis=1)

    ### DNN part
    hidden = x
    for i in range(M):
        hidden = standard_hidden_layer(hidden, n_hidden, BN, Dropout, Dropout_rate)
    u_dnn = tf.layers.dense(hidden, K, name='u_dnn')

    # combine the PT and DNN parts
    u = tf.add( (1 - penalty_const) * u_pt, penalty_const * u_dnn)
    prob = tf.nn.softmax(u)

    # gradient info: only alternative 1 respect to inputs
    gradient_u1_x = [tf.concat(
        [tf.gradients(u[:, 1], [x0])[0], tf.gradients(u[:, 1], [x1])[0], tf.gradients(u[:, 1], [p0])[0],
         tf.gradients(u[:, 1], [p1])[0], tf.gradients(u[:, 1], [z])[0]], axis=1)]
    gradient_prob1_x = [tf.concat(
        [tf.gradients(prob[:, 1], [x0])[0], tf.gradients(prob[:, 1], [x1])[0], tf.gradients(prob[:, 1], [p0])[0],
         tf.gradients(prob[:, 1], [p1])[0], tf.gradients(prob[:, 1], [z])[0]], axis=1)]

    # regularization l2
    l2_regularizer = tf.contrib.layers.l2_regularizer(scale=l2_regu, scope=None)
    vars_ = tf.trainable_variables()
    regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, vars_)

    #
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=u, labels=y), name='cost')
    cost += regularization_penalty
    log_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=u, labels=y), name='log_loss')
    optimizer = tf.train.AdamOptimizer()  # opt: better than simple gradient descents: always converge to stable numbers.
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate) # opt objective
    training_op = optimizer.minimize(cost)  # minimize the opt objective

    #
    gradient_cost_x = [tf.concat([tf.gradients(cost, [x0])[0], tf.gradients(cost, [x1])[0], tf.gradients(cost, [p0])[0],
                                  tf.gradients(cost, [p1])[0], tf.gradients(cost, [z])[0]], axis=1)]

    # eval params
    correct = tf.nn.in_top_k(u, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    cost_training = []
    cost_testing = []
    accuracy_training = []
    accuracy_testing = []
    log_loss_training = []
    log_loss_testing = []

    with tf.Session() as sess:
        # always run this to train the model
        init.run()
        feed_training = {x0: x0_training, x1: x1_training, p0: p0_training, p1: p1_training, z: z_training,
                         y: y_training}
        feed_testing = {x0: x0_testing, x1: x1_testing, p0: p0_testing, p1: p1_testing, z: z_testing, y: y_testing}

        if restore == True:
            # case 1. restore models and evaluate key values
            saver.restore(sess, "tmp/" + MODEL_NAME + ".ckpt")
            u_training = u.eval(feed_dict=feed_training)
            u_testing = u.eval(feed_dict=feed_testing)
            prob_training = prob.eval(feed_dict=feed_training)
            prob_testing = prob.eval(feed_dict=feed_testing)
            #
            current_cost_training = cost.eval(feed_dict=feed_training)
            current_cost_testing = cost.eval(feed_dict=feed_testing)
            current_accuracy_training = accuracy.eval(feed_dict=feed_training)
            current_accuracy_testing = accuracy.eval(feed_dict=feed_testing)
            current_log_loss_training = log_loss.eval(feed_dict=feed_training)
            current_log_loss_testing = log_loss.eval(feed_dict=feed_testing)
            #
            log_loss_training.append(current_log_loss_training)
            log_loss_testing.append(current_log_loss_testing)
            cost_training.append(current_cost_training)
            cost_testing.append(current_cost_testing)
            accuracy_training.append(current_accuracy_training)
            accuracy_testing.append(current_accuracy_testing)
            #
            gradient_u1_x_training = gradient_u1_x[0].eval(feed_dict=feed_training)
            gradient_u1_x_testing = gradient_u1_x[0].eval(feed_dict=feed_testing)
            gradient_prob1_x_training = gradient_prob1_x[0].eval(feed_dict=feed_training)
            gradient_prob1_x_testing = gradient_prob1_x[0].eval(feed_dict=feed_testing)
            gradient_cost_x_training = gradient_cost_x[0].eval(feed_dict=feed_training)
            gradient_cost_x_testing = gradient_cost_x[0].eval(feed_dict=feed_testing)

            # save
            params_dic = {}
            hyper_params_dic = {}
            output_dic = {}
            output_dic['u_training'] = u_training
            output_dic['u_testing'] = u_testing
            output_dic['prob_training'] = prob_training
            output_dic['prob_testing'] = prob_testing

            output_dic['cost_training'] = cost_training
            output_dic['cost_testing'] = cost_testing
            output_dic['accuracy_training'] = accuracy_training
            output_dic['accuracy_testing'] = accuracy_testing
            output_dic['log_loss_training'] = log_loss_training
            output_dic['log_loss_testing'] = log_loss_testing

            output_dic['gradient_u1_x_training'] = gradient_u1_x_training
            output_dic['gradient_u1_x_testing'] = gradient_u1_x_testing
            output_dic['gradient_prob1_x_training'] = gradient_prob1_x_training
            output_dic['gradient_prob1_x_testing'] = gradient_prob1_x_testing
            output_dic['gradient_cost_x_training'] = gradient_cost_x_training
            output_dic['gradient_cost_x_testing'] = gradient_cost_x_testing

        elif restore == False:
            # case 3. train and save the estimated model
            for i in range(n_epoches):
                x0_training_batch, x1_training_batch, p0_training_batch, p1_training_batch, z_training_batch, y_training_batch = \
                    obtain_mini_batch_pt(x0_training, x1_training, p0_training, p1_training, z_training, y_training,
                                         n_mini_batch=100)

                sess.run(training_op, feed_dict={x0: x0_training_batch, x1: x1_training_batch, p0: p0_training_batch,
                                                 p1: p1_training_batch, z: z_training_batch, y: y_training_batch})

                if i % 1000 == 0:
                    print("Iterations", i, "Cost = ", cost.eval(feed_dict=feed_training))

                current_cost_training = cost.eval(feed_dict=feed_training)
                current_cost_testing = cost.eval(feed_dict=feed_testing)
                current_accuracy_training = accuracy.eval(feed_dict=feed_training)
                current_accuracy_testing = accuracy.eval(feed_dict=feed_testing)
                current_log_loss_training = log_loss.eval(feed_dict=feed_training)
                current_log_loss_testing = log_loss.eval(feed_dict=feed_testing)

                cost_training.append(current_cost_training)
                cost_testing.append(current_cost_testing)
                accuracy_training.append(current_accuracy_training)
                accuracy_testing.append(current_accuracy_testing)
                log_loss_training.append(current_log_loss_training)
                log_loss_testing.append(current_log_loss_testing)

            # save the model
            if accuracy_testing[-1] >= current_best_acc:
                saver.save(sess, "tmp/" + MODEL_NAME + ".ckpt")
                current_best_acc = accuracy_testing[-1]
            #
            u_training = u.eval(feed_dict=feed_training)
            u_testing = u.eval(feed_dict=feed_testing)
            prob_training = prob.eval(feed_dict=feed_training)
            prob_testing = prob.eval(feed_dict=feed_testing)
            #
            gradient_u1_x_training = gradient_u1_x[0].eval(feed_dict=feed_training)
            gradient_u1_x_testing = gradient_u1_x[0].eval(feed_dict=feed_testing)
            gradient_prob1_x_training = gradient_prob1_x[0].eval(feed_dict=feed_training)
            gradient_prob1_x_testing = gradient_prob1_x[0].eval(feed_dict=feed_testing)
            gradient_cost_x_training = gradient_cost_x[0].eval(feed_dict=feed_training)
            gradient_cost_x_testing = gradient_cost_x[0].eval(feed_dict=feed_testing)

            # outputs
            params_dic = {}
            hyper_params_dic = {}
            output_dic = {}

            output_dic['cost_training'] = cost_training
            output_dic['cost_testing'] = cost_testing
            output_dic['accuracy_training'] = accuracy_training
            output_dic['accuracy_testing'] = accuracy_testing
            output_dic['log_loss_training'] = log_loss_training
            output_dic['log_loss_testing'] = log_loss_testing

            output_dic['u_training'] = u_training
            output_dic['u_testing'] = u_testing
            output_dic['prob_training'] = prob_training
            output_dic['prob_testing'] = prob_testing

            output_dic['gradient_u1_x_training'] = gradient_u1_x_training
            output_dic['gradient_u1_x_testing'] = gradient_u1_x_testing
            output_dic['gradient_prob1_x_training'] = gradient_prob1_x_training
            output_dic['gradient_prob1_x_testing'] = gradient_prob1_x_testing
            output_dic['gradient_cost_x_training'] = gradient_cost_x_training
            output_dic['gradient_cost_x_testing'] = gradient_cost_x_testing

            output_dic['current_best_acc'] = current_best_acc

    return params_dic, hyper_params_dic, output_dic



def est_pt_resnet_reverse(data, dnn_param_dic, penalty_const, MODEL_NAME, restore=False, n_epoches=10000, n_mini_batch=100, M=3,
                  n_hidden=100, simul_data=None, Dropout=False, Dropout_rate=0.01, BN=False, x_dim=15, K=2, D=7):
    '''
    estimate pt resnet model
    '''
    x0_training, x1_training, p0_training, p1_training, z_training, y_training = data['training']
    x0_testing, x1_testing, p0_testing, p1_testing, z_testing, y_testing = data['testing']
    #
    Dz = z_training.shape[1]
    # build models
    tf.reset_default_graph()
    z = tf.placeholder(dtype=tf.float32, shape=(None, Dz), name='z')
    x0 = tf.placeholder(dtype=tf.float32, shape=(None, 2), name='x0')
    x1 = tf.placeholder(dtype=tf.float32, shape=(None, 2), name='x1')
    p0 = tf.placeholder(dtype=tf.float32, shape=(None, 2), name='p0')
    p1 = tf.placeholder(dtype=tf.float32, shape=(None, 2), name='p1')
    x = tf.concat([x0, x1, p0, p1, z], axis=1, name='x')
    y = tf.placeholder(dtype=tf.int64, shape=(None), name='y')

    gamma_param_0 = pt_param_dic['gamma_param_0']
    beta_param_0 = pt_param_dic['beta_param_0']
    lambda_param_0 = pt_param_dic['lambda_param_0']
    ref_param_0 = pt_param_dic['ref_param_0']
    gamma_param = pt_param_dic['gamma_param']
    beta_param = pt_param_dic['beta_param']
    lambda_param = pt_param_dic['lambda_param']
    ref_param = pt_param_dic['ref_param']

    gamma_initial = tf.add(tf.matmul(z, gamma_param), gamma_param_0)
    beta_initial = tf.add(tf.matmul(z, beta_param), beta_param_0)
    lambda_initial = tf.add(tf.matmul(z, lambda_param), lambda_param_0)
    ref_initial = tf.add(tf.matmul(z, ref_param), ref_param_0)

    gamma = tf.nn.sigmoid(gamma_initial)
    beta = tf.nn.sigmoid(beta_initial)
    lambda_ = tf.nn.sigmoid(lambda_initial) * 4 + 1  # hence constrain lambda_ between 0 and 5
    ref = ref_initial * 1.0

    ### PT part
    # update x1 and x0 by ref pt
    x1_ref_dep = tf.subtract(x1, ref)
    x0_ref_dep = tf.subtract(x0, ref)
    # create zeros
    zeros = tf.zeros_like(x0, dtype=tf.float32)
    pi_1 = tf.exp(- tf.pow((-tf.log(p1)), beta))
    pi_0 = tf.exp(- tf.pow((-tf.log(p0)), beta))
    # compute u_0
    # u_0 postive
    v_0_pos = tf.pow(tf.where(x0_ref_dep > 0, x0_ref_dep, zeros), gamma)
    u_0_pos = tf.multiply(v_0_pos, pi_0)
    # u_0 negative
    v_0_neg = tf.multiply(-lambda_, tf.pow(-tf.where(x0_ref_dep < 0, x0_ref_dep, zeros), gamma))
    u_0_neg = tf.multiply(v_0_neg, pi_0)
    u_0 = tf.reduce_sum(tf.add(u_0_pos, u_0_neg), axis=1, keepdims=True)
    # compute u_1
    # u_1 postive
    v_1_pos = tf.pow(tf.where(x1_ref_dep > 0, x1_ref_dep, zeros), gamma)
    u_1_pos = tf.multiply(v_1_pos, pi_1)
    # u_1 negative
    v_1_neg = tf.multiply(-lambda_, tf.pow(-tf.where(x1_ref_dep < 0, x1_ref_dep, zeros), gamma))
    u_1_neg = tf.multiply(v_1_neg, pi_1)
    u_1 = tf.reduce_sum(tf.add(u_1_pos, u_1_neg), axis=1, keepdims=True)
    #
    u_pt = tf.concat([u_0, u_1], axis=1)

    ### DNN part
    hidden = x
    for i in range(M):
        hidden = standard_hidden_layer(hidden, n_hidden, BN, Dropout, Dropout_rate)
    u_dnn = tf.layers.dense(hidden, K, name='u_dnn')

    # combine the PT and DNN parts
    u = tf.add(u_pt, u_dnn)
    prob = tf.nn.softmax(u)

    # gradient info: only alternative 1 respect to inputs
    gradient_u1_x = [tf.concat(
        [tf.gradients(u[:, 1], [x0])[0], tf.gradients(u[:, 1], [x1])[0], tf.gradients(u[:, 1], [p0])[0],
         tf.gradients(u[:, 1], [p1])[0], tf.gradients(u[:, 1], [z])[0]], axis=1)]
    gradient_prob1_x = [tf.concat(
        [tf.gradients(prob[:, 1], [x0])[0], tf.gradients(prob[:, 1], [x1])[0], tf.gradients(prob[:, 1], [p0])[0],
         tf.gradients(prob[:, 1], [p1])[0], tf.gradients(prob[:, 1], [z])[0]], axis=1)]

    # regularization l2
    l2_regularizer = tf.contrib.layers.l2_regularizer(scale=penalty_const, scope=None)
    vars_ = tf.trainable_variables()
    regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, vars_)

    #
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=u, labels=y), name='cost')
    cost += regularization_penalty
    log_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=u, labels=y), name='log_loss')
    optimizer = tf.train.AdamOptimizer()  # opt: better than simple gradient descents: always converge to stable numbers.
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate) # opt objective
    training_op = optimizer.minimize(cost)  # minimize the opt objective

    #
    gradient_cost_x = [tf.concat([tf.gradients(cost, [x0])[0], tf.gradients(cost, [x1])[0], tf.gradients(cost, [p0])[0],
                                  tf.gradients(cost, [p1])[0], tf.gradients(cost, [z])[0]], axis=1)]

    # eval params
    correct = tf.nn.in_top_k(u, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    cost_training = []
    cost_testing = []
    accuracy_training = []
    accuracy_testing = []
    log_loss_training = []
    log_loss_testing = []

    with tf.Session() as sess:
        # always run this to train the model
        init.run()
        feed_training = {x0: x0_training, x1: x1_training, p0: p0_training, p1: p1_training, z: z_training,
                         y: y_training}
        feed_testing = {x0: x0_testing, x1: x1_testing, p0: p0_testing, p1: p1_testing, z: z_testing, y: y_testing}

        if restore == True:
            # case 1. restore models and evaluate key values
            saver.restore(sess, "tmp/" + MODEL_NAME + ".ckpt")
            u_training = u.eval(feed_dict=feed_training)
            u_testing = u.eval(feed_dict=feed_testing)
            prob_training = prob.eval(feed_dict=feed_training)
            prob_testing = prob.eval(feed_dict=feed_testing)
            #
            current_cost_training = cost.eval(feed_dict=feed_training)
            current_cost_testing = cost.eval(feed_dict=feed_testing)
            current_accuracy_training = accuracy.eval(feed_dict=feed_training)
            current_accuracy_testing = accuracy.eval(feed_dict=feed_testing)
            current_log_loss_training = log_loss.eval(feed_dict=feed_training)
            current_log_loss_testing = log_loss.eval(feed_dict=feed_testing)
            #
            log_loss_training.append(current_log_loss_training)
            log_loss_testing.append(current_log_loss_testing)
            cost_training.append(current_cost_training)
            cost_testing.append(current_cost_testing)
            accuracy_training.append(current_accuracy_training)
            accuracy_testing.append(current_accuracy_testing)
            #
            gradient_u1_x_training = gradient_u1_x[0].eval(feed_dict=feed_training)
            gradient_u1_x_testing = gradient_u1_x[0].eval(feed_dict=feed_testing)
            gradient_prob1_x_training = gradient_prob1_x[0].eval(feed_dict=feed_training)
            gradient_prob1_x_testing = gradient_prob1_x[0].eval(feed_dict=feed_testing)
            gradient_cost_x_training = gradient_cost_x[0].eval(feed_dict=feed_training)
            gradient_cost_x_testing = gradient_cost_x[0].eval(feed_dict=feed_testing)

            # save
            params_dic = {}
            hyper_params_dic = {}
            output_dic = {}
            output_dic['u_training'] = u_training
            output_dic['u_testing'] = u_testing
            output_dic['prob_training'] = prob_training
            output_dic['prob_testing'] = prob_testing

            output_dic['cost_training'] = cost_training
            output_dic['cost_testing'] = cost_testing
            output_dic['accuracy_training'] = accuracy_training
            output_dic['accuracy_testing'] = accuracy_testing
            output_dic['log_loss_training'] = log_loss_training
            output_dic['log_loss_testing'] = log_loss_testing

            output_dic['gradient_u1_x_training'] = gradient_u1_x_training
            output_dic['gradient_u1_x_testing'] = gradient_u1_x_testing
            output_dic['gradient_prob1_x_training'] = gradient_prob1_x_training
            output_dic['gradient_prob1_x_testing'] = gradient_prob1_x_testing
            output_dic['gradient_cost_x_training'] = gradient_cost_x_training
            output_dic['gradient_cost_x_testing'] = gradient_cost_x_testing

        elif restore == False:
            # case 3. train and save the estimated model
            for i in range(n_epoches):
                x0_training_batch, x1_training_batch, p0_training_batch, p1_training_batch, z_training_batch, y_training_batch = \
                    obtain_mini_batch_pt(x0_training, x1_training, p0_training, p1_training, z_training, y_training,
                                         n_mini_batch=100)

                sess.run(training_op, feed_dict={x0: x0_training_batch, x1: x1_training_batch, p0: p0_training_batch,
                                                 p1: p1_training_batch, z: z_training_batch, y: y_training_batch})

                if i % 1000 == 0:
                    print("Iterations", i, "Cost = ", cost.eval(feed_dict=feed_training))
                if i % 20 == 0:
                    current_cost_training = cost.eval(feed_dict=feed_training)
                    current_cost_testing = cost.eval(feed_dict=feed_testing)
                    current_accuracy_training = accuracy.eval(feed_dict=feed_training)
                    current_accuracy_testing = accuracy.eval(feed_dict=feed_testing)
                    current_log_loss_training = log_loss.eval(feed_dict=feed_training)
                    current_log_loss_testing = log_loss.eval(feed_dict=feed_testing)

                    cost_training.append(current_cost_training)
                    cost_testing.append(current_cost_testing)
                    accuracy_training.append(current_accuracy_training)
                    accuracy_testing.append(current_accuracy_testing)
                    log_loss_training.append(current_log_loss_training)
                    log_loss_testing.append(current_log_loss_testing)

            # save the model
            saver.save(sess, "tmp/" + MODEL_NAME + ".ckpt")
            #
            u_training = u.eval(feed_dict=feed_training)
            u_testing = u.eval(feed_dict=feed_testing)
            prob_training = prob.eval(feed_dict=feed_training)
            prob_testing = prob.eval(feed_dict=feed_testing)
            #
            gradient_u1_x_training = gradient_u1_x[0].eval(feed_dict=feed_training)
            gradient_u1_x_testing = gradient_u1_x[0].eval(feed_dict=feed_testing)
            gradient_prob1_x_training = gradient_prob1_x[0].eval(feed_dict=feed_training)
            gradient_prob1_x_testing = gradient_prob1_x[0].eval(feed_dict=feed_testing)
            gradient_cost_x_training = gradient_cost_x[0].eval(feed_dict=feed_training)
            gradient_cost_x_testing = gradient_cost_x[0].eval(feed_dict=feed_testing)

            # outputs
            params_dic = {}
            hyper_params_dic = {}
            output_dic = {}

            output_dic['cost_training'] = cost_training
            output_dic['cost_testing'] = cost_testing
            output_dic['accuracy_training'] = accuracy_training
            output_dic['accuracy_testing'] = accuracy_testing
            output_dic['log_loss_training'] = log_loss_training
            output_dic['log_loss_testing'] = log_loss_testing

            output_dic['u_training'] = u_training
            output_dic['u_testing'] = u_testing
            output_dic['prob_training'] = prob_training
            output_dic['prob_testing'] = prob_testing

            output_dic['gradient_u1_x_training'] = gradient_u1_x_training
            output_dic['gradient_u1_x_testing'] = gradient_u1_x_testing
            output_dic['gradient_prob1_x_training'] = gradient_prob1_x_training
            output_dic['gradient_prob1_x_testing'] = gradient_prob1_x_testing
            output_dic['gradient_cost_x_training'] = gradient_cost_x_training
            output_dic['gradient_cost_x_testing'] = gradient_cost_x_testing

    return params_dic, hyper_params_dic, output_dic

def est_pt_resnet_simultaneous(data, penalty_const, MODEL_NAME, restore=False, n_epoches=10000, n_mini_batch=100, M=3,
                  n_hidden=100, simul_data=None, Dropout=False, Dropout_rate=0.01, BN=False, x_dim=15, K=2, D=7, current_best_acc = 0):
    '''
    estimate pt resnet model
    '''
    x0_training, x1_training, p0_training, p1_training, z_training, y_training = data['training']
    x0_testing, x1_testing, p0_testing, p1_testing, z_testing, y_testing = data['testing']
    #
    Dz = z_training.shape[1]
    # build models
    tf.reset_default_graph()
    z = tf.placeholder(dtype=tf.float32, shape=(None, Dz), name='z')
    x0 = tf.placeholder(dtype=tf.float32, shape=(None, 2), name='x0')
    x1 = tf.placeholder(dtype=tf.float32, shape=(None, 2), name='x1')
    p0 = tf.placeholder(dtype=tf.float32, shape=(None, 2), name='p0')
    p1 = tf.placeholder(dtype=tf.float32, shape=(None, 2), name='p1')
    x = tf.concat([x0, x1, p0, p1, z], axis=1, name='x')
    y = tf.placeholder(dtype=tf.int64, shape=(None), name='y')

    gamma_param_0 = tf.Variable(tf.random_normal([1]), dtype = tf.float32)
    beta_param_0 = tf.Variable(tf.random_normal([1]), dtype = tf.float32)
    lambda_param_0 = tf.Variable(tf.random_normal([1]), dtype = tf.float32)
    ref_param_0 = tf.Variable(tf.random_normal([1]), dtype = tf.float32)
    gamma_param = tf.Variable(tf.random_normal([Dz, 1]), dtype = tf.float32)
    beta_param = tf.Variable(tf.random_normal([Dz, 1]), dtype = tf.float32)
    lambda_param = tf.Variable(tf.random_normal([Dz, 1]), dtype = tf.float32)
    ref_param = tf.Variable(tf.random_normal([Dz, 1]), dtype = tf.float32)

    gamma_initial = tf.add(tf.matmul(z, gamma_param), gamma_param_0)
    beta_initial = tf.add(tf.matmul(z, beta_param), beta_param_0)
    lambda_initial = tf.add(tf.matmul(z, lambda_param), lambda_param_0)
    ref_initial = tf.add(tf.matmul(z, ref_param), ref_param_0)

    gamma = tf.nn.sigmoid(gamma_initial)
    beta = tf.nn.sigmoid(beta_initial)
    lambda_ = tf.nn.sigmoid(lambda_initial) * 4 + 1  # hence constrain lambda_ between 0 and 5
    ref = ref_initial * 1.0

    ### PT part
    # update x1 and x0 by ref pt
    x1_ref_dep = tf.subtract(x1, ref)
    x0_ref_dep = tf.subtract(x0, ref)
    # create zeros
    zeros = tf.zeros_like(x0, dtype=tf.float32)
    pi_1 = tf.exp(- tf.pow((-tf.log(p1)), beta))
    pi_0 = tf.exp(- tf.pow((-tf.log(p0)), beta))
    # compute u_0
    # u_0 postive
    v_0_pos = tf.pow(tf.where(x0_ref_dep > 0, x0_ref_dep, zeros), gamma)
    u_0_pos = tf.multiply(v_0_pos, pi_0)
    # u_0 negative
    v_0_neg = tf.multiply(-lambda_, tf.pow(-tf.where(x0_ref_dep < 0, x0_ref_dep, zeros), gamma))
    u_0_neg = tf.multiply(v_0_neg, pi_0)
    u_0 = tf.reduce_sum(tf.add(u_0_pos, u_0_neg), axis=1, keepdims=True)
    # compute u_1
    # u_1 postive
    v_1_pos = tf.pow(tf.where(x1_ref_dep > 0, x1_ref_dep, zeros), gamma)
    u_1_pos = tf.multiply(v_1_pos, pi_1)
    # u_1 negative
    v_1_neg = tf.multiply(-lambda_, tf.pow(-tf.where(x1_ref_dep < 0, x1_ref_dep, zeros), gamma))
    u_1_neg = tf.multiply(v_1_neg, pi_1)
    u_1 = tf.reduce_sum(tf.add(u_1_pos, u_1_neg), axis=1, keepdims=True)
    #
    u_pt = tf.concat([u_0, u_1], axis=1)

    ### DNN part
    hidden = x
    for i in range(M):
        hidden = standard_hidden_layer(hidden, n_hidden, BN, Dropout, Dropout_rate)
    u_dnn = tf.layers.dense(hidden, K, name='u_dnn')

    # combine the PT and DNN parts
    u = tf.add(u_pt, u_dnn)
    prob = tf.nn.softmax(u)

    # gradient info: only alternative 1 respect to inputs
    gradient_u1_x = [tf.concat(
        [tf.gradients(u[:, 1], [x0])[0], tf.gradients(u[:, 1], [x1])[0], tf.gradients(u[:, 1], [p0])[0],
         tf.gradients(u[:, 1], [p1])[0], tf.gradients(u[:, 1], [z])[0]], axis=1)]
    gradient_prob1_x = [tf.concat(
        [tf.gradients(prob[:, 1], [x0])[0], tf.gradients(prob[:, 1], [x1])[0], tf.gradients(prob[:, 1], [p0])[0],
         tf.gradients(prob[:, 1], [p1])[0], tf.gradients(prob[:, 1], [z])[0]], axis=1)]

    # regularization l2
    l2_regularizer = tf.contrib.layers.l2_regularizer(scale=penalty_const, scope=None)
    vars_ = tf.trainable_variables()
    regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, vars_)

    #
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=u, labels=y), name='cost')
    cost += regularization_penalty
    log_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=u, labels=y), name='log_loss')
    optimizer = tf.train.AdamOptimizer()  # opt: better than simple gradient descents: always converge to stable numbers.
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate) # opt objective
    training_op = optimizer.minimize(cost)  # minimize the opt objective

    #
    gradient_cost_x = [tf.concat([tf.gradients(cost, [x0])[0], tf.gradients(cost, [x1])[0], tf.gradients(cost, [p0])[0],
                                  tf.gradients(cost, [p1])[0], tf.gradients(cost, [z])[0]], axis=1)]

    # eval params
    correct = tf.nn.in_top_k(u, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    cost_training = []
    cost_testing = []
    accuracy_training = []
    accuracy_testing = []
    log_loss_training = []
    log_loss_testing = []

    with tf.Session() as sess:
        # always run this to train the model
        init.run()
        feed_training = {x0: x0_training, x1: x1_training, p0: p0_training, p1: p1_training, z: z_training,
                         y: y_training}
        feed_testing = {x0: x0_testing, x1: x1_testing, p0: p0_testing, p1: p1_testing, z: z_testing, y: y_testing}

        if restore == True:
            # case 1. restore models and evaluate key values
            saver.restore(sess, "tmp/" + MODEL_NAME + ".ckpt")
            u_training = u.eval(feed_dict=feed_training)
            u_testing = u.eval(feed_dict=feed_testing)
            prob_training = prob.eval(feed_dict=feed_training)
            prob_testing = prob.eval(feed_dict=feed_testing)
            #
            current_cost_training = cost.eval(feed_dict=feed_training)
            current_cost_testing = cost.eval(feed_dict=feed_testing)
            current_accuracy_training = accuracy.eval(feed_dict=feed_training)
            current_accuracy_testing = accuracy.eval(feed_dict=feed_testing)
            current_log_loss_training = log_loss.eval(feed_dict=feed_training)
            current_log_loss_testing = log_loss.eval(feed_dict=feed_testing)
            #
            log_loss_training.append(current_log_loss_training)
            log_loss_testing.append(current_log_loss_testing)
            cost_training.append(current_cost_training)
            cost_testing.append(current_cost_testing)
            accuracy_training.append(current_accuracy_training)
            accuracy_testing.append(current_accuracy_testing)
            #
            gradient_u1_x_training = gradient_u1_x[0].eval(feed_dict=feed_training)
            gradient_u1_x_testing = gradient_u1_x[0].eval(feed_dict=feed_testing)
            gradient_prob1_x_training = gradient_prob1_x[0].eval(feed_dict=feed_training)
            gradient_prob1_x_testing = gradient_prob1_x[0].eval(feed_dict=feed_testing)
            gradient_cost_x_training = gradient_cost_x[0].eval(feed_dict=feed_training)
            gradient_cost_x_testing = gradient_cost_x[0].eval(feed_dict=feed_testing)

            # save
            params_dic = {}
            hyper_params_dic = {}
            output_dic = {}
            output_dic['u_training'] = u_training
            output_dic['u_testing'] = u_testing
            output_dic['prob_training'] = prob_training
            output_dic['prob_testing'] = prob_testing

            output_dic['cost_training'] = cost_training
            output_dic['cost_testing'] = cost_testing
            output_dic['accuracy_training'] = accuracy_training
            output_dic['accuracy_testing'] = accuracy_testing
            output_dic['log_loss_training'] = log_loss_training
            output_dic['log_loss_testing'] = log_loss_testing

            output_dic['gradient_u1_x_training'] = gradient_u1_x_training
            output_dic['gradient_u1_x_testing'] = gradient_u1_x_testing
            output_dic['gradient_prob1_x_training'] = gradient_prob1_x_training
            output_dic['gradient_prob1_x_testing'] = gradient_prob1_x_testing
            output_dic['gradient_cost_x_training'] = gradient_cost_x_training
            output_dic['gradient_cost_x_testing'] = gradient_cost_x_testing

        elif restore == False:
            # case 3. train and save the estimated model
            for i in range(n_epoches):
                x0_training_batch, x1_training_batch, p0_training_batch, p1_training_batch, z_training_batch, y_training_batch = \
                    obtain_mini_batch_pt(x0_training, x1_training, p0_training, p1_training, z_training, y_training,
                                         n_mini_batch=100)

                sess.run(training_op, feed_dict={x0: x0_training_batch, x1: x1_training_batch, p0: p0_training_batch,
                                                 p1: p1_training_batch, z: z_training_batch, y: y_training_batch})

                if i % 1000 == 0:
                    print("Iterations", i, "Cost = ", cost.eval(feed_dict=feed_training))
                if i % 20 == 0:
                    current_cost_training = cost.eval(feed_dict=feed_training)
                    current_cost_testing = cost.eval(feed_dict=feed_testing)
                    current_accuracy_training = accuracy.eval(feed_dict=feed_training)
                    current_accuracy_testing = accuracy.eval(feed_dict=feed_testing)
                    current_log_loss_training = log_loss.eval(feed_dict=feed_training)
                    current_log_loss_testing = log_loss.eval(feed_dict=feed_testing)

                    cost_training.append(current_cost_training)
                    cost_testing.append(current_cost_testing)
                    accuracy_training.append(current_accuracy_training)
                    accuracy_testing.append(current_accuracy_testing)
                    log_loss_training.append(current_log_loss_training)
                    log_loss_testing.append(current_log_loss_testing)

            # save the model
            if accuracy_testing[-1] >= current_best_acc:
                saver.save(sess, "tmp/"+MODEL_NAME+".ckpt")
                current_best_acc = accuracy_testing[-1]
            #
            u_training = u.eval(feed_dict=feed_training)
            u_testing = u.eval(feed_dict=feed_testing)
            prob_training = prob.eval(feed_dict=feed_training)
            prob_testing = prob.eval(feed_dict=feed_testing)
            #
            gradient_u1_x_training = gradient_u1_x[0].eval(feed_dict=feed_training)
            gradient_u1_x_testing = gradient_u1_x[0].eval(feed_dict=feed_testing)
            gradient_prob1_x_training = gradient_prob1_x[0].eval(feed_dict=feed_training)
            gradient_prob1_x_testing = gradient_prob1_x[0].eval(feed_dict=feed_testing)
            gradient_cost_x_training = gradient_cost_x[0].eval(feed_dict=feed_training)
            gradient_cost_x_testing = gradient_cost_x[0].eval(feed_dict=feed_testing)

            # outputs
            params_dic = {}
            hyper_params_dic = {}
            output_dic = {}

            output_dic['cost_training'] = cost_training
            output_dic['cost_testing'] = cost_testing
            output_dic['accuracy_training'] = accuracy_training
            output_dic['accuracy_testing'] = accuracy_testing
            output_dic['log_loss_training'] = log_loss_training
            output_dic['log_loss_testing'] = log_loss_testing

            output_dic['u_training'] = u_training
            output_dic['u_testing'] = u_testing
            output_dic['prob_training'] = prob_training
            output_dic['prob_testing'] = prob_testing

            output_dic['gradient_u1_x_training'] = gradient_u1_x_training
            output_dic['gradient_u1_x_testing'] = gradient_u1_x_testing
            output_dic['gradient_prob1_x_training'] = gradient_prob1_x_training
            output_dic['gradient_prob1_x_testing'] = gradient_prob1_x_testing
            output_dic['gradient_cost_x_training'] = gradient_cost_x_training
            output_dic['gradient_cost_x_testing'] = gradient_cost_x_testing

            output_dic['current_best_acc'] = current_best_acc

    return params_dic, hyper_params_dic, output_dic




def est_pt_resnet_simultaneous_use_delta(data, penalty_const, MODEL_NAME, restore=False, n_epoches=10000, n_mini_batch=100, M=3,
                  n_hidden=100, simul_data=None, Dropout=False, Dropout_rate=0.01, BN=False, l2_regu = 1e-50, x_dim=15, K=2, D=7, current_best_acc = 0):
    '''
    estimate pt resnet model
    '''
    x0_training, x1_training, p0_training, p1_training, z_training, y_training = data['training']
    x0_testing, x1_testing, p0_testing, p1_testing, z_testing, y_testing = data['testing']
    #
    Dz = z_training.shape[1]
    # build models
    tf.reset_default_graph()
    z = tf.placeholder(dtype=tf.float32, shape=(None, Dz), name='z')
    x0 = tf.placeholder(dtype=tf.float32, shape=(None, 2), name='x0')
    x1 = tf.placeholder(dtype=tf.float32, shape=(None, 2), name='x1')
    p0 = tf.placeholder(dtype=tf.float32, shape=(None, 2), name='p0')
    p1 = tf.placeholder(dtype=tf.float32, shape=(None, 2), name='p1')
    x = tf.concat([x0, x1, p0, p1, z], axis=1, name='x')
    y = tf.placeholder(dtype=tf.int64, shape=(None), name='y')

    gamma_param_0 = tf.Variable(tf.random_normal([1]), dtype = tf.float32)
    beta_param_0 = tf.Variable(tf.random_normal([1]), dtype = tf.float32)
    lambda_param_0 = tf.Variable(tf.random_normal([1]), dtype = tf.float32)
    ref_param_0 = tf.Variable(tf.random_normal([1]), dtype = tf.float32)
    gamma_param = tf.Variable(tf.random_normal([Dz, 1]), dtype = tf.float32)
    beta_param = tf.Variable(tf.random_normal([Dz, 1]), dtype = tf.float32)
    lambda_param = tf.Variable(tf.random_normal([Dz, 1]), dtype = tf.float32)
    ref_param = tf.Variable(tf.random_normal([Dz, 1]), dtype = tf.float32)

    gamma_initial = tf.add(tf.matmul(z, gamma_param), gamma_param_0)
    beta_initial = tf.add(tf.matmul(z, beta_param), beta_param_0)
    lambda_initial = tf.add(tf.matmul(z, lambda_param), lambda_param_0)
    ref_initial = tf.add(tf.matmul(z, ref_param), ref_param_0)

    gamma = tf.nn.sigmoid(gamma_initial)
    beta = tf.nn.sigmoid(beta_initial)
    lambda_ = tf.nn.sigmoid(lambda_initial) * 4 + 1  # hence constrain lambda_ between 0 and 5
    ref = ref_initial * 1.0

    ### PT part
    # update x1 and x0 by ref pt
    x1_ref_dep = tf.subtract(x1, ref)
    x0_ref_dep = tf.subtract(x0, ref)
    # create zeros
    zeros = tf.zeros_like(x0, dtype=tf.float32)
    pi_1 = tf.exp(- tf.pow((-tf.log(p1)), beta))
    pi_0 = tf.exp(- tf.pow((-tf.log(p0)), beta))
    # compute u_0
    # u_0 postive
    v_0_pos = tf.pow(tf.where(x0_ref_dep > 0, x0_ref_dep, zeros), gamma)
    u_0_pos = tf.multiply(v_0_pos, pi_0)
    # u_0 negative
    v_0_neg = tf.multiply(-lambda_, tf.pow(-tf.where(x0_ref_dep < 0, x0_ref_dep, zeros), gamma))
    u_0_neg = tf.multiply(v_0_neg, pi_0)
    u_0 = tf.reduce_sum(tf.add(u_0_pos, u_0_neg), axis=1, keepdims=True)
    # compute u_1
    # u_1 postive
    v_1_pos = tf.pow(tf.where(x1_ref_dep > 0, x1_ref_dep, zeros), gamma)
    u_1_pos = tf.multiply(v_1_pos, pi_1)
    # u_1 negative
    v_1_neg = tf.multiply(-lambda_, tf.pow(-tf.where(x1_ref_dep < 0, x1_ref_dep, zeros), gamma))
    u_1_neg = tf.multiply(v_1_neg, pi_1)
    u_1 = tf.reduce_sum(tf.add(u_1_pos, u_1_neg), axis=1, keepdims=True)
    #
    u_pt = tf.concat([u_0, u_1], axis=1)

    ### DNN part
    hidden = x
    for i in range(M):
        hidden = standard_hidden_layer(hidden, n_hidden, BN, Dropout, Dropout_rate)
    u_dnn = tf.layers.dense(hidden, K, name='u_dnn')

    # combine the PT and DNN parts
    u = tf.add((1-penalty_const) * u_pt, penalty_const * u_dnn)
    prob = tf.nn.softmax(u)

    # gradient info: only alternative 1 respect to inputs
    gradient_u1_x = [tf.concat(
        [tf.gradients(u[:, 1], [x0])[0], tf.gradients(u[:, 1], [x1])[0], tf.gradients(u[:, 1], [p0])[0],
         tf.gradients(u[:, 1], [p1])[0], tf.gradients(u[:, 1], [z])[0]], axis=1)]
    gradient_prob1_x = [tf.concat(
        [tf.gradients(prob[:, 1], [x0])[0], tf.gradients(prob[:, 1], [x1])[0], tf.gradients(prob[:, 1], [p0])[0],
         tf.gradients(prob[:, 1], [p1])[0], tf.gradients(prob[:, 1], [z])[0]], axis=1)]

    # regularization l2
    l2_regularizer = tf.contrib.layers.l2_regularizer(scale=l2_regu, scope=None)
    vars_ = tf.trainable_variables()
    regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, vars_)

    #
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=u, labels=y), name='cost')
    cost += regularization_penalty
    log_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=u, labels=y), name='log_loss')
    optimizer = tf.train.AdamOptimizer()  # opt: better than simple gradient descents: always converge to stable numbers.
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate) # opt objective
    training_op = optimizer.minimize(cost)  # minimize the opt objective

    #
    gradient_cost_x = [tf.concat([tf.gradients(cost, [x0])[0], tf.gradients(cost, [x1])[0], tf.gradients(cost, [p0])[0],
                                  tf.gradients(cost, [p1])[0], tf.gradients(cost, [z])[0]], axis=1)]

    # eval params
    correct = tf.nn.in_top_k(u, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    cost_training = []
    cost_testing = []
    accuracy_training = []
    accuracy_testing = []
    log_loss_training = []
    log_loss_testing = []

    with tf.Session() as sess:
        # always run this to train the model
        init.run()
        feed_training = {x0: x0_training, x1: x1_training, p0: p0_training, p1: p1_training, z: z_training,
                         y: y_training}
        feed_testing = {x0: x0_testing, x1: x1_testing, p0: p0_testing, p1: p1_testing, z: z_testing, y: y_testing}

        if restore == True:
            # case 1. restore models and evaluate key values
            saver.restore(sess, "tmp/" + MODEL_NAME + ".ckpt")
            u_training = u.eval(feed_dict=feed_training)
            u_testing = u.eval(feed_dict=feed_testing)
            prob_training = prob.eval(feed_dict=feed_training)
            prob_testing = prob.eval(feed_dict=feed_testing)
            #
            current_cost_training = cost.eval(feed_dict=feed_training)
            current_cost_testing = cost.eval(feed_dict=feed_testing)
            current_accuracy_training = accuracy.eval(feed_dict=feed_training)
            current_accuracy_testing = accuracy.eval(feed_dict=feed_testing)
            current_log_loss_training = log_loss.eval(feed_dict=feed_training)
            current_log_loss_testing = log_loss.eval(feed_dict=feed_testing)
            #
            log_loss_training.append(current_log_loss_training)
            log_loss_testing.append(current_log_loss_testing)
            cost_training.append(current_cost_training)
            cost_testing.append(current_cost_testing)
            accuracy_training.append(current_accuracy_training)
            accuracy_testing.append(current_accuracy_testing)
            #
            gradient_u1_x_training = gradient_u1_x[0].eval(feed_dict=feed_training)
            gradient_u1_x_testing = gradient_u1_x[0].eval(feed_dict=feed_testing)
            gradient_prob1_x_training = gradient_prob1_x[0].eval(feed_dict=feed_training)
            gradient_prob1_x_testing = gradient_prob1_x[0].eval(feed_dict=feed_testing)
            gradient_cost_x_training = gradient_cost_x[0].eval(feed_dict=feed_training)
            gradient_cost_x_testing = gradient_cost_x[0].eval(feed_dict=feed_testing)

            # save
            params_dic = {}
            hyper_params_dic = {}
            output_dic = {}
            output_dic['u_training'] = u_training
            output_dic['u_testing'] = u_testing
            output_dic['prob_training'] = prob_training
            output_dic['prob_testing'] = prob_testing

            output_dic['cost_training'] = cost_training
            output_dic['cost_testing'] = cost_testing
            output_dic['accuracy_training'] = accuracy_training
            output_dic['accuracy_testing'] = accuracy_testing
            output_dic['log_loss_training'] = log_loss_training
            output_dic['log_loss_testing'] = log_loss_testing

            output_dic['gradient_u1_x_training'] = gradient_u1_x_training
            output_dic['gradient_u1_x_testing'] = gradient_u1_x_testing
            output_dic['gradient_prob1_x_training'] = gradient_prob1_x_training
            output_dic['gradient_prob1_x_testing'] = gradient_prob1_x_testing
            output_dic['gradient_cost_x_training'] = gradient_cost_x_training
            output_dic['gradient_cost_x_testing'] = gradient_cost_x_testing

        elif restore == False:
            # case 3. train and save the estimated model
            for i in range(n_epoches):
                x0_training_batch, x1_training_batch, p0_training_batch, p1_training_batch, z_training_batch, y_training_batch = \
                    obtain_mini_batch_pt(x0_training, x1_training, p0_training, p1_training, z_training, y_training,
                                         n_mini_batch=100)

                sess.run(training_op, feed_dict={x0: x0_training_batch, x1: x1_training_batch, p0: p0_training_batch,
                                                 p1: p1_training_batch, z: z_training_batch, y: y_training_batch})

                if i % 1000 == 0:
                    print("Iterations", i, "Cost = ", cost.eval(feed_dict=feed_training))
                if i % 20 == 0:
                    current_cost_training = cost.eval(feed_dict=feed_training)
                    current_cost_testing = cost.eval(feed_dict=feed_testing)
                    current_accuracy_training = accuracy.eval(feed_dict=feed_training)
                    current_accuracy_testing = accuracy.eval(feed_dict=feed_testing)
                    current_log_loss_training = log_loss.eval(feed_dict=feed_training)
                    current_log_loss_testing = log_loss.eval(feed_dict=feed_testing)

                    cost_training.append(current_cost_training)
                    cost_testing.append(current_cost_testing)
                    accuracy_training.append(current_accuracy_training)
                    accuracy_testing.append(current_accuracy_testing)
                    log_loss_training.append(current_log_loss_training)
                    log_loss_testing.append(current_log_loss_testing)

            # save the model
            if accuracy_testing[-1] >= current_best_acc:
                saver.save(sess, "tmp/"+MODEL_NAME+".ckpt")
                current_best_acc = accuracy_testing[-1]
            #
            u_training = u.eval(feed_dict=feed_training)
            u_testing = u.eval(feed_dict=feed_testing)
            prob_training = prob.eval(feed_dict=feed_training)
            prob_testing = prob.eval(feed_dict=feed_testing)
            #
            gradient_u1_x_training = gradient_u1_x[0].eval(feed_dict=feed_training)
            gradient_u1_x_testing = gradient_u1_x[0].eval(feed_dict=feed_testing)
            gradient_prob1_x_training = gradient_prob1_x[0].eval(feed_dict=feed_training)
            gradient_prob1_x_testing = gradient_prob1_x[0].eval(feed_dict=feed_testing)
            gradient_cost_x_training = gradient_cost_x[0].eval(feed_dict=feed_training)
            gradient_cost_x_testing = gradient_cost_x[0].eval(feed_dict=feed_testing)

            # outputs
            params_dic = {}
            hyper_params_dic = {}
            output_dic = {}

            output_dic['cost_training'] = cost_training
            output_dic['cost_testing'] = cost_testing
            output_dic['accuracy_training'] = accuracy_training
            output_dic['accuracy_testing'] = accuracy_testing
            output_dic['log_loss_training'] = log_loss_training
            output_dic['log_loss_testing'] = log_loss_testing

            output_dic['u_training'] = u_training
            output_dic['u_testing'] = u_testing
            output_dic['prob_training'] = prob_training
            output_dic['prob_testing'] = prob_testing

            output_dic['gradient_u1_x_training'] = gradient_u1_x_training
            output_dic['gradient_u1_x_testing'] = gradient_u1_x_testing
            output_dic['gradient_prob1_x_training'] = gradient_prob1_x_training
            output_dic['gradient_prob1_x_testing'] = gradient_prob1_x_testing
            output_dic['gradient_cost_x_training'] = gradient_cost_x_training
            output_dic['gradient_cost_x_testing'] = gradient_cost_x_testing

            output_dic['current_best_acc'] = current_best_acc

    return params_dic, hyper_params_dic, output_dic

def obtain_mini_batch_hd(x0_training,x1_training,t1_training,z_training,
                         y_training,n_mini_batch):
    '''
    Return mini_batch
    '''
    N = x0_training.shape[0]
    index = np.random.choice(N, size = n_mini_batch)
     
    x0_training_batch=x0_training[index, :]
    x1_training_batch=x1_training[index, :]
    t1_training_batch=t1_training[index, :]
    z_training_batch=z_training[index, :]
    y_training_batch=y_training[index]
    return x0_training_batch,x1_training_batch,t1_training_batch,z_training_batch,y_training_batch



def est_hd(data, MODEL_NAME, restore = False, n_epoches = 10000, current_best_acc = 0):
    ###
    tf.reset_default_graph()
    
    x0_training, x1_training, t1_training, z_training, y_training = data['training']
    x0_testing, x1_testing, t1_testing, z_testing, y_testing = data['testing']

    # change data type
    x0_training = x0_training.astype('float32')
    x1_training = x1_training.astype('float32')
    t1_training = t1_training.astype('float32')
    z_training = z_training.astype('float32')
    x0_testing = x0_testing.astype('float32')
    x1_testing = x1_testing.astype('float32')
    t1_testing = t1_testing.astype('float32')
    z_testing = z_testing.astype('float32')
    y_training = y_training.astype('int32')
    y_testing = y_testing.astype('int32')


    D = z_training.shape[1]
    
    z = tf.placeholder(dtype = tf.float32, shape = (None, D), name = 'z')
    x0 = tf.placeholder(dtype = tf.float32, shape = (None, 1), name = 'x0')
    x1 = tf.placeholder(dtype = tf.float32, shape = (None, 1), name = 'x1')
    t1 = tf.placeholder(dtype = tf.float32, shape = (None, 1), name = 't1')
    y = tf.placeholder(dtype = tf.int32, shape = (None), name = 'y')
    x = tf.concat([x0, x1, t1], axis = 1)
    
    # 
    r_param_0 = tf.Variable(tf.random_normal([1]), dtype = tf.float32)
    b_param_0 = tf.Variable(tf.random_normal([1]), dtype = tf.float32)
    r_param = tf.Variable(tf.random_normal([D, 1]), dtype = tf.float32)
    b_param = tf.Variable(tf.random_normal([D, 1]), dtype = tf.float32)
    
    r_param_initial = tf.add(tf.matmul(z, r_param), r_param_0)
    b_param_initial = tf.add(tf.matmul(z, b_param), b_param_0)
    
    r = tf.sigmoid(r_param_initial) * 0.1 # reduce to 0 - 0.1 
    b = tf.sigmoid(b_param_initial)
    
    # utilities
    u0 = tf.identity(x0, name = 'u0')
    u1_ = tf.multiply(tf.multiply(x1, b),tf.exp(-tf.multiply(r, t1)))
    u1 = tf.identity(u1_, name = 'u1')
    
    # 
    u = tf.concat([u0,u1], axis = 1)
    prob = tf.nn.softmax(u)
    
    # gradient info: here is alternative 0 respect to inputs
    gradient_u1_x = [tf.concat([tf.gradients(u[:,1], [x0])[0], tf.gradients(u[:,1], [x1])[0], tf.gradients(u[:,1], [t1])[0]], axis = 1)]
    gradient_prob1_x = [tf.concat([tf.gradients(prob[:,1], [x0])[0], tf.gradients(prob[:,1], [x1])[0], tf.gradients(prob[:,1], [t1])[0]], axis = 1)]
    # 
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = u, labels = y), name = 'cost')
    log_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = u, labels = y), name = 'log_loss')

    #
    gradient_cost_x = [tf.concat([tf.gradients(cost, [x0])[0], tf.gradients(cost, [x1])[0], tf.gradients(cost, [t1])[0], tf.gradients(cost, [z])[0]], axis = 1)]
    
    # eval params
    correct = tf.nn.in_top_k(u, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    
    optimizer = tf.train.AdamOptimizer() # default rate 0.001; best so far: 0.01
    training_op = optimizer.minimize(cost) # minimize the opt objective
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
    #n_mini_batch = 500
    cost_training=[]
    cost_testing=[]
    accuracy_training=[]
    accuracy_testing=[]
    log_loss_training=[]
    log_loss_testing=[]
            
    with tf.Session() as sess:
        # always run this to train the model
        init.run()
        feed_training = {x0:x0_training, x1:x1_training, t1:t1_training, z:z_training, y:y_training}
        feed_testing = {x0:x0_testing, x1:x1_testing, t1:t1_testing, z:z_testing, y:y_testing}
    
        if restore == True:
            saver.restore(sess, "tmp/"+MODEL_NAME+".ckpt")
            u_training = u.eval(feed_dict = feed_training)
            u_testing = u.eval(feed_dict = feed_testing)
            prob_training = prob.eval(feed_dict = feed_training)
            prob_testing = prob.eval(feed_dict = feed_testing)
            # 
            current_cost_training = cost.eval(feed_dict=feed_training)
            current_cost_testing = cost.eval(feed_dict=feed_testing)
            current_accuracy_training = accuracy.eval(feed_dict=feed_training)
            current_accuracy_testing = accuracy.eval(feed_dict=feed_testing)
            current_log_loss_training = log_loss.eval(feed_dict = feed_training)
            current_log_loss_testing = log_loss.eval(feed_dict = feed_testing)
            #
            log_loss_training.append(current_log_loss_training)
            log_loss_testing.append(current_log_loss_testing)
            cost_training.append(current_cost_training)
            cost_testing.append(current_cost_testing)
            accuracy_training.append(current_accuracy_training)
            accuracy_testing.append(current_accuracy_testing)
            # 
            gradient_u1_x_training=gradient_u1_x[0].eval(feed_dict = feed_training)
            gradient_u1_x_testing=gradient_u1_x[0].eval(feed_dict = feed_testing)
            gradient_prob1_x_training=gradient_prob1_x[0].eval(feed_dict = feed_training)
            gradient_prob1_x_testing=gradient_prob1_x[0].eval(feed_dict = feed_testing)
            gradient_cost_x_training=gradient_cost_x[0].eval(feed_dict = feed_training)          
            gradient_cost_x_testing=gradient_cost_x[0].eval(feed_dict = feed_testing)
            # save
            params_dic = {}
            hyper_params_dic = {}
            output_dic = {}
            output_dic['u_training'] = u_training
            output_dic['u_testing'] = u_testing
            output_dic['prob_training'] = prob_training
            output_dic['prob_testing'] = prob_testing

            output_dic['cost_training'] = cost_training
            output_dic['cost_testing'] = cost_testing
            output_dic['accuracy_training'] = accuracy_training
            output_dic['accuracy_testing'] = accuracy_testing
            output_dic['log_loss_training'] = log_loss_training
            output_dic['log_loss_testing'] = log_loss_testing
            
            output_dic['gradient_u1_x_training'] = gradient_u1_x_training
            output_dic['gradient_u1_x_testing'] = gradient_u1_x_testing
            output_dic['gradient_prob1_x_training'] = gradient_prob1_x_training
            output_dic['gradient_prob1_x_testing'] = gradient_prob1_x_testing
            output_dic['gradient_cost_x_training'] = gradient_cost_x_training
            output_dic['gradient_cost_x_testing'] = gradient_cost_x_testing
    
        elif restore == False:
            for i in range(n_epoches):
                x0_training_batch,x1_training_batch,t1_training_batch,z_training_batch,y_training_batch = \
                    obtain_mini_batch_hd(x0_training,x1_training,t1_training,z_training,y_training,n_mini_batch=100)
                
                # training
                sess.run(training_op, feed_dict = {x0:x0_training_batch, x1:x1_training_batch, t1:t1_training_batch, z:z_training_batch, y:y_training_batch})
                if i % 1000 == 0:
                    print("Iterations", i, "Cost = ", cost.eval(feed_dict = feed_training))
                if i % 20 == 0:
                    current_cost_training = cost.eval(feed_dict=feed_training)
                    current_cost_testing = cost.eval(feed_dict=feed_testing)
                    current_accuracy_training = accuracy.eval(feed_dict=feed_training)
                    current_accuracy_testing = accuracy.eval(feed_dict=feed_testing)
                    current_log_loss_training = log_loss.eval(feed_dict=feed_training)
                    current_log_loss_testing = log_loss.eval(feed_dict=feed_testing)
    
                    cost_training.append(current_cost_training)
                    cost_testing.append(current_cost_testing)
                    accuracy_training.append(current_accuracy_training)
                    accuracy_testing.append(current_accuracy_testing)
                    log_loss_training.append(current_log_loss_training)
                    log_loss_testing.append(current_log_loss_testing)
    
            # save the model
            if accuracy_testing[-1] >= current_best_acc:
                saver.save(sess, "tmp/"+MODEL_NAME+".ckpt")
                current_best_acc = accuracy_testing[-1]
            #        
            u_training = u.eval(feed_dict = feed_training)
            u_testing = u.eval(feed_dict = feed_testing)
            prob_training = prob.eval(feed_dict = feed_training)
            prob_testing = prob.eval(feed_dict = feed_testing)
            #
            gradient_u1_x_training=gradient_u1_x[0].eval(feed_dict = feed_training)
            gradient_u1_x_testing=gradient_u1_x[0].eval(feed_dict = feed_testing)
            gradient_prob1_x_training=gradient_prob1_x[0].eval(feed_dict = feed_training)
            gradient_prob1_x_testing=gradient_prob1_x[0].eval(feed_dict = feed_testing)
            gradient_cost_x_training=gradient_cost_x[0].eval(feed_dict = feed_training)            
            gradient_cost_x_testing=gradient_cost_x[0].eval(feed_dict = feed_testing)
            #
            r_param_0_value=r_param_0.eval()
            b_param_0_value=b_param_0.eval()
            r_param_value = r_param.eval()
            b_param_value = b_param.eval()
            
            r_value=r.eval(feed_dict = feed_training)
            b_value=b.eval(feed_dict = feed_training)
    
            # outputs
            params_dic = {}
            hyper_params_dic = {}
            output_dic = {}
            
            output_dic['cost_training'] = cost_training
            output_dic['cost_testing'] = cost_testing
            output_dic['accuracy_training'] = accuracy_training
            output_dic['accuracy_testing'] = accuracy_testing
            output_dic['log_loss_training'] = log_loss_training
            output_dic['log_loss_testing'] = log_loss_testing
    
            output_dic['u_training'] = u_training
            output_dic['u_testing'] = u_testing
            output_dic['prob_training'] = prob_training
            output_dic['prob_testing'] = prob_testing
            
            output_dic['gradient_u1_x_training'] = gradient_u1_x_training
            output_dic['gradient_u1_x_testing'] = gradient_u1_x_testing
            output_dic['gradient_prob1_x_training'] = gradient_prob1_x_training
            output_dic['gradient_prob1_x_testing'] = gradient_prob1_x_testing
            output_dic['gradient_cost_x_training'] = gradient_cost_x_training
            output_dic['gradient_cost_x_testing'] = gradient_cost_x_testing

            output_dic['current_best_acc'] = current_best_acc
            
            params_dic['r_param_0']=r_param_0_value
            params_dic['b_param_0']=b_param_0_value
            params_dic['r_param']=r_param_value
            params_dic['b_param']=b_param_value
            params_dic['r']=r_value
            params_dic['b']=b_value
    return params_dic,hyper_params_dic,output_dic
        


def est_hd_resnet(data, hd_param_dic, penalty_const, MODEL_NAME, restore = False, n_epoches = 10000, n_mini_batch=100,
                  M=3, n_hidden=100, simul_data = None, Dropout = False, Dropout_rate = 0.01, BN = False, x_dim = 15,
                  K = 2, D = 7, current_best_acc = 0):
    ###
    tf.reset_default_graph()
    
    x0_training, x1_training, t1_training, z_training, y_training = data['training']
    x0_testing, x1_testing, t1_testing, z_testing, y_testing = data['testing']

    # change data type
    x0_training = x0_training.astype('float32')
    x1_training = x1_training.astype('float32')
    t1_training = t1_training.astype('float32')
    z_training = z_training.astype('float32')
    x0_testing = x0_testing.astype('float32')
    x1_testing = x1_testing.astype('float32')
    t1_testing = t1_testing.astype('float32')
    z_testing = z_testing.astype('float32')
    y_training = y_training.astype('int32')
    y_testing = y_testing.astype('int32')


    D = z_training.shape[1]
    
    z = tf.placeholder(dtype = tf.float32, shape = (None, D), name = 'z')
    x0 = tf.placeholder(dtype = tf.float32, shape = (None, 1), name = 'x0')
    x1 = tf.placeholder(dtype = tf.float32, shape = (None, 1), name = 'x1')
    t1 = tf.placeholder(dtype = tf.float32, shape = (None, 1), name = 't1')
    y = tf.placeholder(dtype = tf.int32, shape = (None), name = 'y')
    x = tf.concat([x0, x1, t1], axis = 1)
    
    # 
    r_param_0 = hd_param_dic['r_param_0']
    b_param_0 = hd_param_dic['b_param_0']
    r_param = hd_param_dic['r_param']
    b_param = hd_param_dic['b_param']
    
    r_param_initial = tf.add(tf.matmul(z, r_param), r_param_0)
    b_param_initial = tf.add(tf.matmul(z, b_param), b_param_0)
    
    r = tf.sigmoid(r_param_initial) * 0.1 # reduce to 0 - 0.1 
    b = tf.sigmoid(b_param_initial)
    
    ### hd utilities
    u0 = tf.identity(x0, name = 'u0')
    u1_ = tf.multiply(tf.multiply(x1, b),tf.exp(-tf.multiply(r, t1)))
    u1 = tf.identity(u1_, name = 'u1')    
    u_hd = tf.concat([u0,u1], axis = 1)

    ### dnn utilities
    hidden = x
    for i in range(M):
        hidden = standard_hidden_layer(hidden, n_hidden, BN, Dropout, Dropout_rate)
    u_dnn = tf.layers.dense(hidden, K, name = 'u_dnn')
    
    u = tf.add(u_hd, u_dnn)
    prob = tf.nn.softmax(u)
    
    # gradient info: here is alternative 0 respect to inputs
    gradient_u1_x = [tf.concat([tf.gradients(u[:,1], [x0])[0], tf.gradients(u[:,1], [x1])[0], tf.gradients(u[:,1], [t1])[0]], axis = 1)]
    gradient_prob1_x = [tf.concat([tf.gradients(prob[:,1], [x0])[0], tf.gradients(prob[:,1], [x1])[0], tf.gradients(prob[:,1], [t1])[0]], axis = 1)]

    # regularization l2
    l2_regularizer = tf.contrib.layers.l2_regularizer(scale=penalty_const, scope=None)
    vars_ = tf.trainable_variables()
    regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, vars_)
    
    ###
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = u, labels = y), name = 'cost')
    cost += regularization_penalty
    log_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = u, labels = y), name = 'log_loss')

    # 
    gradient_cost_x = [tf.concat([tf.gradients(cost, [x0])[0], tf.gradients(cost, [x1])[0], tf.gradients(cost, [t1])[0], tf.gradients(cost, [z])[0]], axis = 1)]
    
    # eval params
    correct = tf.nn.in_top_k(u, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    
    optimizer = tf.train.AdamOptimizer() # default rate 0.001; best so far: 0.01
    training_op = optimizer.minimize(cost) # minimize the opt objective
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
    #n_mini_batch = 500
    cost_training=[]
    cost_testing=[]
    accuracy_training=[]
    accuracy_testing=[]
    log_loss_training=[]
    log_loss_testing=[]
            
    with tf.Session() as sess:
        # always run this to train the model
        init.run()
        feed_training = {x0:x0_training, x1:x1_training, t1:t1_training, z:z_training, y:y_training}
        feed_testing = {x0:x0_testing, x1:x1_testing, t1:t1_testing, z:z_testing, y:y_testing}
    
        if restore == True:
            saver.restore(sess, "tmp/"+MODEL_NAME+".ckpt")
            u_training = u.eval(feed_dict = feed_training)
            u_testing = u.eval(feed_dict = feed_testing)
            prob_training = prob.eval(feed_dict = feed_training)
            prob_testing = prob.eval(feed_dict = feed_testing)
            # 
            current_cost_training = cost.eval(feed_dict=feed_training)
            current_cost_testing = cost.eval(feed_dict=feed_testing)
            current_accuracy_training = accuracy.eval(feed_dict=feed_training)
            current_accuracy_testing = accuracy.eval(feed_dict=feed_testing)
            current_log_loss_training = log_loss.eval(feed_dict = feed_training)
            current_log_loss_testing = log_loss.eval(feed_dict = feed_testing)
            #
            log_loss_training.append(current_log_loss_training)
            log_loss_testing.append(current_log_loss_testing)
            cost_training.append(current_cost_training)
            cost_testing.append(current_cost_testing)
            accuracy_training.append(current_accuracy_training)
            accuracy_testing.append(current_accuracy_testing)
            #            
            gradient_u1_x_training=gradient_u1_x[0].eval(feed_dict = feed_training)
            gradient_u1_x_testing=gradient_u1_x[0].eval(feed_dict = feed_testing)
            gradient_prob1_x_training=gradient_prob1_x[0].eval(feed_dict = feed_training)
            gradient_prob1_x_testing=gradient_prob1_x[0].eval(feed_dict = feed_testing)
            gradient_cost_x_training=gradient_cost_x[0].eval(feed_dict = feed_training)          
            gradient_cost_x_testing=gradient_cost_x[0].eval(feed_dict = feed_testing)

            # save
            params_dic = {}
            hyper_params_dic = {}
            output_dic = {}
            output_dic['u_training'] = u_training
            output_dic['u_testing'] = u_testing
            output_dic['prob_training'] = prob_training
            output_dic['prob_testing'] = prob_testing

            output_dic['cost_training'] = cost_training
            output_dic['cost_testing'] = cost_testing
            output_dic['accuracy_training'] = accuracy_training
            output_dic['accuracy_testing'] = accuracy_testing
            output_dic['log_loss_training'] = log_loss_training
            output_dic['log_loss_testing'] = log_loss_testing
            
            output_dic['gradient_u1_x_training'] = gradient_u1_x_training
            output_dic['gradient_u1_x_testing'] = gradient_u1_x_testing
            output_dic['gradient_prob1_x_training'] = gradient_prob1_x_training
            output_dic['gradient_prob1_x_testing'] = gradient_prob1_x_testing
            output_dic['gradient_cost_x_training'] = gradient_cost_x_training
            output_dic['gradient_cost_x_testing'] = gradient_cost_x_testing
    
        elif restore == False:
            for i in range(n_epoches):
                x0_training_batch,x1_training_batch,t1_training_batch,z_training_batch,y_training_batch = \
                    obtain_mini_batch_hd(x0_training,x1_training,t1_training,z_training,y_training,n_mini_batch=100)
                
                # training
                sess.run(training_op, feed_dict = {x0:x0_training_batch, x1:x1_training_batch, t1:t1_training_batch, z:z_training_batch, y:y_training_batch})
                if i % 1000 == 0:
                    print("Iterations", i, "Cost = ", cost.eval(feed_dict = feed_training))
                if i % 20 == 0:
                    current_cost_training = cost.eval(feed_dict=feed_training)
                    current_cost_testing = cost.eval(feed_dict=feed_testing)
                    current_accuracy_training = accuracy.eval(feed_dict=feed_training)
                    current_accuracy_testing = accuracy.eval(feed_dict=feed_testing)
                    current_log_loss_training = log_loss.eval(feed_dict=feed_training)
                    current_log_loss_testing = log_loss.eval(feed_dict=feed_testing)
    
                    cost_training.append(current_cost_training)
                    cost_testing.append(current_cost_testing)
                    accuracy_training.append(current_accuracy_training)
                    accuracy_testing.append(current_accuracy_testing)
                    log_loss_training.append(current_log_loss_training)
                    log_loss_testing.append(current_log_loss_testing)
    
            # save the model
            if accuracy_testing[-1] >= current_best_acc:
                saver.save(sess, "tmp/"+MODEL_NAME+".ckpt")
                current_best_acc = accuracy_testing[-1]
            #        
            u_training = u.eval(feed_dict = feed_training)
            u_testing = u.eval(feed_dict = feed_testing)
            prob_training = prob.eval(feed_dict = feed_training)
            prob_testing = prob.eval(feed_dict = feed_testing)
            #
            gradient_u1_x_training=gradient_u1_x[0].eval(feed_dict = feed_training)
            gradient_u1_x_testing=gradient_u1_x[0].eval(feed_dict = feed_testing)
            gradient_prob1_x_training=gradient_prob1_x[0].eval(feed_dict = feed_training)
            gradient_prob1_x_testing=gradient_prob1_x[0].eval(feed_dict = feed_testing)
            gradient_cost_x_training=gradient_cost_x[0].eval(feed_dict = feed_training)            
            gradient_cost_x_testing=gradient_cost_x[0].eval(feed_dict = feed_testing)            

            # outputs
            params_dic = {}
            hyper_params_dic = {}
            output_dic = {}
            
            output_dic['cost_training'] = cost_training
            output_dic['cost_testing'] = cost_testing
            output_dic['accuracy_training'] = accuracy_training
            output_dic['accuracy_testing'] = accuracy_testing
            output_dic['log_loss_training'] = log_loss_training
            output_dic['log_loss_testing'] = log_loss_testing
    
            output_dic['u_training'] = u_training
            output_dic['u_testing'] = u_testing
            output_dic['prob_training'] = prob_training
            output_dic['prob_testing'] = prob_testing
            
            output_dic['gradient_u1_x_training'] = gradient_u1_x_training
            output_dic['gradient_u1_x_testing'] = gradient_u1_x_testing
            output_dic['gradient_prob1_x_training'] = gradient_prob1_x_training
            output_dic['gradient_prob1_x_testing'] = gradient_prob1_x_testing            
            output_dic['gradient_cost_x_training'] = gradient_cost_x_training
            output_dic['gradient_cost_x_testing'] = gradient_cost_x_testing

            output_dic['current_best_acc'] = current_best_acc

    return params_dic,hyper_params_dic,output_dic


def est_hd_resnet_use_delta(data, hd_param_dic, penalty_const, MODEL_NAME, restore=False, n_epoches=10000, n_mini_batch = 100,
                  M=3, n_hidden=100, simul_data=None, Dropout=False, Dropout_rate=0.01, l2_regu = 1e-50, BN=False, x_dim=15,
                  K=2, D=7, current_best_acc=0):
    ###
    tf.reset_default_graph()

    x0_training, x1_training, t1_training, z_training, y_training = data['training']
    x0_testing, x1_testing, t1_testing, z_testing, y_testing = data['testing']

    # change data type
    x0_training = x0_training.astype('float32')
    x1_training = x1_training.astype('float32')
    t1_training = t1_training.astype('float32')
    z_training = z_training.astype('float32')
    x0_testing = x0_testing.astype('float32')
    x1_testing = x1_testing.astype('float32')
    t1_testing = t1_testing.astype('float32')
    z_testing = z_testing.astype('float32')
    y_training = y_training.astype('int32')
    y_testing = y_testing.astype('int32')

    D = z_training.shape[1]

    z = tf.placeholder(dtype=tf.float32, shape=(None, D), name='z')
    x0 = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='x0')
    x1 = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='x1')
    t1 = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='t1')
    y = tf.placeholder(dtype=tf.int32, shape=(None), name='y')
    x = tf.concat([x0, x1, t1], axis=1)

    #
    r_param_0 = hd_param_dic['r_param_0']
    b_param_0 = hd_param_dic['b_param_0']
    r_param = hd_param_dic['r_param']
    b_param = hd_param_dic['b_param']

    r_param_initial = tf.add(tf.matmul(z, r_param), r_param_0)
    b_param_initial = tf.add(tf.matmul(z, b_param), b_param_0)

    r = tf.sigmoid(r_param_initial) * 0.1  # reduce to 0 - 0.1
    b = tf.sigmoid(b_param_initial)

    ### hd utilities
    u0 = tf.identity(x0, name='u0')
    u1_ = tf.multiply(tf.multiply(x1, b), tf.exp(-tf.multiply(r, t1)))
    u1 = tf.identity(u1_, name='u1')
    u_hd = tf.concat([u0, u1], axis=1)

    ### dnn utilities
    hidden = x
    for i in range(M):
        hidden = standard_hidden_layer(hidden, n_hidden, BN, Dropout, Dropout_rate)
    u_dnn = tf.layers.dense(hidden, K, name='u_dnn')

    u = tf.add((1-penalty_const) * u_hd, penalty_const * u_dnn)
    prob = tf.nn.softmax(u)

    # gradient info: here is alternative 0 respect to inputs
    gradient_u1_x = [
        tf.concat([tf.gradients(u[:, 1], [x0])[0], tf.gradients(u[:, 1], [x1])[0], tf.gradients(u[:, 1], [t1])[0]],
                  axis=1)]
    gradient_prob1_x = [tf.concat(
        [tf.gradients(prob[:, 1], [x0])[0], tf.gradients(prob[:, 1], [x1])[0], tf.gradients(prob[:, 1], [t1])[0]],
        axis=1)]

    # regularization l2
    l2_regularizer = tf.contrib.layers.l2_regularizer(scale=l2_regu, scope=None)
    vars_ = tf.trainable_variables()
    regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, vars_)

    ###
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=u, labels=y), name='cost')
    cost += regularization_penalty
    log_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=u, labels=y), name='log_loss')

    #
    gradient_cost_x = [tf.concat([tf.gradients(cost, [x0])[0], tf.gradients(cost, [x1])[0], tf.gradients(cost, [t1])[0],
                                  tf.gradients(cost, [z])[0]], axis=1)]

    # eval params
    correct = tf.nn.in_top_k(u, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    optimizer = tf.train.AdamOptimizer()  # default rate 0.001; best so far: 0.01
    training_op = optimizer.minimize(cost)  # minimize the opt objective
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # n_mini_batch = 500
    cost_training = []
    cost_testing = []
    accuracy_training = []
    accuracy_testing = []
    log_loss_training = []
    log_loss_testing = []

    with tf.Session() as sess:
        # always run this to train the model
        init.run()
        feed_training = {x0: x0_training, x1: x1_training, t1: t1_training, z: z_training, y: y_training}
        feed_testing = {x0: x0_testing, x1: x1_testing, t1: t1_testing, z: z_testing, y: y_testing}

        if restore == True:
            saver.restore(sess, "tmp/" + MODEL_NAME + ".ckpt")
            u_training = u.eval(feed_dict=feed_training)
            u_testing = u.eval(feed_dict=feed_testing)
            prob_training = prob.eval(feed_dict=feed_training)
            prob_testing = prob.eval(feed_dict=feed_testing)
            #
            current_cost_training = cost.eval(feed_dict=feed_training)
            current_cost_testing = cost.eval(feed_dict=feed_testing)
            current_accuracy_training = accuracy.eval(feed_dict=feed_training)
            current_accuracy_testing = accuracy.eval(feed_dict=feed_testing)
            current_log_loss_training = log_loss.eval(feed_dict=feed_training)
            current_log_loss_testing = log_loss.eval(feed_dict=feed_testing)
            #
            log_loss_training.append(current_log_loss_training)
            log_loss_testing.append(current_log_loss_testing)
            cost_training.append(current_cost_training)
            cost_testing.append(current_cost_testing)
            accuracy_training.append(current_accuracy_training)
            accuracy_testing.append(current_accuracy_testing)
            #
            gradient_u1_x_training = gradient_u1_x[0].eval(feed_dict=feed_training)
            gradient_u1_x_testing = gradient_u1_x[0].eval(feed_dict=feed_testing)
            gradient_prob1_x_training = gradient_prob1_x[0].eval(feed_dict=feed_training)
            gradient_prob1_x_testing = gradient_prob1_x[0].eval(feed_dict=feed_testing)
            gradient_cost_x_training = gradient_cost_x[0].eval(feed_dict=feed_training)
            gradient_cost_x_testing = gradient_cost_x[0].eval(feed_dict=feed_testing)

            # save
            params_dic = {}
            hyper_params_dic = {}
            output_dic = {}
            output_dic['u_training'] = u_training
            output_dic['u_testing'] = u_testing
            output_dic['prob_training'] = prob_training
            output_dic['prob_testing'] = prob_testing

            output_dic['cost_training'] = cost_training
            output_dic['cost_testing'] = cost_testing
            output_dic['accuracy_training'] = accuracy_training
            output_dic['accuracy_testing'] = accuracy_testing
            output_dic['log_loss_training'] = log_loss_training
            output_dic['log_loss_testing'] = log_loss_testing

            output_dic['gradient_u1_x_training'] = gradient_u1_x_training
            output_dic['gradient_u1_x_testing'] = gradient_u1_x_testing
            output_dic['gradient_prob1_x_training'] = gradient_prob1_x_training
            output_dic['gradient_prob1_x_testing'] = gradient_prob1_x_testing
            output_dic['gradient_cost_x_training'] = gradient_cost_x_training
            output_dic['gradient_cost_x_testing'] = gradient_cost_x_testing

        elif restore == False:
            for i in range(n_epoches):
                x0_training_batch, x1_training_batch, t1_training_batch, z_training_batch, y_training_batch = \
                    obtain_mini_batch_hd(x0_training, x1_training, t1_training, z_training, y_training,
                                         n_mini_batch=100)

                # training
                sess.run(training_op, feed_dict={x0: x0_training_batch, x1: x1_training_batch, t1: t1_training_batch,
                                                 z: z_training_batch, y: y_training_batch})
                if i % 1000 == 0:
                    print("Iterations", i, "Cost = ", cost.eval(feed_dict=feed_training))
                if i % 20 == 0:
                    current_cost_training = cost.eval(feed_dict=feed_training)
                    current_cost_testing = cost.eval(feed_dict=feed_testing)
                    current_accuracy_training = accuracy.eval(feed_dict=feed_training)
                    current_accuracy_testing = accuracy.eval(feed_dict=feed_testing)
                    current_log_loss_training = log_loss.eval(feed_dict=feed_training)
                    current_log_loss_testing = log_loss.eval(feed_dict=feed_testing)

                    cost_training.append(current_cost_training)
                    cost_testing.append(current_cost_testing)
                    accuracy_training.append(current_accuracy_training)
                    accuracy_testing.append(current_accuracy_testing)
                    log_loss_training.append(current_log_loss_training)
                    log_loss_testing.append(current_log_loss_testing)

            # save the model
            if accuracy_testing[-1] >= current_best_acc:
                saver.save(sess, "tmp/" + MODEL_NAME + ".ckpt")
                current_best_acc = accuracy_testing[-1]
            #
            u_training = u.eval(feed_dict=feed_training)
            u_testing = u.eval(feed_dict=feed_testing)
            prob_training = prob.eval(feed_dict=feed_training)
            prob_testing = prob.eval(feed_dict=feed_testing)
            #
            gradient_u1_x_training = gradient_u1_x[0].eval(feed_dict=feed_training)
            gradient_u1_x_testing = gradient_u1_x[0].eval(feed_dict=feed_testing)
            gradient_prob1_x_training = gradient_prob1_x[0].eval(feed_dict=feed_training)
            gradient_prob1_x_testing = gradient_prob1_x[0].eval(feed_dict=feed_testing)
            gradient_cost_x_training = gradient_cost_x[0].eval(feed_dict=feed_training)
            gradient_cost_x_testing = gradient_cost_x[0].eval(feed_dict=feed_testing)

            # outputs
            params_dic = {}
            hyper_params_dic = {}
            output_dic = {}

            output_dic['cost_training'] = cost_training
            output_dic['cost_testing'] = cost_testing
            output_dic['accuracy_training'] = accuracy_training
            output_dic['accuracy_testing'] = accuracy_testing
            output_dic['log_loss_training'] = log_loss_training
            output_dic['log_loss_testing'] = log_loss_testing

            output_dic['u_training'] = u_training
            output_dic['u_testing'] = u_testing
            output_dic['prob_training'] = prob_training
            output_dic['prob_testing'] = prob_testing

            output_dic['gradient_u1_x_training'] = gradient_u1_x_training
            output_dic['gradient_u1_x_testing'] = gradient_u1_x_testing
            output_dic['gradient_prob1_x_training'] = gradient_prob1_x_training
            output_dic['gradient_prob1_x_testing'] = gradient_prob1_x_testing
            output_dic['gradient_cost_x_training'] = gradient_cost_x_training
            output_dic['gradient_cost_x_testing'] = gradient_cost_x_testing

            output_dic['current_best_acc'] = current_best_acc

    return params_dic, hyper_params_dic, output_dic





def est_hd_resnet_use_delta_save_train(data, hd_param_dic, penalty_const, MODEL_NAME, restore=False, n_epoches=10000, n_mini_batch = 100,
                  M=3, n_hidden=100, simul_data=None, Dropout=False, Dropout_rate=0.01, l2_regu = 1e-50, BN=False, x_dim=15,
                  K=2, D=7, current_best_acc=0):
    ###
    tf.reset_default_graph()

    x0_training, x1_training, t1_training, z_training, y_training = data['training']
    x0_testing, x1_testing, t1_testing, z_testing, y_testing = data['testing']

    # change data type
    x0_training = x0_training.astype('float32')
    x1_training = x1_training.astype('float32')
    t1_training = t1_training.astype('float32')
    z_training = z_training.astype('float32')
    x0_testing = x0_testing.astype('float32')
    x1_testing = x1_testing.astype('float32')
    t1_testing = t1_testing.astype('float32')
    z_testing = z_testing.astype('float32')
    y_training = y_training.astype('int32')
    y_testing = y_testing.astype('int32')

    D = z_training.shape[1]

    z = tf.placeholder(dtype=tf.float32, shape=(None, D), name='z')
    x0 = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='x0')
    x1 = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='x1')
    t1 = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='t1')
    y = tf.placeholder(dtype=tf.int32, shape=(None), name='y')
    x = tf.concat([x0, x1, t1], axis=1)

    #
    r_param_0 = hd_param_dic['r_param_0']
    b_param_0 = hd_param_dic['b_param_0']
    r_param = hd_param_dic['r_param']
    b_param = hd_param_dic['b_param']

    r_param_initial = tf.add(tf.matmul(z, r_param), r_param_0)
    b_param_initial = tf.add(tf.matmul(z, b_param), b_param_0)

    r = tf.sigmoid(r_param_initial) * 0.1  # reduce to 0 - 0.1
    b = tf.sigmoid(b_param_initial)

    ### hd utilities
    u0 = tf.identity(x0, name='u0')
    u1_ = tf.multiply(tf.multiply(x1, b), tf.exp(-tf.multiply(r, t1)))
    u1 = tf.identity(u1_, name='u1')
    u_hd = tf.concat([u0, u1], axis=1)

    ### dnn utilities
    hidden = x
    for i in range(M):
        hidden = standard_hidden_layer(hidden, n_hidden, BN, Dropout, Dropout_rate)
    u_dnn = tf.layers.dense(hidden, K, name='u_dnn')

    u = tf.add((1-penalty_const) * u_hd, penalty_const * u_dnn)
    prob = tf.nn.softmax(u)

    # gradient info: here is alternative 0 respect to inputs
    gradient_u1_x = [
        tf.concat([tf.gradients(u[:, 1], [x0])[0], tf.gradients(u[:, 1], [x1])[0], tf.gradients(u[:, 1], [t1])[0]],
                  axis=1)]
    gradient_prob1_x = [tf.concat(
        [tf.gradients(prob[:, 1], [x0])[0], tf.gradients(prob[:, 1], [x1])[0], tf.gradients(prob[:, 1], [t1])[0]],
        axis=1)]

    # regularization l2
    l2_regularizer = tf.contrib.layers.l2_regularizer(scale=l2_regu, scope=None)
    vars_ = tf.trainable_variables()
    regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, vars_)

    ###
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=u, labels=y), name='cost')
    cost += regularization_penalty
    log_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=u, labels=y), name='log_loss')

    #
    gradient_cost_x = [tf.concat([tf.gradients(cost, [x0])[0], tf.gradients(cost, [x1])[0], tf.gradients(cost, [t1])[0],
                                  tf.gradients(cost, [z])[0]], axis=1)]

    # eval params
    correct = tf.nn.in_top_k(u, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    optimizer = tf.train.AdamOptimizer()  # default rate 0.001; best so far: 0.01
    training_op = optimizer.minimize(cost)  # minimize the opt objective
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # n_mini_batch = 500
    cost_training = []
    cost_testing = []
    accuracy_training = []
    accuracy_testing = []
    log_loss_training = []
    log_loss_testing = []

    with tf.Session() as sess:
        # always run this to train the model
        init.run()
        feed_training = {x0: x0_training, x1: x1_training, t1: t1_training, z: z_training, y: y_training}
        feed_testing = {x0: x0_testing, x1: x1_testing, t1: t1_testing, z: z_testing, y: y_testing}

        if restore == True:
            saver.restore(sess, "tmp/" + MODEL_NAME + ".ckpt")
            u_training = u.eval(feed_dict=feed_training)
            u_testing = u.eval(feed_dict=feed_testing)
            prob_training = prob.eval(feed_dict=feed_training)
            prob_testing = prob.eval(feed_dict=feed_testing)
            #
            current_cost_training = cost.eval(feed_dict=feed_training)
            current_cost_testing = cost.eval(feed_dict=feed_testing)
            current_accuracy_training = accuracy.eval(feed_dict=feed_training)
            current_accuracy_testing = accuracy.eval(feed_dict=feed_testing)
            current_log_loss_training = log_loss.eval(feed_dict=feed_training)
            current_log_loss_testing = log_loss.eval(feed_dict=feed_testing)
            #
            log_loss_training.append(current_log_loss_training)
            log_loss_testing.append(current_log_loss_testing)
            cost_training.append(current_cost_training)
            cost_testing.append(current_cost_testing)
            accuracy_training.append(current_accuracy_training)
            accuracy_testing.append(current_accuracy_testing)
            #
            gradient_u1_x_training = gradient_u1_x[0].eval(feed_dict=feed_training)
            gradient_u1_x_testing = gradient_u1_x[0].eval(feed_dict=feed_testing)
            gradient_prob1_x_training = gradient_prob1_x[0].eval(feed_dict=feed_training)
            gradient_prob1_x_testing = gradient_prob1_x[0].eval(feed_dict=feed_testing)
            gradient_cost_x_training = gradient_cost_x[0].eval(feed_dict=feed_training)
            gradient_cost_x_testing = gradient_cost_x[0].eval(feed_dict=feed_testing)

            # save
            params_dic = {}
            hyper_params_dic = {}
            output_dic = {}
            output_dic['u_training'] = u_training
            output_dic['u_testing'] = u_testing
            output_dic['prob_training'] = prob_training
            output_dic['prob_testing'] = prob_testing

            output_dic['cost_training'] = cost_training
            output_dic['cost_testing'] = cost_testing
            output_dic['accuracy_training'] = accuracy_training
            output_dic['accuracy_testing'] = accuracy_testing
            output_dic['log_loss_training'] = log_loss_training
            output_dic['log_loss_testing'] = log_loss_testing

            output_dic['gradient_u1_x_training'] = gradient_u1_x_training
            output_dic['gradient_u1_x_testing'] = gradient_u1_x_testing
            output_dic['gradient_prob1_x_training'] = gradient_prob1_x_training
            output_dic['gradient_prob1_x_testing'] = gradient_prob1_x_testing
            output_dic['gradient_cost_x_training'] = gradient_cost_x_training
            output_dic['gradient_cost_x_testing'] = gradient_cost_x_testing

        elif restore == False:
            for i in range(n_epoches):
                x0_training_batch, x1_training_batch, t1_training_batch, z_training_batch, y_training_batch = \
                    obtain_mini_batch_hd(x0_training, x1_training, t1_training, z_training, y_training,
                                         n_mini_batch=100)

                # training
                sess.run(training_op, feed_dict={x0: x0_training_batch, x1: x1_training_batch, t1: t1_training_batch,
                                                 z: z_training_batch, y: y_training_batch})
                if i % 1000 == 0:
                    print("Iterations", i, "Cost = ", cost.eval(feed_dict=feed_training))

                current_cost_training = cost.eval(feed_dict=feed_training)
                current_cost_testing = cost.eval(feed_dict=feed_testing)
                current_accuracy_training = accuracy.eval(feed_dict=feed_training)
                current_accuracy_testing = accuracy.eval(feed_dict=feed_testing)
                current_log_loss_training = log_loss.eval(feed_dict=feed_training)
                current_log_loss_testing = log_loss.eval(feed_dict=feed_testing)

                cost_training.append(current_cost_training)
                cost_testing.append(current_cost_testing)
                accuracy_training.append(current_accuracy_training)
                accuracy_testing.append(current_accuracy_testing)
                log_loss_training.append(current_log_loss_training)
                log_loss_testing.append(current_log_loss_testing)

            # save the model
            if accuracy_testing[-1] >= current_best_acc:
                saver.save(sess, "tmp/" + MODEL_NAME + ".ckpt")
                current_best_acc = accuracy_testing[-1]
            #
            u_training = u.eval(feed_dict=feed_training)
            u_testing = u.eval(feed_dict=feed_testing)
            prob_training = prob.eval(feed_dict=feed_training)
            prob_testing = prob.eval(feed_dict=feed_testing)
            #
            gradient_u1_x_training = gradient_u1_x[0].eval(feed_dict=feed_training)
            gradient_u1_x_testing = gradient_u1_x[0].eval(feed_dict=feed_testing)
            gradient_prob1_x_training = gradient_prob1_x[0].eval(feed_dict=feed_training)
            gradient_prob1_x_testing = gradient_prob1_x[0].eval(feed_dict=feed_testing)
            gradient_cost_x_training = gradient_cost_x[0].eval(feed_dict=feed_training)
            gradient_cost_x_testing = gradient_cost_x[0].eval(feed_dict=feed_testing)

            # outputs
            params_dic = {}
            hyper_params_dic = {}
            output_dic = {}

            output_dic['cost_training'] = cost_training
            output_dic['cost_testing'] = cost_testing
            output_dic['accuracy_training'] = accuracy_training
            output_dic['accuracy_testing'] = accuracy_testing
            output_dic['log_loss_training'] = log_loss_training
            output_dic['log_loss_testing'] = log_loss_testing

            output_dic['u_training'] = u_training
            output_dic['u_testing'] = u_testing
            output_dic['prob_training'] = prob_training
            output_dic['prob_testing'] = prob_testing

            output_dic['gradient_u1_x_training'] = gradient_u1_x_training
            output_dic['gradient_u1_x_testing'] = gradient_u1_x_testing
            output_dic['gradient_prob1_x_training'] = gradient_prob1_x_training
            output_dic['gradient_prob1_x_testing'] = gradient_prob1_x_testing
            output_dic['gradient_cost_x_training'] = gradient_cost_x_training
            output_dic['gradient_cost_x_testing'] = gradient_cost_x_testing

            output_dic['current_best_acc'] = current_best_acc

    return params_dic, hyper_params_dic, output_dic

def est_hd_resnet_reverse(data, dnn_param_dic, penalty_const, MODEL_NAME, restore=False, n_epoches=10000, n_mini_batch=100, M=3,
                  n_hidden=100, simul_data=None, Dropout=False, Dropout_rate=0.01, BN=False, x_dim=15, K=2, D=7):
    ###
    tf.reset_default_graph()

    x0_training, x1_training, t1_training, z_training, y_training = data['training']
    x0_testing, x1_testing, t1_testing, z_testing, y_testing = data['testing']

    D = z_training.shape[1]

    z = tf.placeholder(dtype=tf.float32, shape=(None, D), name='z')
    x0 = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='x0')
    x1 = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='x1')
    t1 = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='t1')
    y = tf.placeholder(dtype=tf.int32, shape=(None), name='y')
    x = tf.concat([x0, x1, t1], axis=1)

    #
    r_param_0 = hd_param_dic['r_param_0']
    b_param_0 = hd_param_dic['b_param_0']
    r_param = hd_param_dic['r_param']
    b_param = hd_param_dic['b_param']

    r_param_initial = tf.add(tf.matmul(z, r_param), r_param_0)
    b_param_initial = tf.add(tf.matmul(z, b_param), b_param_0)

    r = tf.sigmoid(r_param_initial) * 0.1  # reduce to 0 - 0.1
    b = tf.sigmoid(b_param_initial)

    ### hd utilities
    u0 = tf.identity(x0, name='u0')
    u1_ = tf.multiply(tf.multiply(x1, b), tf.exp(-tf.multiply(r, t1)))
    u1 = tf.identity(u1_, name='u1')
    u_hd = tf.concat([u0, u1], axis=1)

    ### dnn utilities
    hidden = x
    for i in range(M):
        hidden = standard_hidden_layer(hidden, n_hidden, BN, Dropout, Dropout_rate)
    u_dnn = tf.layers.dense(hidden, K, name='u_dnn')

    u = tf.add(u_hd, u_dnn)
    prob = tf.nn.softmax(u)

    # gradient info: here is alternative 0 respect to inputs
    gradient_u1_x = [
        tf.concat([tf.gradients(u[:, 1], [x0])[0], tf.gradients(u[:, 1], [x1])[0], tf.gradients(u[:, 1], [t1])[0]],
                  axis=1)]
    gradient_prob1_x = [tf.concat(
        [tf.gradients(prob[:, 1], [x0])[0], tf.gradients(prob[:, 1], [x1])[0], tf.gradients(prob[:, 1], [t1])[0]],
        axis=1)]

    # regularization l2
    l2_regularizer = tf.contrib.layers.l2_regularizer(scale=penalty_const, scope=None)
    vars_ = tf.trainable_variables()
    regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, vars_)

    ###
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=u, labels=y), name='cost')
    cost += regularization_penalty
    log_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=u, labels=y), name='log_loss')

    #
    gradient_cost_x = [tf.concat([tf.gradients(cost, [x0])[0], tf.gradients(cost, [x1])[0], tf.gradients(cost, [t1])[0],
                                  tf.gradients(cost, [z])[0]], axis=1)]

    # eval params
    correct = tf.nn.in_top_k(u, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    optimizer = tf.train.AdamOptimizer()  # default rate 0.001; best so far: 0.01
    training_op = optimizer.minimize(cost)  # minimize the opt objective
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # n_mini_batch = 500
    cost_training = []
    cost_testing = []
    accuracy_training = []
    accuracy_testing = []
    log_loss_training = []
    log_loss_testing = []

    with tf.Session() as sess:
        # always run this to train the model
        init.run()
        feed_training = {x0: x0_training, x1: x1_training, t1: t1_training, z: z_training, y: y_training}
        feed_testing = {x0: x0_testing, x1: x1_testing, t1: t1_testing, z: z_testing, y: y_testing}

        if restore == True:
            saver.restore(sess, "tmp/" + MODEL_NAME + ".ckpt")
            u_training = u.eval(feed_dict=feed_training)
            u_testing = u.eval(feed_dict=feed_testing)
            prob_training = prob.eval(feed_dict=feed_training)
            prob_testing = prob.eval(feed_dict=feed_testing)
            #
            current_cost_training = cost.eval(feed_dict=feed_training)
            current_cost_testing = cost.eval(feed_dict=feed_testing)
            current_accuracy_training = accuracy.eval(feed_dict=feed_training)
            current_accuracy_testing = accuracy.eval(feed_dict=feed_testing)
            current_log_loss_training = log_loss.eval(feed_dict=feed_training)
            current_log_loss_testing = log_loss.eval(feed_dict=feed_testing)
            #
            log_loss_training.append(current_log_loss_training)
            log_loss_testing.append(current_log_loss_testing)
            cost_training.append(current_cost_training)
            cost_testing.append(current_cost_testing)
            accuracy_training.append(current_accuracy_training)
            accuracy_testing.append(current_accuracy_testing)
            #
            gradient_u1_x_training = gradient_u1_x[0].eval(feed_dict=feed_training)
            gradient_u1_x_testing = gradient_u1_x[0].eval(feed_dict=feed_testing)
            gradient_prob1_x_training = gradient_prob1_x[0].eval(feed_dict=feed_training)
            gradient_prob1_x_testing = gradient_prob1_x[0].eval(feed_dict=feed_testing)
            gradient_cost_x_training = gradient_cost_x[0].eval(feed_dict=feed_training)
            gradient_cost_x_testing = gradient_cost_x[0].eval(feed_dict=feed_testing)

            # save
            params_dic = {}
            hyper_params_dic = {}
            output_dic = {}
            output_dic['u_training'] = u_training
            output_dic['u_testing'] = u_testing
            output_dic['prob_training'] = prob_training
            output_dic['prob_testing'] = prob_testing

            output_dic['cost_training'] = cost_training
            output_dic['cost_testing'] = cost_testing
            output_dic['accuracy_training'] = accuracy_training
            output_dic['accuracy_testing'] = accuracy_testing
            output_dic['log_loss_training'] = log_loss_training
            output_dic['log_loss_testing'] = log_loss_testing

            output_dic['gradient_u1_x_training'] = gradient_u1_x_training
            output_dic['gradient_u1_x_testing'] = gradient_u1_x_testing
            output_dic['gradient_prob1_x_training'] = gradient_prob1_x_training
            output_dic['gradient_prob1_x_testing'] = gradient_prob1_x_testing
            output_dic['gradient_cost_x_training'] = gradient_cost_x_training
            output_dic['gradient_cost_x_testing'] = gradient_cost_x_testing

        elif restore == False:
            for i in range(n_epoches):
                x0_training_batch, x1_training_batch, t1_training_batch, z_training_batch, y_training_batch = \
                    obtain_mini_batch_hd(x0_training, x1_training, t1_training, z_training, y_training,
                                         n_mini_batch=100)

                # training
                sess.run(training_op, feed_dict={x0: x0_training_batch, x1: x1_training_batch, t1: t1_training_batch,
                                                 z: z_training_batch, y: y_training_batch})
                if i % 1000 == 0:
                    print("Iterations", i, "Cost = ", cost.eval(feed_dict=feed_training))
                if i % 20 == 0:
                    current_cost_training = cost.eval(feed_dict=feed_training)
                    current_cost_testing = cost.eval(feed_dict=feed_testing)
                    current_accuracy_training = accuracy.eval(feed_dict=feed_training)
                    current_accuracy_testing = accuracy.eval(feed_dict=feed_testing)
                    current_log_loss_training = log_loss.eval(feed_dict=feed_training)
                    current_log_loss_testing = log_loss.eval(feed_dict=feed_testing)

                    cost_training.append(current_cost_training)
                    cost_testing.append(current_cost_testing)
                    accuracy_training.append(current_accuracy_training)
                    accuracy_testing.append(current_accuracy_testing)
                    log_loss_training.append(current_log_loss_training)
                    log_loss_testing.append(current_log_loss_testing)

            # save the model
            saver.save(sess, "tmp/" + MODEL_NAME + ".ckpt")
            #
            u_training = u.eval(feed_dict=feed_training)
            u_testing = u.eval(feed_dict=feed_testing)
            prob_training = prob.eval(feed_dict=feed_training)
            prob_testing = prob.eval(feed_dict=feed_testing)
            #
            gradient_u1_x_training = gradient_u1_x[0].eval(feed_dict=feed_training)
            gradient_u1_x_testing = gradient_u1_x[0].eval(feed_dict=feed_testing)
            gradient_prob1_x_training = gradient_prob1_x[0].eval(feed_dict=feed_training)
            gradient_prob1_x_testing = gradient_prob1_x[0].eval(feed_dict=feed_testing)
            gradient_cost_x_training = gradient_cost_x[0].eval(feed_dict=feed_training)
            gradient_cost_x_testing = gradient_cost_x[0].eval(feed_dict=feed_testing)

            # outputs
            params_dic = {}
            hyper_params_dic = {}
            output_dic = {}

            output_dic['cost_training'] = cost_training
            output_dic['cost_testing'] = cost_testing
            output_dic['accuracy_training'] = accuracy_training
            output_dic['accuracy_testing'] = accuracy_testing
            output_dic['log_loss_training'] = log_loss_training
            output_dic['log_loss_testing'] = log_loss_testing

            output_dic['u_training'] = u_training
            output_dic['u_testing'] = u_testing
            output_dic['prob_training'] = prob_training
            output_dic['prob_testing'] = prob_testing

            output_dic['gradient_u1_x_training'] = gradient_u1_x_training
            output_dic['gradient_u1_x_testing'] = gradient_u1_x_testing
            output_dic['gradient_prob1_x_training'] = gradient_prob1_x_training
            output_dic['gradient_prob1_x_testing'] = gradient_prob1_x_testing
            output_dic['gradient_cost_x_training'] = gradient_cost_x_training
            output_dic['gradient_cost_x_testing'] = gradient_cost_x_testing

    return params_dic, hyper_params_dic, output_dic


def est_hd_resnet_simultaneous(data, penalty_const, MODEL_NAME, restore=False, n_epoches=10000, n_mini_batch=100, M=3,
                  n_hidden=100, simul_data=None, Dropout=False, Dropout_rate=0.01, BN=False, x_dim=15, K=2, D=7, current_best_acc = 0):
    ###
    tf.reset_default_graph()

    x0_training, x1_training, t1_training, z_training, y_training = data['training']
    x0_testing, x1_testing, t1_testing, z_testing, y_testing = data['testing']
    # change data type
    x0_training = x0_training.astype('float32')
    x1_training = x1_training.astype('float32')
    t1_training = t1_training.astype('float32')
    z_training = z_training.astype('float32')
    x0_testing = x0_testing.astype('float32')
    x1_testing = x1_testing.astype('float32')
    t1_testing = t1_testing.astype('float32')
    z_testing = z_testing.astype('float32')
    y_training = y_training.astype('int32')
    y_testing = y_testing.astype('int32')
    #

    D = z_training.shape[1]

    z = tf.placeholder(dtype=tf.float32, shape=(None, D), name='z')
    x0 = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='x0')
    x1 = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='x1')
    t1 = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='t1')
    y = tf.placeholder(dtype=tf.int32, shape=(None), name='y')
    x = tf.concat([x0, x1, t1], axis=1)

    #
    r_param_0 = tf.Variable(tf.random_normal([1]), dtype = tf.float32)
    b_param_0 = tf.Variable(tf.random_normal([1]), dtype = tf.float32)
    r_param = tf.Variable(tf.random_normal([D, 1]), dtype = tf.float32)
    b_param = tf.Variable(tf.random_normal([D, 1]), dtype = tf.float32)

    r_param_initial = tf.add(tf.matmul(z, r_param), r_param_0)
    b_param_initial = tf.add(tf.matmul(z, b_param), b_param_0)

    r = tf.sigmoid(r_param_initial) * 0.1  # reduce to 0 - 0.1
    b = tf.sigmoid(b_param_initial)

    ### hd utilities
    u0 = tf.identity(x0, name='u0')
    u1_ = tf.multiply(tf.multiply(x1, b), tf.exp(-tf.multiply(r, t1)))
    u1 = tf.identity(u1_, name='u1')
    u_hd = tf.concat([u0, u1], axis=1)

    ### dnn utilities
    hidden = x
    for i in range(M):
        hidden = standard_hidden_layer(hidden, n_hidden, BN, Dropout, Dropout_rate)
    u_dnn = tf.layers.dense(hidden, K, name='u_dnn')

    u = tf.add(u_hd, u_dnn)
    prob = tf.nn.softmax(u)

    # gradient info: here is alternative 0 respect to inputs
    gradient_u1_x = [
        tf.concat([tf.gradients(u[:, 1], [x0])[0], tf.gradients(u[:, 1], [x1])[0], tf.gradients(u[:, 1], [t1])[0]],
                  axis=1)]
    gradient_prob1_x = [tf.concat(
        [tf.gradients(prob[:, 1], [x0])[0], tf.gradients(prob[:, 1], [x1])[0], tf.gradients(prob[:, 1], [t1])[0]],
        axis=1)]

    # regularization l2
    l2_regularizer = tf.contrib.layers.l2_regularizer(scale=penalty_const, scope=None)
    vars_ = tf.trainable_variables()
    regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, vars_)

    ###
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=u, labels=y), name='cost')
    cost += regularization_penalty
    log_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=u, labels=y), name='log_loss')

    #
    gradient_cost_x = [tf.concat([tf.gradients(cost, [x0])[0], tf.gradients(cost, [x1])[0], tf.gradients(cost, [t1])[0],
                                  tf.gradients(cost, [z])[0]], axis=1)]

    # eval params
    correct = tf.nn.in_top_k(u, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    optimizer = tf.train.AdamOptimizer()  # default rate 0.001; best so far: 0.01
    training_op = optimizer.minimize(cost)  # minimize the opt objective
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # n_mini_batch = 500
    cost_training = []
    cost_testing = []
    accuracy_training = []
    accuracy_testing = []
    log_loss_training = []
    log_loss_testing = []

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # always run this to train the model
        init.run()
        feed_training = {x0: x0_training, x1: x1_training, t1: t1_training, z: z_training, y: y_training}
        feed_testing = {x0: x0_testing, x1: x1_testing, t1: t1_testing, z: z_testing, y: y_testing}

        if restore == True:
            saver.restore(sess, "tmp/" + MODEL_NAME + ".ckpt")
            u_training = u.eval(feed_dict=feed_training)
            u_testing = u.eval(feed_dict=feed_testing)
            prob_training = prob.eval(feed_dict=feed_training)
            prob_testing = prob.eval(feed_dict=feed_testing)
            #
            current_cost_training = cost.eval(feed_dict=feed_training)
            current_cost_testing = cost.eval(feed_dict=feed_testing)
            current_accuracy_training = accuracy.eval(feed_dict=feed_training)
            current_accuracy_testing = accuracy.eval(feed_dict=feed_testing)
            current_log_loss_training = log_loss.eval(feed_dict=feed_training)
            current_log_loss_testing = log_loss.eval(feed_dict=feed_testing)
            #
            log_loss_training.append(current_log_loss_training)
            log_loss_testing.append(current_log_loss_testing)
            cost_training.append(current_cost_training)
            cost_testing.append(current_cost_testing)
            accuracy_training.append(current_accuracy_training)
            accuracy_testing.append(current_accuracy_testing)
            #
            gradient_u1_x_training = gradient_u1_x[0].eval(feed_dict=feed_training)
            gradient_u1_x_testing = gradient_u1_x[0].eval(feed_dict=feed_testing)
            gradient_prob1_x_training = gradient_prob1_x[0].eval(feed_dict=feed_training)
            gradient_prob1_x_testing = gradient_prob1_x[0].eval(feed_dict=feed_testing)
            gradient_cost_x_training = gradient_cost_x[0].eval(feed_dict=feed_training)
            gradient_cost_x_testing = gradient_cost_x[0].eval(feed_dict=feed_testing)

            # save
            params_dic = {}
            hyper_params_dic = {}
            output_dic = {}
            output_dic['u_training'] = u_training
            output_dic['u_testing'] = u_testing
            output_dic['prob_training'] = prob_training
            output_dic['prob_testing'] = prob_testing

            output_dic['cost_training'] = cost_training
            output_dic['cost_testing'] = cost_testing
            output_dic['accuracy_training'] = accuracy_training
            output_dic['accuracy_testing'] = accuracy_testing
            output_dic['log_loss_training'] = log_loss_training
            output_dic['log_loss_testing'] = log_loss_testing

            output_dic['gradient_u1_x_training'] = gradient_u1_x_training
            output_dic['gradient_u1_x_testing'] = gradient_u1_x_testing
            output_dic['gradient_prob1_x_training'] = gradient_prob1_x_training
            output_dic['gradient_prob1_x_testing'] = gradient_prob1_x_testing
            output_dic['gradient_cost_x_training'] = gradient_cost_x_training
            output_dic['gradient_cost_x_testing'] = gradient_cost_x_testing

        elif restore == False:
            for i in range(n_epoches):
                x0_training_batch, x1_training_batch, t1_training_batch, z_training_batch, y_training_batch = \
                    obtain_mini_batch_hd(x0_training, x1_training, t1_training, z_training, y_training,
                                         n_mini_batch=100)

                # training
                sess.run(training_op, feed_dict={x0: x0_training_batch, x1: x1_training_batch, t1: t1_training_batch,
                                                 z: z_training_batch, y: y_training_batch})
                if i % 1000 == 0:
                    print("Iterations", i, "Cost = ", cost.eval(feed_dict=feed_training))
                if i % 20 == 0:
                    current_cost_training = cost.eval(feed_dict=feed_training)
                    current_cost_testing = cost.eval(feed_dict=feed_testing)
                    current_accuracy_training = accuracy.eval(feed_dict=feed_training)
                    current_accuracy_testing = accuracy.eval(feed_dict=feed_testing)
                    current_log_loss_training = log_loss.eval(feed_dict=feed_training)
                    current_log_loss_testing = log_loss.eval(feed_dict=feed_testing)

                    cost_training.append(current_cost_training)
                    cost_testing.append(current_cost_testing)
                    accuracy_training.append(current_accuracy_training)
                    accuracy_testing.append(current_accuracy_testing)
                    log_loss_training.append(current_log_loss_training)
                    log_loss_testing.append(current_log_loss_testing)

            # save the model
            if accuracy_testing[-1] >= current_best_acc:
                saver.save(sess, "tmp/"+MODEL_NAME+".ckpt")
                current_best_acc = accuracy_testing[-1]
            #
            u_training = u.eval(feed_dict=feed_training)
            u_testing = u.eval(feed_dict=feed_testing)
            prob_training = prob.eval(feed_dict=feed_training)
            prob_testing = prob.eval(feed_dict=feed_testing)
            #
            gradient_u1_x_training = gradient_u1_x[0].eval(feed_dict=feed_training)
            gradient_u1_x_testing = gradient_u1_x[0].eval(feed_dict=feed_testing)
            gradient_prob1_x_training = gradient_prob1_x[0].eval(feed_dict=feed_training)
            gradient_prob1_x_testing = gradient_prob1_x[0].eval(feed_dict=feed_testing)
            gradient_cost_x_training = gradient_cost_x[0].eval(feed_dict=feed_training)
            gradient_cost_x_testing = gradient_cost_x[0].eval(feed_dict=feed_testing)

            # outputs
            params_dic = {}
            hyper_params_dic = {}
            output_dic = {}

            output_dic['cost_training'] = cost_training
            output_dic['cost_testing'] = cost_testing
            output_dic['accuracy_training'] = accuracy_training
            output_dic['accuracy_testing'] = accuracy_testing
            output_dic['log_loss_training'] = log_loss_training
            output_dic['log_loss_testing'] = log_loss_testing

            output_dic['u_training'] = u_training
            output_dic['u_testing'] = u_testing
            output_dic['prob_training'] = prob_training
            output_dic['prob_testing'] = prob_testing

            output_dic['gradient_u1_x_training'] = gradient_u1_x_training
            output_dic['gradient_u1_x_testing'] = gradient_u1_x_testing
            output_dic['gradient_prob1_x_training'] = gradient_prob1_x_training
            output_dic['gradient_prob1_x_testing'] = gradient_prob1_x_testing
            output_dic['gradient_cost_x_training'] = gradient_cost_x_training
            output_dic['gradient_cost_x_testing'] = gradient_cost_x_testing

            output_dic['current_best_acc'] = current_best_acc

    return params_dic, hyper_params_dic, output_dic








def est_hd_resnet_simultaneous_use_delta(data, penalty_const, MODEL_NAME, restore=False, n_epoches=10000, n_mini_batch=100, M=3,
                  n_hidden=100, simul_data=None, Dropout=False, Dropout_rate=0.01, BN=False, l2_regu = 1e-50, x_dim=15, K=2, D=7, current_best_acc = 0):
    ###
    tf.reset_default_graph()

    x0_training, x1_training, t1_training, z_training, y_training = data['training']
    x0_testing, x1_testing, t1_testing, z_testing, y_testing = data['testing']
    # change data type
    x0_training = x0_training.astype('float32')
    x1_training = x1_training.astype('float32')
    t1_training = t1_training.astype('float32')
    z_training = z_training.astype('float32')
    x0_testing = x0_testing.astype('float32')
    x1_testing = x1_testing.astype('float32')
    t1_testing = t1_testing.astype('float32')
    z_testing = z_testing.astype('float32')
    y_training = y_training.astype('int32')
    y_testing = y_testing.astype('int32')
    #

    D = z_training.shape[1]

    z = tf.placeholder(dtype=tf.float32, shape=(None, D), name='z')
    x0 = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='x0')
    x1 = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='x1')
    t1 = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='t1')
    y = tf.placeholder(dtype=tf.int32, shape=(None), name='y')
    x = tf.concat([x0, x1, t1], axis=1)

    #
    r_param_0 = tf.Variable(tf.random_normal([1]), dtype = tf.float32)
    b_param_0 = tf.Variable(tf.random_normal([1]), dtype = tf.float32)
    r_param = tf.Variable(tf.random_normal([D, 1]), dtype = tf.float32)
    b_param = tf.Variable(tf.random_normal([D, 1]), dtype = tf.float32)

    r_param_initial = tf.add(tf.matmul(z, r_param), r_param_0)
    b_param_initial = tf.add(tf.matmul(z, b_param), b_param_0)

    r = tf.sigmoid(r_param_initial) * 0.1  # reduce to 0 - 0.1
    b = tf.sigmoid(b_param_initial)

    ### hd utilities
    u0 = tf.identity(x0, name='u0')
    u1_ = tf.multiply(tf.multiply(x1, b), tf.exp(-tf.multiply(r, t1)))
    u1 = tf.identity(u1_, name='u1')
    u_hd = tf.concat([u0, u1], axis=1)

    ### dnn utilities
    hidden = x
    for i in range(M):
        hidden = standard_hidden_layer(hidden, n_hidden, BN, Dropout, Dropout_rate)
    u_dnn = tf.layers.dense(hidden, K, name='u_dnn')

    u = tf.add((1-penalty_const) * u_hd, penalty_const * u_dnn)
    prob = tf.nn.softmax(u)

    # gradient info: here is alternative 0 respect to inputs
    gradient_u1_x = [
        tf.concat([tf.gradients(u[:, 1], [x0])[0], tf.gradients(u[:, 1], [x1])[0], tf.gradients(u[:, 1], [t1])[0]],
                  axis=1)]
    gradient_prob1_x = [tf.concat(
        [tf.gradients(prob[:, 1], [x0])[0], tf.gradients(prob[:, 1], [x1])[0], tf.gradients(prob[:, 1], [t1])[0]],
        axis=1)]

    # regularization l2
    l2_regularizer = tf.contrib.layers.l2_regularizer(scale=l2_regu, scope=None)
    vars_ = tf.trainable_variables()
    regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, vars_)

    ###
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=u, labels=y), name='cost')
    cost += regularization_penalty
    log_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=u, labels=y), name='log_loss')

    #
    gradient_cost_x = [tf.concat([tf.gradients(cost, [x0])[0], tf.gradients(cost, [x1])[0], tf.gradients(cost, [t1])[0],
                                  tf.gradients(cost, [z])[0]], axis=1)]

    # eval params
    correct = tf.nn.in_top_k(u, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    optimizer = tf.train.AdamOptimizer()  # default rate 0.001; best so far: 0.01
    training_op = optimizer.minimize(cost)  # minimize the opt objective
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # n_mini_batch = 500
    cost_training = []
    cost_testing = []
    accuracy_training = []
    accuracy_testing = []
    log_loss_training = []
    log_loss_testing = []

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # always run this to train the model
        init.run()
        feed_training = {x0: x0_training, x1: x1_training, t1: t1_training, z: z_training, y: y_training}
        feed_testing = {x0: x0_testing, x1: x1_testing, t1: t1_testing, z: z_testing, y: y_testing}

        if restore == True:
            saver.restore(sess, "tmp/" + MODEL_NAME + ".ckpt")
            u_training = u.eval(feed_dict=feed_training)
            u_testing = u.eval(feed_dict=feed_testing)
            prob_training = prob.eval(feed_dict=feed_training)
            prob_testing = prob.eval(feed_dict=feed_testing)
            #
            current_cost_training = cost.eval(feed_dict=feed_training)
            current_cost_testing = cost.eval(feed_dict=feed_testing)
            current_accuracy_training = accuracy.eval(feed_dict=feed_training)
            current_accuracy_testing = accuracy.eval(feed_dict=feed_testing)
            current_log_loss_training = log_loss.eval(feed_dict=feed_training)
            current_log_loss_testing = log_loss.eval(feed_dict=feed_testing)
            #
            log_loss_training.append(current_log_loss_training)
            log_loss_testing.append(current_log_loss_testing)
            cost_training.append(current_cost_training)
            cost_testing.append(current_cost_testing)
            accuracy_training.append(current_accuracy_training)
            accuracy_testing.append(current_accuracy_testing)
            #
            gradient_u1_x_training = gradient_u1_x[0].eval(feed_dict=feed_training)
            gradient_u1_x_testing = gradient_u1_x[0].eval(feed_dict=feed_testing)
            gradient_prob1_x_training = gradient_prob1_x[0].eval(feed_dict=feed_training)
            gradient_prob1_x_testing = gradient_prob1_x[0].eval(feed_dict=feed_testing)
            gradient_cost_x_training = gradient_cost_x[0].eval(feed_dict=feed_training)
            gradient_cost_x_testing = gradient_cost_x[0].eval(feed_dict=feed_testing)

            # save
            params_dic = {}
            hyper_params_dic = {}
            output_dic = {}
            output_dic['u_training'] = u_training
            output_dic['u_testing'] = u_testing
            output_dic['prob_training'] = prob_training
            output_dic['prob_testing'] = prob_testing

            output_dic['cost_training'] = cost_training
            output_dic['cost_testing'] = cost_testing
            output_dic['accuracy_training'] = accuracy_training
            output_dic['accuracy_testing'] = accuracy_testing
            output_dic['log_loss_training'] = log_loss_training
            output_dic['log_loss_testing'] = log_loss_testing

            output_dic['gradient_u1_x_training'] = gradient_u1_x_training
            output_dic['gradient_u1_x_testing'] = gradient_u1_x_testing
            output_dic['gradient_prob1_x_training'] = gradient_prob1_x_training
            output_dic['gradient_prob1_x_testing'] = gradient_prob1_x_testing
            output_dic['gradient_cost_x_training'] = gradient_cost_x_training
            output_dic['gradient_cost_x_testing'] = gradient_cost_x_testing

        elif restore == False:
            for i in range(n_epoches):
                x0_training_batch, x1_training_batch, t1_training_batch, z_training_batch, y_training_batch = \
                    obtain_mini_batch_hd(x0_training, x1_training, t1_training, z_training, y_training,
                                         n_mini_batch=100)

                # training
                sess.run(training_op, feed_dict={x0: x0_training_batch, x1: x1_training_batch, t1: t1_training_batch,
                                                 z: z_training_batch, y: y_training_batch})
                if i % 1000 == 0:
                    print("Iterations", i, "Cost = ", cost.eval(feed_dict=feed_training))
                if i % 20 == 0:
                    current_cost_training = cost.eval(feed_dict=feed_training)
                    current_cost_testing = cost.eval(feed_dict=feed_testing)
                    current_accuracy_training = accuracy.eval(feed_dict=feed_training)
                    current_accuracy_testing = accuracy.eval(feed_dict=feed_testing)
                    current_log_loss_training = log_loss.eval(feed_dict=feed_training)
                    current_log_loss_testing = log_loss.eval(feed_dict=feed_testing)

                    cost_training.append(current_cost_training)
                    cost_testing.append(current_cost_testing)
                    accuracy_training.append(current_accuracy_training)
                    accuracy_testing.append(current_accuracy_testing)
                    log_loss_training.append(current_log_loss_training)
                    log_loss_testing.append(current_log_loss_testing)

            # save the model
            if accuracy_testing[-1] >= current_best_acc:
                saver.save(sess, "tmp/"+MODEL_NAME+".ckpt")
                current_best_acc = accuracy_testing[-1]
            #
            u_training = u.eval(feed_dict=feed_training)
            u_testing = u.eval(feed_dict=feed_testing)
            prob_training = prob.eval(feed_dict=feed_training)
            prob_testing = prob.eval(feed_dict=feed_testing)
            #
            gradient_u1_x_training = gradient_u1_x[0].eval(feed_dict=feed_training)
            gradient_u1_x_testing = gradient_u1_x[0].eval(feed_dict=feed_testing)
            gradient_prob1_x_training = gradient_prob1_x[0].eval(feed_dict=feed_training)
            gradient_prob1_x_testing = gradient_prob1_x[0].eval(feed_dict=feed_testing)
            gradient_cost_x_training = gradient_cost_x[0].eval(feed_dict=feed_training)
            gradient_cost_x_testing = gradient_cost_x[0].eval(feed_dict=feed_testing)

            # outputs
            params_dic = {}
            hyper_params_dic = {}
            output_dic = {}

            output_dic['cost_training'] = cost_training
            output_dic['cost_testing'] = cost_testing
            output_dic['accuracy_training'] = accuracy_training
            output_dic['accuracy_testing'] = accuracy_testing
            output_dic['log_loss_training'] = log_loss_training
            output_dic['log_loss_testing'] = log_loss_testing

            output_dic['u_training'] = u_training
            output_dic['u_testing'] = u_testing
            output_dic['prob_training'] = prob_training
            output_dic['prob_testing'] = prob_testing

            output_dic['gradient_u1_x_training'] = gradient_u1_x_training
            output_dic['gradient_u1_x_testing'] = gradient_u1_x_testing
            output_dic['gradient_prob1_x_training'] = gradient_prob1_x_training
            output_dic['gradient_prob1_x_testing'] = gradient_prob1_x_testing
            output_dic['gradient_cost_x_training'] = gradient_cost_x_training
            output_dic['gradient_cost_x_testing'] = gradient_cost_x_testing

            output_dic['current_best_acc'] = current_best_acc

    return params_dic, hyper_params_dic, output_dic































