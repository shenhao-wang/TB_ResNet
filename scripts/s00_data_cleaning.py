#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 12:48:50 2019

s0_data_cleaning

@author: shenhao
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pickle

#cd "/Users/shenhao/Dropbox (MIT)/Shenhao_Jinhua (1)/10_ut_resnet/code" 
df_sp_train = pd.read_csv('data/raw/data_AV_Singapore_v1_sp_train.csv')
df_sp_validation = pd.read_csv('data/raw/data_AV_Singapore_v1_sp_validation.csv')
df_sp_test = pd.read_csv('data/raw/data_AV_Singapore_v1_sp_test.csv')

data_dic = {'training':df_sp_train, 'validation':df_sp_validation, 'testing':df_sp_test}

#df_sp_train.shape[0]+df_sp_validation.shape[0]+df_sp_test.shape[0]


# save 
with open('data/process/cm.pickle', 'wb') as data:
    pickle.dump(data_dic, data, protocol=pickle.HIGHEST_PROTOCOL)


########################################################################
df=pd.read_stata("data/raw/20060431_risk.dta")
df_time = pd.read_stata("data/raw/20060431_time.dta")

### clean dataset for time preference
var_names_time = ['choice', 't', 'y', 'reward', 'chinese', 'moneykeeper', 'age', 
                  'gender', 'edu', 'income', 'market', 'south', 'payment_g2']
df_time_n = df_time[var_names_time]
df_time_n['reward'].mean()
# change the scale
df_time_n[['reward', 'y']] = df_time_n[['reward', 'y']] * 1e-5
df_time_n['t'] = df_time_n['t'] * 0.1

# training and testing
np.random.seed(10)
random_index = np.arange(df_time_n.shape[0])
np.random.shuffle(random_index)
training_index = random_index[:np.int(len(random_index)*4/5)]
testing_index = random_index[np.int(len(random_index)*4/5):]
training_df = df_time_n.loc[training_index, :]
testing_df = df_time_n.loc[testing_index, :]
data_time_dic = {'training':training_df, 'testing':testing_df}
# 

with open('data/process/time.pickle', 'wb') as data:
    pickle.dump(data_time_dic, data, protocol=pickle.HIGHEST_PROTOCOL)
                           
#plt.hist(df_time_n['t'])
#plt.hist(df_time_n['y'])
#plt.hist(df_time_n['reward'])

### clean dataset for risk preference                           
lottery_m1 = \
    [[40, 0.3, 10, 0.7, 68, 0.1, 5, 0.9],
     [40, 0.3, 10, 0.7, 75, 0.1, 5, 0.9],
     [40, 0.3, 10, 0.7, 83, 0.1, 5, 0.9],
     [40, 0.3, 10, 0.7, 93, 0.1, 5, 0.9],
     [40, 0.3, 10, 0.7, 106, 0.1, 5, 0.9],
     [40, 0.3, 10, 0.7, 125, 0.1, 5, 0.9],
     [40, 0.3, 10, 0.7, 150, 0.1, 5, 0.9],
     [40, 0.3, 10, 0.7, 185, 0.1, 5, 0.9],
     [40, 0.3, 10, 0.7, 220, 0.1, 5, 0.9],
     [40, 0.3, 10, 0.7, 300, 0.1, 5, 0.9],
     [40, 0.3, 10, 0.7, 400, 0.1, 5, 0.9],
     [40, 0.3, 10, 0.7, 600, 0.1, 5, 0.9],
     [40, 0.3, 10, 0.7, 1000, 0.1, 5, 0.9],
     [40, 0.3, 10, 0.7, 1700, 0.1, 5, 0.9],
            ]
lottery_m1 = np.array(lottery_m1)

lottery_m2 = \
    [[40, 0.9, 30, 0.1, 54, 0.7, 5, 0.3],
     [40, 0.9, 30, 0.1, 56, 0.7, 5, 0.3],
     [40, 0.9, 30, 0.1, 58, 0.7, 5, 0.3],
     [40, 0.9, 30, 0.1, 60, 0.7, 5, 0.3],
     [40, 0.9, 30, 0.1, 62, 0.7, 5, 0.3],
     [40, 0.9, 30, 0.1, 65, 0.7, 5, 0.3],
     [40, 0.9, 30, 0.1, 68, 0.7, 5, 0.3],
     [40, 0.9, 30, 0.1, 72, 0.7, 5, 0.3],
     [40, 0.9, 30, 0.1, 77, 0.7, 5, 0.3],
     [40, 0.9, 30, 0.1, 83, 0.7, 5, 0.3],
     [40, 0.9, 30, 0.1, 90, 0.7, 5, 0.3],
     [40, 0.9, 30, 0.1, 100, 0.7, 5, 0.3],
     [40, 0.9, 30, 0.1, 110, 0.7, 5, 0.3],
     [40, 0.9, 30, 0.1, 130, 0.7, 5, 0.3],
            ]
lottery_m2 = np.array(lottery_m2)

lottery_m3 = \
    [[25, 0.5, -4, 0.5, 30, 0.5, -21, 0.5],
     [4, 0.5, -4, 0.5, 30, 0.5, -21, 0.5],
     [1, 0.5, -4, 0.5, 30, 0.5, -21, 0.5],
     [1, 0.5, -4, 0.5, 30, 0.5, -16, 0.5],
     [1, 0.5, -8, 0.5, 30, 0.5, -16, 0.5],
     [1, 0.5, -8, 0.5, 30, 0.5, -14, 0.5],
     [1, 0.5, -8, 0.5, 30, 0.5, -11, 0.5],
            ]
lottery_m3 = np.array(lottery_m3)

var_names_m1 = ['id', 'q1', 'age','gender','edu','income', \
                'mnincome','market','vfctnc','south','lambda1', 'lambda2', \
                'headnowork', 'rainfall', 'chinese', 'nmlrlincome']

var_names_m2 = ['id', 'q2', 'age','gender','edu','income', \
                'mnincome','market','vfctnc','south','lambda1', 'lambda2', \
                'headnowork', 'rainfall', 'chinese', 'nmlrlincome']

var_names_m3 = ['id', 'q3', 'age','gender','edu','income', \
                'mnincome','market','vfctnc','south','lambda1', 'lambda2', \
                'headnowork', 'rainfall', 'chinese', 'nmlrlincome']

def process_matrix(lottery_m, var_names_m, df, q_name):
    '''
    choose var_names_m from df.
    augment lottery_m to df_i.
    create new choice variables
    '''
    # process the first matrix
    d = df[var_names_m]    
    # 1. expand matrix
    d_expand = pd.DataFrame(np.repeat(d.values, lottery_m.shape[0], axis = 0),
                             columns = d.columns)
    if q_name == 'q1':     
        d_expand['series'] = np.tile(np.arange(1, lottery_m.shape[0] + 1), (1,d.shape[0])).T
    elif q_name == 'q2':
        d_expand['series'] = np.tile(np.arange(1 + 14, lottery_m.shape[0] + 1 + 14), (1,d.shape[0])).T
    elif q_name == 'q3':
        d_expand['series'] = np.tile(np.arange(1 + 14 + 14, lottery_m.shape[0] + 1 + 14 + 14), (1,d.shape[0])).T

    # 2. add alternative-specific variables
    alt_spec_m = np.tile(lottery_m, (d.shape[0], 1))
    alt_spec_df = pd.DataFrame(alt_spec_m, 
                               columns = ['x00', 'p00', 'x01', 'p01', 'x10', 'p10', 'x11', 'p11'])
    df_i = pd.concat((d_expand, alt_spec_df), axis = 1)
    # 3. create choice variables
    # option B: 1; option A: 0;
    df_i['choice'] = 0
    df_i.loc[np.multiply(df_i['series'] >= df_i[q_name], df_i[q_name] > 0), 'choice'] = 1 
#    print(df_i.head(20))
#    print(df_i.tail(20))
    return df_i

df1 = process_matrix(lottery_m1, var_names_m1, df, 'q1')
df2 = process_matrix(lottery_m2, var_names_m2, df, 'q2')
df3 = process_matrix(lottery_m3, var_names_m3, df, 'q3')

df1.drop(['q1'], axis = 1, inplace = True)
df2.drop(['q2'], axis = 1, inplace = True)
df3.drop(['q3'], axis = 1, inplace = True)

df_full_alt_spec = pd.concat([df1, df2, df3], axis = 0)
df_full_alt_spec.index = np.arange(df_full_alt_spec.shape[0])
#print(df_full_alt_spec.shape)
#print(df_full_alt_spec.head(20))

# 1) remove very large x10!!! (> 200); 2) shrink the scale by 10.0...
df_full_alt_spec_valid = df_full_alt_spec.loc[df_full_alt_spec['x10'] < 200.0, :]
df_full_alt_spec_valid.index = np.arange(df_full_alt_spec_valid.shape[0])
df_full_alt_spec_valid[['x00', 'x01', 'x10', 'x11']] = df_full_alt_spec_valid[['x00', 'x01', 'x10', 'x11']]/10.0

# split into training and testing.
np.random.seed(0)
random_index = np.arange(df_full_alt_spec_valid.shape[0])
np.random.shuffle(random_index)
training_index = random_index[:np.int(len(random_index)*4/5)]
testing_index = random_index[np.int(len(random_index)*4/5):]
training_df = df_full_alt_spec_valid.loc[training_index, :]
testing_df = df_full_alt_spec_valid.loc[testing_index, :]
data_dic = {'training':training_df, 'testing':testing_df}

# save 
with open('data/process/risk.pickle', 'wb') as data:
    pickle.dump(data_dic, data, protocol=pickle.HIGHEST_PROTOCOL)























