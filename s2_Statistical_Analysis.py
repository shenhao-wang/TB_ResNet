import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle
pd.options.mode.chained_assignment = None  # default='warn'
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")


######################CM data set
# same as previous paper, ignored

with open('data/process/cm.pickle', 'rb') as data:
    data_dic = pickle.load(data)

data_train = data_dic['training']
data_test = data_dic['testing']
print('CM num of train', len(data_train))
print('CM num of test', len(data_test))


#######################HD data set
with open('data/process/time.pickle', 'rb') as data:
    data_dic = pickle.load(data)

data_train = data_dic['training']
data_test = data_dic['testing']

print('HD num of train', len(data_train))
print('HD num of test', len(data_test))

data_all = pd.concat([data_train,data_test])

data_all_stat = data_all.describe()

data_all_stat.to_csv('output/table/data_HD_statistics.csv')

for i in range(2):
    num = len(data_all.loc[data_all['choice']==i])
    print('HD choice',i, num, 'proportion:', num/len(data_all))

#######################PT data set
with open('data/process/risk.pickle', 'rb') as data:
    data_dic = pickle.load(data)

data_train = data_dic['training']
data_test = data_dic['testing']
print('PT num of train', len(data_train))
print('PT num of test', len(data_test))

data_all = pd.concat([data_train,data_test])

data_all_stat = data_all.describe()

data_all_stat.to_csv('output/table/data_PT_statistics.csv')

for i in range(2):
    num = len(data_all.loc[data_all['choice']==i])
    print('PT choice',i, num, 'proportion:', num/len(data_all))
