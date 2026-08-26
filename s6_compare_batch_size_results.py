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
from matplotlib.ticker import FormatStrFormatter






def hd_compare():

    results = {'Model':[],'Prediction_acc':[],'Cross_entropy_loss':[],'F1_score':[]}


    task_type = 'hd'
    batch_size_list = [64, 128]

    with open('data/process/time.pickle', 'rb') as data:
        hd_data = pickle.load(data)

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

    for batch_size in batch_size_list:
        with open('hd_info_use_delta_batch_size_' + str(batch_size) + '.pickle', 'rb') as f:
            hd_info = pickle.load(f)

        info = copy.deepcopy(hd_info)

        MODEL_NAME_LIST = ['hd_resnet_0.05']


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
            for replic in range(num_replications):
                # print(info[MODEL_NAME]['data_output'][replic]['accuracy_testing'])
                acc_test_list.append(info[MODEL_NAME]['data_output'][replic]['accuracy_testing'][-1])
                acc_training_list.append(info[MODEL_NAME]['data_output'][replic]['accuracy_training'][-1])
                cost_training_list.append(info[MODEL_NAME]['data_output'][replic]['cost_training'][-1])
                cost_testing_list.append(info[MODEL_NAME]['data_output'][replic]['cost_testing'][-1])
                log_loss_training_list.append(info[MODEL_NAME]['data_output'][replic]['log_loss_training'][-1])
                log_loss_testing_list.append(info[MODEL_NAME]['data_output'][replic]['log_loss_testing'][-1])
                prob_training = info[MODEL_NAME]['data_output'][replic]['prob_training']
                prob_testing = info[MODEL_NAME]['data_output'][replic]['prob_testing']
                current_pred_y_train = np.argmax(prob_training,1)
                current_pred_y_test = np.argmax(prob_testing,1)
                if task_type == 'cm':
                    y_training = copy.deepcopy(cm_y_training)
                    y_testing = copy.deepcopy(cm_y_testing)

                elif task_type == 'pt':
                    y_training = copy.deepcopy(pt_y_training)
                    y_testing = copy.deepcopy(pt_y_testing)
                else:
                    y_training = copy.deepcopy(hd_y_training)
                    y_testing = copy.deepcopy(hd_y_testing)

            current_f1_train = f1_score(y_training, current_pred_y_train, average='weighted')
            current_f1_test = f1_score(y_testing, current_pred_y_test, average='weighted')
            f1_score_training_list.append(current_f1_train)
            f1_score_testing_list.append(current_f1_test)
        model_name_final = MODEL_NAME + '_batch_size_' + str(batch_size)
        results['Model'].append(model_name_final)
        results['Prediction_acc'].append(acc_test_list[0])
        results['Cross_entropy_loss'].append(log_loss_testing_list[0])
        results['F1_score'].append(f1_score_testing_list[0])
    results = pd.DataFrame(results)
    results.to_csv('output/table/hd_batch_size_compare.csv',index=False)




def cm_compare():

    results = {'Model':[],'Prediction_acc':[],'Cross_entropy_loss':[],'F1_score':[]}


    task_type = 'cm'
    batch_size_list = [64, 128]

    with open('data/process/cm.pickle', 'rb') as data:
        cm_data = pickle.load(data)

    ########################################process hd_data###########################
    y_vars = ['choice']
    cm_y_training = cm_data['training'][y_vars].values[:, 0]
    cm_y_testing = cm_data['testing'][y_vars].values[:, 0]

    for batch_size in batch_size_list:
        with open('cm_info_use_delta_batch_size_' + str(batch_size) + '.pickle', 'rb') as f:
            cm_info = pickle.load(f)

        info = copy.deepcopy(cm_info)

        MODEL_NAME_LIST = ['cm_resnet_0.008']


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
            for replic in range(num_replications):
                # print(info[MODEL_NAME]['data_output'][replic]['accuracy_testing'])
                acc_test_list.append(info[MODEL_NAME]['data_output'][replic]['accuracy_testing'][-1])
                acc_training_list.append(info[MODEL_NAME]['data_output'][replic]['accuracy_training'][-1])
                cost_training_list.append(info[MODEL_NAME]['data_output'][replic]['cost_training'][-1])
                cost_testing_list.append(info[MODEL_NAME]['data_output'][replic]['cost_testing'][-1])
                log_loss_training_list.append(info[MODEL_NAME]['data_output'][replic]['log_loss_training'][-1])
                log_loss_testing_list.append(info[MODEL_NAME]['data_output'][replic]['log_loss_testing'][-1])
                prob_training = info[MODEL_NAME]['data_output'][replic]['prob_training']
                prob_testing = info[MODEL_NAME]['data_output'][replic]['prob_testing']
                current_pred_y_train = np.argmax(prob_training,1)
                current_pred_y_test = np.argmax(prob_testing,1)
                if task_type == 'cm':
                    y_training = copy.deepcopy(cm_y_training)
                    y_testing = copy.deepcopy(cm_y_testing)

                elif task_type == 'pt':
                    y_training = copy.deepcopy(pt_y_training)
                    y_testing = copy.deepcopy(pt_y_testing)
                else:
                    y_training = copy.deepcopy(hd_y_training)
                    y_testing = copy.deepcopy(hd_y_testing)

            current_f1_train = f1_score(y_training, current_pred_y_train, average='weighted')
            current_f1_test = f1_score(y_testing, current_pred_y_test, average='weighted')
            f1_score_training_list.append(current_f1_train)
            f1_score_testing_list.append(current_f1_test)
        model_name_final = MODEL_NAME + '_batch_size_' + str(batch_size)
        results['Model'].append(model_name_final)
        results['Prediction_acc'].append(acc_test_list[0])
        results['Cross_entropy_loss'].append(log_loss_testing_list[0])
        results['F1_score'].append(f1_score_testing_list[0])
    results = pd.DataFrame(results)
    results.to_csv('output/table/cm_batch_size_compare.csv',index=False)





def pt_compare():

    results = {'Model':[],'Prediction_acc':[],'Cross_entropy_loss':[],'F1_score':[]}


    task_type = 'pt'
    batch_size_list = [64, 128]

    with open('data/process/risk.pickle', 'rb') as data:
        pt_data = pickle.load(data)

    ########################################process hd_data###########################
    y_var = ['choice']
    training_df = pt_data['training']
    testing_df = pt_data['testing']
    pt_y_training = training_df[y_var].values[:, 0]
    pt_y_testing = testing_df[y_var].values[:, 0]

    for batch_size in batch_size_list:
        with open('pt_info_use_delta_batch_size_' + str(batch_size) + '.pickle', 'rb') as f:
            pt_info = pickle.load(f)

        info = copy.deepcopy(pt_info)

        MODEL_NAME_LIST = ['pt_resnet_0.9']


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
            for replic in range(num_replications):
                # print(info[MODEL_NAME]['data_output'][replic]['accuracy_testing'])
                acc_test_list.append(info[MODEL_NAME]['data_output'][replic]['accuracy_testing'][-1])
                acc_training_list.append(info[MODEL_NAME]['data_output'][replic]['accuracy_training'][-1])
                cost_training_list.append(info[MODEL_NAME]['data_output'][replic]['cost_training'][-1])
                cost_testing_list.append(info[MODEL_NAME]['data_output'][replic]['cost_testing'][-1])
                log_loss_training_list.append(info[MODEL_NAME]['data_output'][replic]['log_loss_training'][-1])
                log_loss_testing_list.append(info[MODEL_NAME]['data_output'][replic]['log_loss_testing'][-1])
                prob_training = info[MODEL_NAME]['data_output'][replic]['prob_training']
                prob_testing = info[MODEL_NAME]['data_output'][replic]['prob_testing']
                current_pred_y_train = np.argmax(prob_training,1)
                current_pred_y_test = np.argmax(prob_testing,1)
                if task_type == 'cm':
                    y_training = copy.deepcopy(cm_y_training)
                    y_testing = copy.deepcopy(cm_y_testing)

                elif task_type == 'pt':
                    y_training = copy.deepcopy(pt_y_training)
                    y_testing = copy.deepcopy(pt_y_testing)
                else:
                    y_training = copy.deepcopy(hd_y_training)
                    y_testing = copy.deepcopy(hd_y_testing)

            current_f1_train = f1_score(y_training, current_pred_y_train, average='weighted')
            current_f1_test = f1_score(y_testing, current_pred_y_test, average='weighted')
            f1_score_training_list.append(current_f1_train)
            f1_score_testing_list.append(current_f1_test)
        model_name_final = MODEL_NAME + '_batch_size_' + str(batch_size)
        results['Model'].append(model_name_final)
        results['Prediction_acc'].append(acc_test_list[0])
        results['Cross_entropy_loss'].append(log_loss_testing_list[0])
        results['F1_score'].append(f1_score_testing_list[0])
    results = pd.DataFrame(results)
    results.to_csv('output/table/pt_batch_size_compare.csv',index=False)



if __name__ == '__main__':
    # hd_compare()
    # cm_compare()
    pt_compare()