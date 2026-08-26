import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import re
from scipy.interpolate import make_interp_spline, BSpline



colors = sns.color_palette('muted')
def plot_curve(data_table,name_list, save_fig, penalty_const_list,task, spline_or_not):
    data_table = data_table.loc[data_table['model_name'].isin(name_list)]
    data_table = data_table.reset_index(drop=True)
    font_size = 16
    fig, ax = plt.subplots(figsize=(6, 5))
    #
    y = np.array(data_table['Prediction Accuracy (Testing)'])
    x_old = np.arange(len(data_table))
    if not spline_or_not:
        l1 = ax.plot(x_old, y, linewidth = 4,color = colors[1], label = 'Accuracy')
        #
    else:
        # xnew = np.linspace(min(x_old), max(x_old), 300)
        # spl = make_interp_spline(x_old, y, k=1)  # type:
        # power_smooth = spl(xnew)
        ax.scatter(x_old, y, color = colors[1], alpha = 0.5, s = 18, label = 'Accuracy')
        # l1 = ax.plot(xnew, power_smooth, linewidth=4, color=colors[1], label='Accuracy')
        if task == 'HD':
            sns_ret = sns.regplot(x_old, y, ax=ax, color=colors[1], order=4,
                                  line_kws={'linewidth': 4, 'label': 'Accuracy'}, scatter=False, ci=0)
        else:
            sns_ret = sns.regplot(x_old, y, ax = ax, color = colors[1], order = 2,line_kws={'linewidth':4,'label': 'Accuracy'},scatter=False,ci=0)
        l1 = sns_ret.get_lines()

    if task == 'CM':
        best_theta = 0.008
        idx = penalty_const_list.index(best_theta) + 1
        x_line = [x_old[idx],x_old[idx]]
        y_line = [0,1]
        ax.plot(x_line,y_line,'--',color = 'red',alpha = 0.5)
        ax.text(x_old[idx] - 7, (0.48+0.6)/2 - 0.03, r'$\delta^*$=' + str(best_theta), fontsize = font_size)

        y_ticks = list(np.arange(0.49, 0.60, 0.02))
        ax.set_yticks(y_ticks)
        y_tickslabel =[]
        for y in y_ticks:
            y_tickslabel.append(round(y,2))
        ax.set_yticklabels(y_tickslabel, fontsize=font_size)

        # used_penalty = [1e-5,0.008,0.05]
        # index_list = [penalty_const_list.index(ele) for ele in used_penalty]
        interval = round(len(penalty_const_list)/3)
        mid_label = [interval, 2*interval]
        index_list = [0] +  mid_label + [-1]
        x_ticks = np.array(x_old)[index_list]
        new_ticks = ['MNL'] + [str(penalty_const_list[ele]) for ele in mid_label] + ['DNN']
        ax.set_ylim([0.48, 0.6])
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(new_ticks, fontsize=font_size)
    elif task =='PT':
        best_theta = 0.9
        idx = penalty_const_list.index(best_theta) + 1
        x_line = [x_old[idx],x_old[idx]]
        y_line = [0,1]
        ax.plot(x_line,y_line,'--',color = 'red',alpha = 0.5)
        ax.text(x_old[idx] - 6, (0.67+0.95)/2 - 0.03, r'$\delta^*$=' + str(best_theta), fontsize = font_size)

        y_ticks = list(np.arange(0.67, 1, 0.05))
        ax.set_yticks(y_ticks)
        y_tickslabel =[]
        for y in y_ticks:
            y_tickslabel.append(round(y,2))
        ax.set_yticklabels(y_tickslabel, fontsize=font_size)

        # used_penalty = [1e-5,0.008,0.05]
        # index_list = [penalty_const_list.index(ele) for ele in used_penalty]
        interval = round(len(penalty_const_list)/3)
        mid_label = [interval, 2*interval]
        index_list = [0] +  mid_label + [-1]
        x_ticks = np.array(x_old)[index_list]
        new_ticks = ['PT'] + [str(penalty_const_list[ele]) for ele in mid_label] + ['DNN']
        ax.set_ylim([0.67, 0.97])
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(new_ticks, fontsize=font_size)
    else:
        best_theta = 0.05
        idx = penalty_const_list.index(best_theta) + 1
        x_line = [x_old[idx],x_old[idx]]
        y_line = [0,1]
        ax.plot(x_line, y_line, '--', color='red')
        ax.text(x_old[idx] - 6, (0.56+0.85)/2 - 0.03, r'$\delta^*$=' + str(best_theta), fontsize = font_size)


        y_ticks = list(np.arange(0.57, 1, 0.05))
        ax.set_yticks(y_ticks)
        y_tickslabel =[]
        for y in y_ticks:
            y_tickslabel.append(round(y,2))
        ax.set_yticklabels(y_tickslabel, fontsize=font_size)

        # used_penalty = [1e-5,0.008,0.05]
        # index_list = [penalty_const_list.index(ele) for ele in used_penalty]
        interval = round(len(penalty_const_list)/3)
        mid_label = [interval, 2*interval]
        index_list = [0] +  mid_label + [-1]
        x_ticks = np.array(x_old)[index_list]
        new_ticks = ['HD'] + [str(penalty_const_list[ele-1]) for ele in mid_label] + ['DNN']
        ax.set_ylim([0.56, 0.85])
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(new_ticks, fontsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)
    ax.set_xlabel(r'$\delta$', fontsize=font_size)
    ax.set_ylabel('Prediction Accuracy (Testing)', fontsize=font_size)
    #================
    ax2 = ax.twinx()  #

    # plt.xlim([-5, 95])
    y2 = data_table['Cross-entropy Loss (Testing)']
    if not spline_or_not:
        l2 = ax2.plot(x_old, data_table['Cross-entropy Loss (Testing)'], linewidth=4,
                 color=colors[5], label='Cross-entropy Loss')
    else:
        ax2.scatter(x_old, y2, color = colors[5], alpha = 0.5, s = 18, label = 'Cross-entropy Loss')
        # l1 = ax.plot(xnew, power_smooth, linewidth=4, color=colors[1], label='Accuracy')
        if task == 'HD':
            sns_ret = sns.regplot(x_old, y2, ax=ax2, color=colors[5], order=4, line_kws={'linewidth': 4}, scatter=False,
                                  ci=0)
        else:
            sns_ret = sns.regplot(x_old, y2, ax = ax2, color = colors[5], order = 3,line_kws={'linewidth':4},scatter=False,ci=0)
        l2 = sns_ret.get_lines()

    if task == 'CM':
        y_ticks = list(np.arange(1.1, 2.8+0.3, 0.3))
        ax2.set_yticks(y_ticks)
        y_tickslabel =[]
        for y in y_ticks:
            y_tickslabel.append(round(y,1))
        ax2.set_yticklabels(y_tickslabel, fontsize=font_size)
        ax2.set_ylim([1.0, 3.0])
    ax2.tick_params(axis="y", labelsize=font_size)
    ax2.set_ylabel('Cross-entropy Loss (Testing)', fontsize=font_size)

    lns = l1 +  l2
    # labs = [l.get_label() for l in lns]
    labs = ['Accuracy','Cross-entropy Loss']
    plt.legend(lns, labs, fontsize=font_size)
    plt.tight_layout()
    if save_fig == 1:
        plt.savefig('output/performance/convergence_of_delta_' + task + '.png', dpi=200)
    else:
        plt.show()


def plot_convergence(CM, PT, HD, save_fig):
    if CM:
        model_list = ['CM']
        penalty_const_list = [1e-10, 1e-5, 1e-4, 0.001, 0.002, 0.004, 0.005, 0.006, 0.007,
                              0.008, 0.009, 0.05, 0.03, 0.1, 0.3, 0.5, 0.95]

        data_table = pd.read_csv('output/table/0_cm_table_use_delta.csv')
        data_table = data_table.rename(columns = {'Unnamed: 0':'model_name'})
        # data_table['delta'] = data_table['model_name'].apply(lambda x: float(x[x.find("(")+1:x.find(")")]))
        for penalty in penalty_const_list:
            name = 'Resnet (' + str(penalty) + ')'
            if name in list(data_table['model_name']):
                model_list.append(name)
        model_list.append('DNN')
        # print(model_list[1])
        # print(data_table['model_name'].iloc[1])
        # print(model_list[1] == data_table['model_name'].iloc[1])
        plot_curve(data_table,model_list, save_fig, penalty_const_list,'CM',spline_or_not = True)

    if PT:
        model_list = ['PT']
        penalty_const_list = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 0.001, 0.002, 0.004, 0.005, 0.007,
                              0.008, 0.009, 0.01, 0.03, 0.05, 0.3, 0.5, 0.8, 0.9, 0.99]

        data_table = pd.read_csv('output/table/0_pt_table_use_delta.csv')
        data_table = data_table.rename(columns = {'Unnamed: 0':'model_name'})
        # data_table['delta'] = data_table['model_name'].apply(lambda x: float(x[x.find("(")+1:x.find(")")]))
        for penalty in penalty_const_list:
            name = 'Resnet (' + str(penalty) + ')'
            if name in list(data_table['model_name']):
                model_list.append(name)
        model_list.append('DNN')
        # print(model_list[1])
        # print(data_table['model_name'].iloc[1])
        # print(model_list[1] == data_table['model_name'].iloc[1])
        plot_curve(data_table,model_list, save_fig, penalty_const_list,'PT',spline_or_not = True)


    if HD:
        model_list = ['HD']
        penalty_const_list = [1e-7, 1e-6, 1e-5, 1e-4, 0.001, 0.005, 0.006, 0.007,
                              0.008, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 0.8, 0.95, 0.99,  0.9999]

        data_table = pd.read_csv('output/table/0_hd_table_use_delta.csv')
        data_table = data_table.rename(columns = {'Unnamed: 0':'model_name'})
        # data_table['delta'] = data_table['model_name'].apply(lambda x: float(x[x.find("(")+1:x.find(")")]))
        for penalty in penalty_const_list:
            name = 'Resnet (' + str(penalty) + ')'
            if name in list(data_table['model_name']):
                model_list.append(name)
        model_list.append('DNN')
        # print(model_list[1])
        # print(data_table['model_name'].iloc[1])
        # print(model_list[1] == data_table['model_name'].iloc[1])
        plot_curve(data_table,model_list, save_fig, penalty_const_list,'HD',spline_or_not = True)



if __name__ == '__main__':
    CM = True
    PT = True
    HD = True
    plot_convergence(CM, PT, HD, save_fig = 1)

