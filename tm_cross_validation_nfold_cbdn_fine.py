# required packages
# import numpy as np
import cupy as np
import time
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pandas as pd
import sys
import random
import getopt

# required scripts
from ecog_data.import_files import *
from ecog_data.metrics import *
from ecog_data.ecog_visualization import *
from ecog_data.data_selection import *
from temporal_models.generate_stimulus_timecourse import *
from temporal_models.tm_objective import *
from temporal_models.utils import *

"""
Author: A. Brands

Outputs: .txt file

    # model fit
        r_sq_pd.to_csv(dir+'Documents/code/ECoG_adaptation/temporal_models/data/cross_validation/r_sq_' + model + '.txt', sep=' ', header=True)

    # parameter values
        param_pd.to_csv(dir+'Documents/code/ECoG_adaptation/temporal_models/data/cross_validation/param_' + model + '.txt', sep=' ', header=True)

Description: Cross-validation of the model where each image category is fitted a separate scaling parameter (data averaged over ALL subjects).

    The number of folds is equal to the number of training samples.
        Example: when using only onepulse trials average for preferred and nonpreferred trials, nFolds = 6 (i.e. 6 ISIs)

    command in terminal:

    python tm_cross_validation.py -model -data_fit

        --> model: zhou_l, zhou, cbdn, groen, brands
        --> data_fit: 0, 1, 2, 3 (type of trials used to fit the data)
"""

def main():

    # model hyperparameters
    max_nfev = 10000
    sample_rate = 512
    broadband = True
    CV_th = 20

    # define root
    linux = True # indicates whether script runs on mac or linux
    if linux:
        dir = '/home/amber/ownCloud/'
    else:
        dir = '/Users/a.m.brandsuva.nl/surfdrive/'

    # try to catch the parsers for command line options (check model)
    models = ['zhou', 'zhou_l', 'cbdn', 'cbdn_fine', 'groen', 'brands']
    model = sys.argv[1]
    if model != 'cbdn_fine':
        data_fit = int(sys.argv[2])

    if model not in models:
        sys.exit('Proposed model does not exist...')
    else:
        print(30*'-')
        print('Model: ' + model)
        print(30*'-')

    # define number of Folds
    if (model in ['zhou', 'cbdn', 'groen', 'brands']):
        if data_fit == 4:
            nFolds = 12
    elif (model == 'zhou_l'):
        if data_fit == 4:
            nFolds = 6
    elif model == 'cbdn_fine':
        nFolds = 12

    # import files
    stim_cat = np.loadtxt(dir+'Documents/code/ECoG_adaptation/predefined_variables/cond_stim.txt', dtype=str)
    ISI = np.loadtxt(dir+'Documents/code/ECoG_adaptation/predefined_variables/cond_stim.txt', dtype=str)
    t = np.loadtxt(dir+'Documents/code/ECoG_adaptation/predefined_variables/t.txt')

    # add timepts
    t_str = []
    for j in range(len(t)):
        t_str.append(str(j))

    # obtain index and name of responsiv electrodes
    responsive_electrodes = pd.read_csv(dir+'Documents/code/ECoG_adaptation/ecog_data/electrodes_responsive/all_responsive_electrodes_VA_' + str(CV_th) + '.txt', header=0, delimiter=' ', index_col=0)

    # responsive_electrodes = responsive_electrodes[responsive_electrodes.subject != 'sub-som748']
    # responsive_electrodes.reset_index(drop=True, inplace=True)
    n_electrodes = len(responsive_electrodes)
    print(responsive_electrodes)

    # import pandas dataframe to save CV values
    r_sq_pd = pd.read_csv(dir+'Documents/code/ECoG_adaptation/temporal_models/data/cross_validation/r_sq_' + model + '.txt', delimiter=' ', header=0, index_col=0)
    param_pd = pd.read_csv(dir+'Documents/code/ECoG_adaptation/temporal_models/data/cross_validation/param_' + model + '.txt', delimiter=' ', header=0, index_col=0)

    # set model parameters
    if model == 'zhou_l':
        params_names, x0, lb, ub = fit_model('zhou')
    else:
        params_names, x0, lb, ub = fit_model(model)
    # print(params_names)
    # print(x0)
    # print(lb)
    # print(ub)

    # # initiate dataframe for performance per fold
    # r_sq_pd = pd.DataFrame()
    # r_sq_pd['subject'] = responsive_electrodes.loc[:, 'subject']
    # r_sq_pd['electrode_idx'] = responsive_electrodes.loc[:, 'electrode_idx']
    # r_sq_pd['electrode'] = responsive_electrodes.loc[:, 'electrode']
    # r_sq_pd['r2'] = np.zeros(n_electrodes)
    # for i in range(nFolds):
    #     r_sq_pd[str(i + 1)] = np.zeros(n_electrodes)
    #
    # # initiate dataframe for parameter values per fold
    # param_pd = pd.DataFrame()
    # param_pd['subject'] = responsive_electrodes.loc[:, 'subject']
    # param_pd['electrode_idx'] = responsive_electrodes.loc[:, 'electrode_idx']
    # param_pd['electrode'] = responsive_electrodes.loc[:, 'electrode']
    # param_pd['r2'] = np.zeros(n_electrodes)
    # for k in range(len(params_names)):
    #     param_pd[params_names[k]] = np.zeros(n_electrodes)

    # initiate pred array
    r_sq = np.zeros(n_electrodes)

    # iterate over electrodes and fit model
    current_subject = ''
    for i in range(n_electrodes):
    # for i in range(1):

        # defines electrode and subject
        subject = responsive_electrodes.loc[i, 'subject']
        electrode_name = responsive_electrodes.loc[i, 'electrode']
        electrode_idx = int(responsive_electrodes.loc[i, 'electrode_idx'])

        try: # create electrode directory
            os.mkdir(dir+'Documents/code/ECoG_adaptation/figures/temporal_models/cross_validation/per_electrode/' + subject + '_' + electrode_name)
        except:
            print('Electrode folder already exists...')

        # print progress
        print('Performing cross validation for ', subject, ': ', electrode_name, '...')

        if subject != current_subject:

            # update subject
            current_subject = subject

            # import info
            t, events, channels, electrodes = import_info(subject, dir)

            # load info excluded epochs
            if broadband:
                excluded_epochs = pd.read_csv(dir+'Documents/code/ECoG_adaptation/ecog_data/electrodes_responsive/' + subject + '_excluded_epochs_broadband.txt', sep=' ', header=0, dtype=int)
            else:
                excluded_epochs = pd.read_csv(dir+'Documents/code/ECoG_adaptation/ecog_data/electrodes_responsive/' + subject + '_excluded_epochs_voltage.txt', sep=' ', header=0, dtype=int)

        # import data to be fitted
        y = pd.read_csv(dir+'Documents/code/ECoG_adaptation/temporal_models/data/model_fit/per_electrode/' + subject + '_' + electrode_name + '/cbDN_fine/all_trials.txt', sep=' ', header=0)
        print(y)

        # generate stimulus time courses
        stim = y.copy()
        for j in range(len(stim)):
            stim.iloc[j, 3:3+len(t)] = generate_stimulus_timecourse(stim.loc[j, 'trial'], int(stim.loc[j, 'ISI']), dir)

        # # create indices to select train and test data
        # y_idx = np.arange(len(stim)).tolist()
        # # print(y_idx)

        # split into six categories
        y_idx_per_cat = []
        for j in range(len(stim_cat)):
            temp = y[y.cat == stim_cat[j]].index.tolist()
            y_idx_per_cat.append(temp)

        # cross-validate
        r_sq_temp = np.zeros((nFolds)) # held-out test samples
        param_pd_temp = np.zeros((nFolds, len(params_names)))
        for fold_idx in range(nFolds): # cross-validate leave-one-out approach
        # for fold_idx in range(3): # cross-validate leave-one-out approach

            # initiate figure
            fig, axs = plt.subplots(1, 6, figsize=(10,2))

            # start time
            startTime = time.time()

            # get test indices and create train data (pseudo-random held-out data)
            test_idx = []
            for j in range(len(stim_cat)):
                test_idx.append(random.sample(y_idx_per_cat[j], 1)[0])          # choose random test sample
                y_idx_per_cat[j].remove(test_idx[j])                            # remove from test indices

        #     # train indices
        #     test_idx = random.sample(y_idx, 6)
        #     # print(test_idx)

            # TEST indices
            train_idx = np.arange(len(stim)).tolist()
            train_idx.remove(test_idx[0])
            train_idx.remove(test_idx[1])
            train_idx.remove(test_idx[2])
            train_idx.remove(test_idx[3])
            train_idx.remove(test_idx[4])
            train_idx.remove(test_idx[5])

            # # update list
            # y_idx.remove(test_idx[0])
            # y_idx.remove(test_idx[1])
            # y_idx.remove(test_idx[2])
            # y_idx.remove(test_idx[3])
            # y_idx.remove(test_idx[4])
            # y_idx.remove(test_idx[5])
            # # print(X_idx)

            # select training and test data
            X_train = np.array(stim.loc[train_idx, t_str])
            y_train = np.array(y.loc[train_idx, t_str])

            X_test = np.array(stim.loc[test_idx, t_str])
            y_test = np.array(y.loc[test_idx, t_str])

            # print(X_train)
            # print(y_train)
            #
            # print(X_test)
            # print(y_test)

            # determine conditions
            # print(y)
            info = y.iloc[train_idx, 0:3]
            info.reset_index(inplace=True, drop=True)
            # print(info)

            # fit model
            np.seterr(divide='ignore', invalid='ignore') # inhibit printing division errors
            res = optimize.least_squares(objective_cbDN_fine, x0, args=(X_train, y_train, info, sample_rate, dir), max_nfev=max_nfev, bounds=(lb, ub))

            # retrieve parameters
            popt = res.x

            # print progress
            print(30*'-')
            print('Fitted params for electrode: ' + electrode_name + ' (' + subject + '):')
            print('(model: ' + model.capitalize() + ')\n')
            for k in range(len(params_names)):
                print(params_names[k] + ': ' + str(popt[k]))
            param_pd_temp[fold_idx, :] = popt
            print(30*'-')

            # test on held-out set
            r_sq = np.zeros(len(X_test))
            for k in range(len(X_test)):

                # cross-validate
                stim_temp, pred_temp = model_cbDN_fine(X_test[k], stim.loc[test_idx[k], 'trial'], int(stim.loc[test_idx[k], 'ISI']), stim.loc[test_idx[k], 'cat'], sample_rate, popt, dir)
                r_sq = r_squared(y_test[k], pred_temp)
                # print(pred_temp)

                # plot results
                axs[k].plot(y_test[k], 'k')
                axs[k].plot(pred_temp, 'r')
                axs[k].set_title('Trial: ' + y.loc[test_idx[k], 'trial'] + '\nImg. class: ' + y.loc[test_idx[k], 'cat'] + '\nISI: ' + str(y.loc[test_idx[k], 'ISI']), fontsize=8)

            r_sq_temp[fold_idx] = np.mean(r_sq)

            # add fold to pd dataframe
            r_sq_pd.loc[i, str(fold_idx+1)] = np.round(r_sq_temp[fold_idx], 4)

            # print progress
            print('Fold ' + str(fold_idx+1) + ': R2 for (', subject + ',', electrode_name, ') held-out data is', np.round(r_sq_temp[fold_idx], 2), '.')

            # save figure
            # plt.show()
            plt.tight_layout()
            plt.savefig(dir+'Documents/code/ECoG_adaptation/figures/temporal_models/cross_validation/per_electrode/' + subject + '_' + electrode_name + '/' + model + '_fold' + str(fold_idx+1))

            # determine time it took to run script (check GPU-access)
            executionTime = (time.time() - startTime)
            print('Execution time in seconds: ' + str(executionTime))

            # close figure
            plt.close()

        # average prediction for held-out set
        r_sq_mean = np.mean(r_sq_temp)
        r_sq_pd.loc[i, 'r2'] = np.round(r_sq_mean, 4)
        param_pd.loc[i, 'r2'] = np.round(r_sq_mean, 4)

        # average parameter values over held-out sets
        param_mean = np.mean(param_pd_temp, axis=0)
        param_pd.loc[i, params_names] = np.round(param_mean, 4)

        # print progress
        print('\n')
        print(60*'#')
        print('Done! R2 for (', subject + ',', electrode_name, ') test data is', np.round(r_sq_mean, 2), '.')
        print(60*'#', '\n')

        # save param values
        r_sq_pd.to_csv(dir+'Documents/code/ECoG_adaptation/temporal_models/data/cross_validation/r_sq_' + model + '.txt', sep=' ', header=True)
        param_pd.to_csv(dir+'Documents/code/ECoG_adaptation/temporal_models/data/cross_validation/param_' + model + '.txt', sep=' ', header=True)


if __name__ == '__main__':
    main()
