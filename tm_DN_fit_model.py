# required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize

# required scripts
from utils.objective_function import *

"""
Author: A. Brands

Description: fits the DN model to data using least squares algorithm.


"""

def main():

    ################### BEFORE RUNNING THE SCRIPT ###################

    # Please specify points 1-4 below.

    # 1. ----------------------
    # Choose model type that is going to be simulated.
    # (That is, either 'zhou' or 'groen' needs to be commmented out)

    model = 'zhou'
    model = 'groen'

    # 2. ----------------------
    # Define parameter space.

    # Depending on whether the variation of Zhou et al. Groen et al. is
    # fitted, the following parameter values (i.e. initial value, lower
    # bound and upper bound) need to be stated:

    # Zhou et al.: tau

    # Groen et al.: tau_pos, tau_neg, n_irf,

    # for BOTH variations: weight, shift, scale, n, sigma, tau_a

    if model == 'zhou':
        tau = 0.005
        tau_lb = 0.001
        tau_ub = 1

    elif model == 'groen':

        tau_pos = 0.005
        tau_pos_lb = 0.001
        tau_pos_ub = 1

        tau_neg = 0.005
        tau_neg_lb = 0.001
        tau_neg_ub = 1

        n_irf = 3
        n_irf_lb = 2
        n_irf_ub = 10

    weight = 0
    weight_lb = 0
    weight_ub = 1

    tau_a = 0.07
    tau_a_lb = 0.01
    tau_a_ub = 2

    n = 2
    n_lb = 0.1
    n_ub = 3

    sigma = 0.15
    sigma_lb = 0.01
    sigma_ub = 1

    shift = 0.06
    shift_lb = 0
    shift_ub = 0.1

    scale = 2
    scale_lb = 0.1
    scale_ub = 200

    # 3. ----------------------
    # Prepare data, please see points 3a and 3b.

    # 3a. Define directory where data is stored on the computer.
    dir = ''

    # 3b preprocess data.
    # For fitting the model, the data needs to be presented as an np.array with
    # dim(samples x timepts). In other words, the rows contain the samples, and
    # the columns contain the timepoints.
    # IMPORTANT: when only one sample needs to be fitted (i.e. 1 row), the
    # array should have dimensions (1, timepts) and not be a vector.

    # Below example code with a datafile where datapoints are separated with
    # a space (i.e. ' ') imported using pandas:

    # import data
    data_raw = pd.read_csv(dir, sep=' ')

    # convert to array
    data = np.array(data_raw)

    # determine shape
    shape = data.shape
    n_samples = shape[0]
    timepts = shape[1]

    # 4. ----------------------
    # Create input data which is fed into the model (i.e. stimulus timecourse).
    # The stimulus timecourse should have the same shape as the data (i.e. same
    # number of samples, n_samples, and timepoints, timepts).

    stim = np.array()

    # 4. ----------------------
    # Define sampling rate
    sample_rate = 512

    # Done! You can run the script now.

    ###########################################################################

    # initiate timeseries
    t = np.arange(timepts) * (1/sample_rate)

    # initiate params
    if model == 'zhou':

        # initiate param for model fit
        params_names = ['tau', 'weight', 'shift', 'scale', 'n', 'sigma', 'tau_a']
        x0 = [tau, weight, shift, scale, n, sigma, tau_a]
        lb = [tau_lb, weight_lb, shift_lb, scale_lb, n_lb, sigma_lb, tau_a_lb]
        up = [tau_ub, weight_ub, shift_ub, scale_ub, n_ub, sigma_ub, tau_a_ub]

    elif model == 'groen':

        # initiate param for model fit
        params_names = ['tau_pos', 'tau_neg', 'n_irf', 'weight', 'shift', 'scale', 'n', 'sigma', 'tau_a']
        x0 = [tau_pos, tau_neg, n_irf, weight, shift, scale, n, sigma, tau_a]
        lb = [tau_pos_lb, tau_neg_lb, n_irf_lb, weight_lb, shift_lb, scale_lb, n_lb, sigma_lb, tau_a_lb]
        up = [tau_pos_ub, tau_neg_ub, n_irf_ub, weight_ub, shift_ub, scale_ub, n_ub, sigma_ub, tau_a_ub]

    # fit model
    res = optimize.least_squares(objective_function, x0, args=(stim, data, sample_rate, model), bounds=(lb, up))

    # retrieve parameters
    print(res)
    popt = res.x

    # print progress
    print('\n############# Fitted params :\n')
    for k in range(len(params_names)):
        print(params_names[k] + ': ' + str(np.round(popt[k],2)))
    print('\n')


if __name__ == '__main__':
    main()
