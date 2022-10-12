# required models
import numpy as np

# required scripts
from models.Model_Zhou_et_al import Model_Zhou_et_al
from models.Model_Groen_et_al import Model_Groen_et_al

def objective_function(params, stim, data, sample_rate, model):

    # compute model and fit
    fit = 0
    for i in range(len(stim)):

        # simulate model
        model_run = compute_model(stim[i, :], sample_rate, params, model)

        # compute cost
        fit = fit + np.sum((model_run - data[i, :])**2)

    return fit

def compute_model(stim, sample_rate, params, model):

    # retrieve model
    if model == 'zhou':

        # define params
        tau = params[0]
        weight = params[1]
        shift = params[2]
        scale = params[3]
        n = params[4]
        sigma = params[5]
        tau_a = params[6]

        # initiate model
        model = Model_Zhou_et_al(stim, sample_rate, tau, weight, shift, scale, n, sigma, tau_a)

    elif model == 'groen':

        # define params
        tau_pos = params[0]
        tau_neg = params[1]
        n_irf = params[2]
        weight = params[3]
        shift = params[4]
        scale = params[5]
        n = params[6]
        sigma = params[7]
        tau_a = params[8]

        # initiate model
        model = Model_Groen_et_al(stim, sample_rate, tau_pos, tau_neg, n_irf, weight, shift, scale, n, sigma, tau_a)

    # introduce shift
    stim_shift = model.response_shift(stim)

    # compute delayed normalisation model
    linear = model.lin(stim_shift)
    linear_rectf = model.rectf(linear)
    linear_rectf_exp = model.exp(linear_rectf)
    linear_rectf_exp_norm_delay = model.norm_delay(linear_rectf_exp, linear)

    # scale model
    linear_rectf_exp_norm_delay = linear_rectf_exp_norm_delay * scale

    return linear_rectf_exp_norm_delay
