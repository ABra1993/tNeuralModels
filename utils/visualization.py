# required packages
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from colour import Color
import numpy as np

def plot_models(t, stim, type, models, params, params_label):
    """ Plots several temporal models with a set of parameter values. The different models are:

    lin : convolves input with an Impulse Response Function (IRF)
    rectf : full-wave rectification
    exp : exponentiation
    norm : normalization of the input with a semi-saturation constant
    delay: delayed normalization of the input with a semi-saturation constant

    params
    -----------------------
    t : array dim(1, T)
        contains timepoints for one trial
    stim : array dim(1, T)
        stimulus time course
    models : array dim(n, T)
        simulated broadband data for all timepoints (T) for all the different models (n)
    params: array dim(1, n)
        parameters (n) used to simulate temporal models (for more detail, see Models.py)

    returns
    -----------------------
    fig : matplotlib figure
        timeseries of the different models given a stimulus timecourse

    """

    # initiate figure
    fig = plt.figure(figsize=(12, 6))
    plt.title('Temporal model according to ' + type.capitalize() + ' et al. (' + str(len(params)) + ' parameters)')

    # define figure layout
    plt.xlabel('Time')
    plt.ylabel('Model prediction (normalized)')
    lw=2
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # set plot colours of models
    red = Color('red')
    colors = list(red.range_to(Color('green'), len(models)))
    RGB = np.zeros((len(models), 3))
    for i in range(len(models)):
        RGB[i, :] = list(colors[i].rgb)

    # plot stimulus timecourse
    plt.plot(t, stim, color='grey', label='stimulus', lw=lw)

    # plot models
    plt.plot(t, models[0]/max(models[0]), color=RGB[0, :], label='l', lw=lw)                             # linear
    plt.plot(t, models[1]/max(models[1]), color=RGB[1, :], label='l+rect',lw=lw)                         # linear + rectf
    plt.plot(t, models[2]/max(models[2]), color=RGB[2, :], label='l+rectf+exp',lw=lw)                    # linear + rectf + exp
    plt.plot(t, models[3]/max(models[3]), color=RGB[3, :], label='l+rectf+exp+norm',lw=lw)               # linear + rectf + exp + norm
    plt.plot(t, models[4]/max(models[4]), color=RGB[4, :], label='l+rectf+exp+norm with delay',lw=lw)    # linear + rectf + exp + delayed norm

    # add params to legend
    for i in range(len(params)):
        plt.plot([], [], 'w', label=params_label[i] + str(params[i]))
    plt.legend(fontsize=6, bbox_to_anchor=(1.04, 1))

    return fig
