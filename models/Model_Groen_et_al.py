# required packages
import numpy as np
import matplotlib.pyplot as plt

# required scripts
from utils.response_function import *

class Model_Groen_et_al:
    """ Simulation of several temporal models to predict a neuronal response given a stimulus
    time series as input.

    This class contains the following modeling components (defined as class functions):
    lin : convolves input with an Impulse Response Function (IRF)
    rectf : full-wave rectification
    exp : exponentiation

    Options for (divise) normalization (i.e. computation for the value of the denominator):
    norm : normalization of the input with a semi-saturation constant
    delay: delayed normalization of the input with a semi-saturation constant

    params
    -----------------------
    stim : array dim(T)
        contains timepoints where t = (1,...,T)
    sample_rate : int
        frequency of measurement
    tau_pos : float
        controls positive impulse response function (IRF)
    tau_neg : float
        fcontrols negative IRF
    n_irf : float
        phase delay for positive and negative IRF
    weight : float
        ratio of negative to positive IRFs
    shift : float
        time between stimulus onset and when the signal reaches the cortex
    scale : float
        response gain
    n : float
        exponent
    sigma : float
        semi-saturation constant
    tau_a : time window of adaptation

    """

    def __init__(self, stim, sample_rate, tau_pos, tau_neg, n_irf, weight, shift, scale, n, sigma, tau_a):

        # assign class variables
        self.tau_neg = tau_neg
        self.tau_pos = tau_pos
        self.n_irf = n_irf
        self.weight = weight
        self.shift = shift
        self.scale = scale
        self.n = n
        self.sigma = sigma
        self.tau_a = tau_a

        # iniate temporal variables
        self.numtimepts =  len(stim)
        self.srate = sample_rate

        # set up model parameters
        self.n_irf = max(round(self.n_irf),1)                                   # n_irf has to be an integer and can't be zero

        # compute timepoints
        self.t = np.arange(0, self.numtimepts)/self.srate

        # compute the impulse response function (used in the nominator, convolution of the stimulus)
        irf_pos = gammaPDF(self.t, self.tau_pos, self.n_irf)
        irf_neg = gammaPDF(self.t, 1.5*self.tau_neg, self.n_irf)
        self.irf = irf_pos - self.weight*irf_neg

        # create exponential decay filter (for the normalization, convolution of the linear response)
        self.norm_irf = exponential_decay(self.t, self.tau_a)

    def response_shift(self, input):
        """ Shifts response in time in the case that there is a delay betwween stimulus onset and response.  """

        # add shift to the stimulus
        sft = np.round(self.shift/(1/self.srate))
        stimtmp = np.pad(input, (int(sft), 0), 'constant', constant_values=0)
        stim = stimtmp[0: self.numtimepts]

        return stim

    def lin(self, input):
        """ Convolves input with the Impulse Resone Function (irf) """

        # compute the convolution
        linrsp = np.convolve(input, self.irf, 'full')
        linrsp = linrsp[0:self.numtimepts]

        return linrsp

    def rectf(self, input):
        """ Full-wave rectification of the input. """

        rectf = abs(input)

        return rectf

    def exp(self, input):
        """ Exponentiation of the input. """

        exp = input**self.n

        return exp

    def norm(self, input, linrsp):
        """ Normalization of the input. """

        # compute the normalized response
        demrsp = self.sigma**self.n + abs(linrsp)**self.n                       # semi-saturate + exponentiate
        normrsp = input/demrsp                                                  # divide

        return normrsp

    def norm_delay(self, input, linrsp):
        """ Introduces delay in normalization of input """

        # compute the normalized delayed response
        poolrsp = np.convolve(linrsp, self.norm_irf, 'full')
        poolrsp = poolrsp[0:self.numtimepts]
        demrsp = self.sigma**self.n + abs(poolrsp)**self.n                      # semi-saturate + exponentiate
        normrsp = input/demrsp                                                  # divide

        return normrsp
