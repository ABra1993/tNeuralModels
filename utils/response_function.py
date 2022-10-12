# required packages
import numpy as np
import math

def gammaPDF(t, tau, n):
    """ Computes gamma function for a given timeseries with parameters tau and n.

    params
    -----------------------
    t : array dim(T)
        contains timepoints where t = (1,...,T)
    tau : float
        controls peak time
    n : int
        controls response decrease

    returns
    -----------------------
    y_norm : array dim(T)
        output timeseries
    """

    # computation of output
    y = (t/tau)**(n - 1) * np.exp(-t/tau) / (tau * math.factorial(n - 1))

    # normalization of output
    y_norm = y/np.sum(y)

    return y_norm


def exponential_decay(t, tau):
    """ Computes response function for a given timeseries with parameter tau.

    params
    -----------------------
    t : array dim(T)
        contains timepoints where t = (1,...,T)
    tau : float
        controls response decrease

    returns
    -----------------------
    y_norm : array dim(T)
        output timeseries
    """

    # computation of output
    y = np.exp(-t/tau)

    # normalization of output
    y_norm = y/np.sum(y)

    return y_norm
