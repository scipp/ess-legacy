# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# flake8: noqa: E501
"""
This script implements a few functions in order to perform a simple data reduction of POWGEN data
using scipp.
"""

import numpy as np
import scippneutron as scn
from scipy.fft import irfft
from scipy.fft import rfft


def fft_smooth(dataset, n, order):
    """
    Performs smoothing of spectra using Butterworth filter 

    dataset: input scipp histogrammed Dataset

    n: integer to define cutoff

    order: order of the Butterworth filter
    """
    
    # n, order must be positive
    assert isinstance(n, int), 'n should be an integer'
    assert isinstance(order, int), 'order should be an integer'
    assert (n > 0 and order > 0), 'n and order should be positive'

    # coordinates of input
    assert ('tof' in dataset.dims and 'spectrum' in dataset.dims), \
        "the dimensions of the input dataset should be 'tof' and 'spectrum'"

    # check histogrammed data
    assert len(dataset.coords['tof'].values) > 2, "dataset should be histogrammed"
    
    # extract data and errors (standard deviation) to smoothen
    data = dataset.data.values
    stds = np.sqrt(dataset.data.variances)
    
    # create output dataset
    dataset_output = dataset.copy()
    
    # calculate fft for values and errors
    # TODO check how to propagate errors
    y_fft = rfft(data)
    std_fft = rfft(stds)
    
    # define and apply filter
    cutoff = y_fft.shape[1] / n
    # print(f"cutoff: {cutoff}. 2D", y_fft.shape, len(y_fft.shape))
    filter_vector = np.array([1 / (1 + pow(i / cutoff, 2*order)) for i in range(y_fft.shape[1])])
    
    yfft_filtered = np.empty_like(y_fft)
    yfft_filtered = y_fft * filter_vector[np.newaxis, :] 

    stdfft_filtered = np.empty_like(y_fft)
    stdfft_filtered = std_fft * filter_vector[np.newaxis, :]       
    
    # calculate irfft for the output
    dataset_output.data.values = irfft(yfft_filtered, 
                                       n=dataset_output.data.values.shape[-1])
    dataset_output.data.variances = irfft(
        stdfft_filtered,
        n=dataset_output.data.variances.shape[-1]
        )**2
    
    return dataset_output
    