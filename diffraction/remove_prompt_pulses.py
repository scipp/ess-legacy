# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)

import numpy as np
import scipp as sc
import scippneutron as scn


def calculate_pulse_times(tmin, tmax, period):
    """
    Calculates the intervals of tof to be used for removing pulses in remove_prompt_pulses

    tmin: minimum time-of-flight in microsecconds

    tmax: maximum time-of-flight in microseconds

    period: period of the source in microseconds
    """
    times = []
    time = 0.
    # find when the first prompt pulse would be
    while time < tmin:
        time += period
    # calculate all times possible
    while time < tmax:
        times.append(time)
        time += period
    return times


def remove_prompt_pulses(dataset, width, frequency=None):
    """
    Removes the prompt pulse for a time of flight measurement.

    dataset: input scipp Dataset

    width: width of the time-of-flight (in microseconds) to remove from the data

    frequency: frequency of the source (in Hz) used to calculate the minimum time 
               of flight to filter.
               If not provided it is calculated from the attribute in the dataset
    """
    # check inputs
    assert width > 0, 'the width has to be positive'

    # dataset must have tof as 1 of its coordinates
    assert 'tof' in list(dataset.dims), 'tof must be one of the coordinates of dataset'
    assert dataset.coords['tof'].unit == sc.units.us, 'tof must be expressed in microseconds'
    
    if not frequency:
        frequency = np.median(dataset.attrs['frequency'].values.values)

    # period, min and max tof in microseconds
    period = 1000000. / frequency 
    tof_min = np.min(dataset.coords['tof'].values)  
    tof_max = np.max(dataset.coords['tof'].values)
    
    # create tof boundaries to apply mask and create boolean mask to be applied
    range = [tof_min]
    mask_bool = [False]
    
    for indx, pulse_time in enumerate(calculate_pulse_times(tof_min, tof_max, period)):
        range.extend([pulse_time, pulse_time+width])
        mask_bool.extend([True, False])
        
    # add tof_max if latest pulse_time+width < tof_max 
    # otherwise remove last False value since, in this case, the last time interval should be masked
    if tof_max > pulse_time+width:
        range.append(tof_max)
    else:
        del mask_bool[-1]
        
    ppulse4mask = sc.array(dims=['tof'],
                           unit= dataset.events.coords['tof'].unit,
                           values=range)
    dataset = sc.bin(dataset, edges=[ppulse4mask])
    dataset.masks['prompt-pulse-bin'] = sc.array(dims=['tof'], values=mask_bool)

    return dataset