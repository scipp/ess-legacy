# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# flake8: noqa: E501
import numpy as np
import scipp as sc


def save_xye(dataset, filename):
    """ 
    Save histogrammed tof spectrum to .xye file for GSAS-II

    dataset: scipp dataset to be save

    filename: name of the output .xye file.
              By default, the file is created in the same folder 
              as this script.

    The header of the file has a particular syntax.  
    Please check GSAS manual for details at 
    https://subversion.xray.aps.anl.gov/EXPGUI/gsas/all/GSAS%20Manual.pdf
    """

    # check tof and spectrum in dimensions of dataset
    assert set(['tof', 'spectrum']) == set(pos_hist.dims), \
    "The dimensions of the input dataset must be 'spectrum' and 'tof'"

    # check type of dataset: binned, histogrammed
    assert len(dataset.coords['tof'].shape)==1 and dataset.coords['tof'].shape[0], \
    "The dataset has to be histogrammed in 'tof'"

    # check unit of tof
    assert dataset.coords['tof'].unit==sc.units.us, 'TOF has to be in microseconds'

    # check extension
    assert os.path.splitext(filename)[-1] in ['.gsa', '.xye', '.fxye']

    summed = sc.sum(pos_hist, 'spectrum')

    tof_coord = summed.coords['tof'].values

    # write header
    number_of_channels = summed.coords['tof'].shape[0]
    tof_min = np.min(summed.coords['tof'].values)
    tof_max = np.max(summed.coords['tof'].values)
    delta_tof_over_tof = [tof_coord[i+1]/tof_coord - 1 for i in range(len(tof_coord)-2)]
    
    if 'instrument_name' in list(summed.attrs.keys()):
        instrument = summed.attrs['instrument_name'].values
    else:
        instrument = ''
        
    if 'Filename' in list(summed.attrs.keys()):
        input_file = summed.attrs['Filename'].values
    else:
        input_file = ''

    if 'run_number' in list(summed.attrs.keys()):
        run_nb = summed.attrs['run_number'].values
    else:
        run_nb = ''
        
    if 'Wavelength' in list(summed.attrs.keys()):
        wavelength = summed.attrs['Wavelength'].values.data.values[0]
    else:
        wavelength = ''
        
    header_for_xye = (f"Instrument: {instrument} Run number: {run_nb}\n"
                      f"Filename: {input_file} Wavelength: {wavelength} A\n"
                      f"BANK 1 {number_of_channels} {number_of_channels} "
                      f"SLOG {tof_min} {tof_max} {np.average(delta_tof_over_tof)} 0 FXYE")
    
    print(header_for_xye)

    x = 0.5 * (summed.coords['tof'].values[:-1] + summed.coords['tof'].values[1:])
    y = summed.data.values
    err = summed.data.variances
    
    np.savetxt(filename, 
               np.c_[x, y, err],
               header=header_for_xye, 
               fmt='%f', 
               delimiter=8*' ', 
               comments='') 

    return
