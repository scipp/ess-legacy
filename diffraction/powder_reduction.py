#!/usr/bin/env python
# coding: utf-8
import numpy as np
import scipp as sc
import scippneutron as scn
from calibration import load_calibration
from calibration import convert_with_calibration
from smooth_data import smooth_data
from absorption import absorption_correction
from scippneutron.tof.conversions import beamline, elastic


def powder_reduction(sample='sample.nxs',
                     calibration=None ,
                     vanadium=None,
                     empty_instr=None,
                     lambda_binning: tuple = (0.7, 10.35, 5615),
                     **absorp) -> sc.DataArray:

    """
    Simple WISH reduction workflow

    Note
    ----

    The sample data were not recorded using the same layout
    of WISH as the Vanadium and empty instrument. That's why:
    - loading calibration for Vanadium used a different IDF
    - the Vanadium correction involved cropping the sample data
      to the first 5 groups (panels)
    ----

    Corrections applied:
    - Vanadium correction
    - Absorption correction
    - Normalization by monitors
    - Conversion considering calibration
    - Masking and grouping detectors into panels

    Parameters
    ----------
    sample: Nexus event file

    calibration: .cal file following Mantid's standards
        The columns correspond to detectors' IDs, offset, selection of 
        detectors and groups

    vanadium: Nexus event file

    empty_instr: Nexus event file

    lambda_binning: min, max and number of steps for binning in wavelength
                    min and max are in Angstroms

    **absorp: dictionary containing information to correct absorption for
              Sample and Vanadium.
              There could be only up to two elements related to the correction
              for Vanadium:
              the radius and height of the cylindrical sample shape.
              To distinguish them from the inputs related to the sample, their
              names in the dictionary  are 'CylinderVanadiumRadius' and
              'CylinderVanadiumHeight'. The other keysof the 'absorp'
              dictionary follow Mantid's syntax and are related to the sample
              data only.
              See help of Mantid's algorithm CylinderAbsorption for details
              https://docs.mantidproject.org/nightly/algorithms/CylinderAbsorption-v1.html

    Returns
    -------
    Scipp dataset containing reduced data in d-spacing

    Hints
    -----

    To plot the output data, one can histogram in d-spacing and sum according
    to groups using scipp.histogram and sc.sum, respectively.

    """
    # Load counts
    sample_data = scn.load(sample,
                           advanced_geometry=True,
                           load_pulse_times=False,
                           mantid_args={'LoadMonitors': True})

    # Load calibration
    if calibration is None:
        raise ValueError('Calibration should be defined to run this reduction script.')
      
    input_load_cal = {"InstrumentName": "WISH"}
    cal = load_calibration(calibration, mantid_args=input_load_cal)
    # Merge table with detector->spectrum mapping from sample
    # (implicitly checking that detectors between sample and calibration
    # are the same)
    cal_sample = sc.merge(cal, sample_data.coords['detector_info'].value)
    # Compute spectrum mask from detector mask
    mask = sc.groupby(cal_sample['mask'], group='spectrum').any('detector')

    # Compute spectrum groups from detector groups
    g = sc.groupby(cal_sample['group'], group='spectrum')

    group = g.min('detector')

    assert sc.identical(group, g.max('detector')), \
        "Calibration table has mismatching group for detectors in same spectrum"

    sample_data.coords['group'] = group.data
    sample_data.masks['mask'] = mask.data

    del cal

    # Correct 4th monitor spectrum
    # There are 5 monitors for WISH. Only one, the fourth one, is selected for
    # correction (like in the real WISH workflow).

    # Select fourth monitor and convert from tof to wavelength
    mon4_lambda = scn.convert(sample_data.attrs['monitor4'].values,
                              'tof',
                              'wavelength',
                              scatter=False)

    # Smooth monitor
    mon4_smooth = smooth_data(mon4_lambda,
                              dim='wavelength',
                              NPoints=40)
    # Delete intermediate data
    del mon4_lambda

    # Correct data
    # 1. Normalize to monitor
    # Convert to wavelength (counts)
    graph = {**beamline(scatter=True), **elastic("tof")}
    sample_lambda = sample_data.transform_coords("wavelength", graph=graph)

    del sample_data

    # Rebin monitors' data
    lambda_min, lambda_max, number_bins = lambda_binning

    edges_lambda = sc.Variable(dims=['wavelength'],
                               unit=sc.units.angstrom,
                               values=np.linspace(lambda_min,
                                                  lambda_max,
                                                  num=number_bins))
    mon_rebin = sc.rebin(mon4_smooth,
                         'wavelength',
                         edges_lambda)

    # Realign sample data
    sample_lambda = sample_lambda.bins / sc.lookup(func=mon_rebin,
                                                   dim='wavelength')

    del mon_rebin, mon4_smooth

    # 2. absorption correction
    if bool(absorp):
        # Copy dictionary of absorption parameters
        absorp_sample = absorp.copy()
        # Remove input related to Vanadium if present in absorp dictionary
        found_vana_info = [key for key in absorp_sample.keys() if 'Vanadium' in key]

        for item in found_vana_info:
            absorp_sample.pop(item, None)

        # Calculate absorption correction for sample data
        correction = absorption_correction(sample,
                                           lambda_binning,
                                           **absorp_sample)

        # the 3 following lines of code are to place info about source and
        # sample position at the right place in the correction dataArray in
        # order to proceed to the normalization

        del correction.coords['source_position']
        del correction.coords['sample_position']
        del correction.coords['position']

        correction_rebin = sc.rebin(correction, 'wavelength', edges_lambda)

        del correction

        sample_lambda = sample_lambda.bins / sc.lookup(func=correction_rebin,
                                                       dim='wavelength')

    # 3. Convert to d-spacing with option of taking calibration into account
    # Convert to tof before converting to d-spacing 
    # (the graph used in the 2nd step requires tof as input dimension)
    sample_tof = sample_lambda.transform_coords("tof", graph=graph)
    
    # remove wavelength coordinate and set tof as one of the main ones
    sample_tof = sample_tof.rename_dims({'wavelength': 'tof'})
 
    del sample_tof.coords['wavelength']  # free up some memory, if not needed any more
    
    if sample_tof.bins is not None:
        del sample_tof.bins.coords['wavelength']  # free up a lot of memory
        
    # Calculate dspacing from calibration file
    sample_dspacing = convert_with_calibration(sample_tof, cal_sample)

    del cal_sample, sample_lambda

    # 4. Focus panels
    # Assuming sample is in d-spacing: Focus into groups
    focused = sc.groupby(sample_dspacing, group='group').bins.concat('spectrum')

    del sample_dspacing

    # 5. Vanadium correction (requires Vanadium and Empty instrument)
    if vanadium is not None and empty_instr is not None:
        print("Proceed with reduction of Vanadium data ")

        vana_red_focused = process_vanadium_data(vanadium,
                                                 empty_instr,
                                                 lambda_binning,
                                                 calibration,
                                                 **absorp)

        # The following selection of groups depends on the loaded data for
        # Sample, Vanadium and Empty instrument
        focused = focused['group', 0:5].copy()

        # histogram vanadium for normalizing + cleaning 'metadata'
        d_min, d_max, number_dbins = (1., 10., 2000)

        edges_dspacing = sc.Variable(dims=['dspacing'],
                                     unit=sc.units.angstrom,
                                     values=np.linspace(d_min,
                                                        d_max,
                                                        num=number_dbins))
        vana_histo = sc.histogram(vana_red_focused, bins=edges_dspacing)

        del vana_red_focused

        vana_histo.coords['detector_info'] = focused.coords['detector_info'].copy()

        # normalize by vanadium
        result = focused.bins / sc.lookup(func=vana_histo, dim='dspacing')

        del vana_histo, focused

    else:
        # If no Vanadium correction
        result = focused.bins

    return result



# #################################################################
# Process event data for Vanadium and Empty instrument
# #################################################################
def process_event_data(file, lambda_binning: tuple) -> sc.DataArray:
    """
    Load and reduce event data (used for Vanadium and Empty instrument)

    file: Nexus file to be loaded and its data reduced in this script

    lambda_binning: lambda_min, lamba_max, number_of_bins

    """

    # load nexus file
    event_data = scn.load(file,
                          advanced_geometry=True,
                          load_pulse_times=False,
                          mantid_args={'LoadMonitors': True})

    # ################################
    # Monitor correction
    # extract monitor and convert from tof to wavelength
    mon4_lambda = scn.convert(event_data.attrs['monitor4'].values,
                              'tof',
                              'wavelength',
                              scatter=False)

    mon4_smooth = smooth_data(mon4_lambda, dim='wavelength', NPoints=40)

    del mon4_lambda

    # ################################
    # vana and EC
    # convert to lambda
    graph = {**beamline(scatter=True), **elastic("tof")}
    event_lambda = event_data.transform_coords("wavelength", graph=graph)

    # normalize to monitor
    lambda_min, lambda_max, number_bins = lambda_binning

    edges_lambda = sc.Variable(dims=['wavelength'],
                               unit=sc.units.angstrom,
                               values=np.linspace(lambda_min,
                                                  lambda_max,
                                                  num=number_bins))

    mon_rebin = sc.rebin(mon4_smooth,
                         'wavelength',
                         edges_lambda)

    del mon4_smooth

    event_lambda_norm = event_lambda.bins / sc.lookup(func=mon_rebin,
                                                      dim='wavelength')

    del mon_rebin, event_lambda

    return event_lambda_norm


def process_vanadium_data(vanadium,
                          empty_instr,
                          lambda_binning: tuple,
                          calibration=None,
                          **absorp) -> sc.DataArray:
    """
    Create corrected vanadium dataset

    Correction applied to Vanadium data only
    1. Subtract empty instrument
    2. Correct absorption
    3. Use calibration for grouping
    4. Focus into groups

    Parameters
    ----------
    vanadium : Vanadium nexus datafile

    empty_instr: Empty instrument nexus file

    lambda_binning: format=(lambda_min, lambda_min, number_of_bins)
                    lambda_min and lambda_max are in Angstroms

    calibration: calibration file
                 Mantid format

    **absorp: dictionary containing information to correct absorption for
              sample and vanadium
              Only the inputs related to Vanadium will be selected to calculate
              the correction
              see docstrings of powder_reduction for more details
              see help of Mantid's algorithm CylinderAbsorption for details
        https://docs.mantidproject.org/nightly/algorithms/CylinderAbsorption-v1.html

    """
    vana_red = process_event_data(vanadium, lambda_binning)
    ec_red = process_event_data(empty_instr, lambda_binning)

    # remove 'spectrum' from wavelength coordinate and match this coordinate
    # between Vanadium and Empty instrument data
    min_lambda = vana_red.coords['wavelength'].values[:, 0].min()
    max_lambda = vana_red.coords['wavelength'].values[:, 1].max()
    vana_red.coords['wavelength'] = sc.Variable(dims=['wavelength'],
                                                unit=sc.units.angstrom,
                                                values=np.linspace(min_lambda,
                                                                   max_lambda,
                                                                   num=2))

    ec_red.coords['wavelength'] = sc.Variable(dims=['wavelength'],
                                              unit=sc.units.angstrom,
                                              values=np.linspace(min_lambda,
                                                                 max_lambda,
                                                                 num=2))

    # vana - EC
    ec_red.coords['wavelength'] = vana_red.coords['wavelength']
    vana_red.bins.concatenate(-ec_red, out=vana_red)

    del ec_red

    # Absorption correction applied
    if bool(absorp):
        # The values of number_density, scattering and attenuation are
        # hard-coded since they must correspond to Vanadium. Only radius and
        # height of the Vanadium cylindrical sample shape can be set. The
        # names of these inputs if present have to be renamed to match
        # the requirements of Mantid's algorithm CylinderAbsorption

        #  Create dictionary to calculate absorption correction for Vanadium.
        absorp_vana = {key.replace('Vanadium', 'Sample'):
                           value for key, value in absorp.items()
                       if 'Vanadium' in key}
        absorp_vana['SampleNumberDensity'] = 0.07118
        absorp_vana['ScatteringXSection'] = 5.16
        absorp_vana['AttenuationXSection'] = 4.8756

        correction = absorption_correction(vanadium,
                                           lambda_binning,
                                           **absorp_vana)

        # the 3 following lines of code are to place info about source and
        # sample position at the right place in the correction dataArray in 
        # order to proceed to the normalization

        del correction.coords['source_position']
        del correction.coords['sample_position']
        del correction.coords['position']

        lambda_min, lambda_max, number_bins = lambda_binning

        edges_lambda = sc.Variable(dims=['wavelength'],
                                   unit=sc.units.angstrom,
                                   values=np.linspace(lambda_min,
                                                       lambda_max,
                                                       num=number_bins))

        correction = sc.rebin(correction, 'wavelength', edges_lambda)

        vana_red = vana_red.bins / sc.lookup(func=correction,
                                             dim='wavelength')

        del correction

    # Calibration
    # Load
    input_load_cal = {'InstrumentFilename': 'WISH_Definition.xml'}
    calvana = load_calibration(calibration, mantid_args=input_load_cal)
    # Merge table with detector->spectrum mapping from vanadium
    # (implicitly checking that detectors between vanadium and
    # calibration are the same)
    cal_vana = sc.merge(calvana, vana_red.coords['detector_info'].value)

    del calvana

    # Compute spectrum mask from detector mask
    maskvana = sc.groupby(cal_vana['mask'], group='spectrum').any('detector')

    # Compute spectrum groups from detector groups
    gvana = sc.groupby(cal_vana['group'], group='spectrum')

    groupvana = gvana.min('detector')

    assert sc.identical(groupvana, gvana.max('detector')), \
        ("Calibration table has mismatching group "
         "for detectors in same spectrum")

    vana_red.coords['group'] = groupvana.data
    vana_red.masks['mask'] = maskvana.data
    
    # convert to tof: required input unit for converting to d-spacing with calibration
    graph = {**beamline(scatter=True), **elastic("tof")}
    vana_tof = vana_red.transform_coords("tof", graph=graph)
    # clean-up unused 'wavelength 'coordinates
    vana_tof = vana_tof.rename_dims({'wavelength': 'tof'})
    del vana_tof.coords['wavelength']  # free up some memory, if not needed any more
    if vana_tof.bins is not None:
        del vana_tof.bins.coords['wavelength']  # free up a lot of memory
    
    # convert to d-spacing with calibration
    vana_dspacing = convert_with_calibration(vana_tof, cal_vana)

    del vana_red, cal_vana

    # Focus
    focused_vana = \
        sc.groupby(vana_dspacing, group='group').bins.concat('spectrum')

    del vana_dspacing

    return focused_vana


if __name__ == "__main__":

    # The value 5615 for the number of bins in wavelength corresponds to the
    # value in Mantid after rebinning
    sample_file = 'WISH00043525.nxs'
    cal_file = 'WISH_cycle_15_4_noends_10to10_dodgytube_removed_feb2016.cal'
    vanadium_file = 'WISH00019612.nxs'
    empty_instrument_file = 'WISH00019618.nxs'
    input_for_absorption = {'AttenuationXSection': 2.595,
                            'ScatteringXSection': 5.463,
                            'SampleNumberDensity': 0.025,
                            'CylinderSampleRadius': 4.,
                            'CylinderSampleHeight': 0.55,
                            'CylinderVanadiumRadius': 4.,
                            'CylinderVanadiumHeight': 0.55}

    focused = powder_reduction(sample=sample_file,
                               calibration=cal_file,
                               vanadium=vanadium_file,
                               empty_instr=empty_instrument_file,
                               lambda_binning=(0.7, 10.35, 5615),
                               **input_for_absorption)

    # # to plot the data, one could histogram in d-spacing
    # dmin, dmax, numbers_bins_d = (1., 10., 2000)
    # dspacing_bins = sc.Variable(dims=['dspacing'],
    #                             values=np.linspace(dmin, dmax, numbers_bins_d),
    #                             unit=sc.units.angstrom)
    # focused_hist = sc.histogram(focused, bins=dspacing_bins)
    # import matplotlib.pyplot as plt
    # plt.plot(focused_hist['group', 0].values)
    # plt.show()
