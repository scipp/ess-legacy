#!/usr/bin/env python
# coding: utf-8
import numpy as np
import scipp as sc
from scipp.neutron.diffraction import load_calibration
from smooth_data import smooth_data
from spline_background import bspline_background
from absorption import absorption_correction


def powder_reduction(sample='sample.nxs',
                     calibration=None,
                     vanadium=None,
                     empty_instr=None,
                     lambda_binning=(0.7, 10.35, 5615),
                     dspacing_binning=(1., 10., 9000),
                     **absorp):

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
        The columns correspond to detectors' IDs, offset, selection of detectors
        and groups

    vanadium: Nexus event file

    empty_instr: Nexus event file

    lambda_binning: min, max and number of steps for binning in wavelength
                    min and max are in Angstroms

    dspacing_binning: min, max and number of steps for binning in d-spacing
                      min and max are in Angstroms

    **absorp: dictionary containing information to correct absorption for Sample and
              Vanadium.
              There could be only up to two elements related to the correction for Vanadium: 
              the radius and height of the cylindrical sample shape.
              To distinguish them from the inputs related to the sample, their names in the 
              dictionary  are 'CylinderVanadiumRadius' and 'CylinderVanadiumHeight'. The other keys
              of the 'absorp' dictionary follow Mantid's syntax and are related to the sample data 
              only.    
              see help of Mantid's algorithm CylinderAbsorption for details
              https://docs.mantidproject.org/nightly/algorithms/CylinderAbsorption-v1.html

    Returns
    -------
    Scipp dataset containing reduced data in d-spacing

    Hints
    -----

    To plot the output data, one can histogram in d-spacing and sum according to groups
    using scipp.histogram and sc.sum, respectively.

    """
    # Load counts
    sample_data = sc.neutron.load(sample,
                                  load_pulse_times=False,
                                  mantid_args={'LoadMonitors': True})

    # Load calibration
    if calibration is not None:
        input_load_cal = {"InstrumentName": "WISH"}
        cal = load_calibration(calibration, mantid_args=input_load_cal)
        # Merge table with detector->spectrum mapping from sample
        # (implicitly checking that detectors between sample and calibration are the same)
        cal_sample = sc.merge(cal, sample_data.coords['detector_info'].value)
        # Compute spectrum mask from detector mask
        mask = sc.groupby(cal_sample['mask'], group='spectrum').any('detector')

        # Compute spectrum groups from detector groups
        g = sc.groupby(cal_sample['group'], group='spectrum')

        group = g.min('detector')

        assert group == g.max('detector'), \
            "Calibration table has mismatching group for detectors in same spectrum"

        sample_data.coords['group'] = group.data
        sample_data.masks['mask'] = mask.data

    # Correct 4th monitor spectrum
    # There are 5 monitors for WISH. Only one, the fourth one, is selected for
    # correction (like in the real WISH workflow).

    # Select fourth monitor and convert from tof to wavelength
    mon4_lambda = sc.neutron.convert(sample_data.attrs['monitor4'].values,
                                     'tof', 'wavelength')

    # Spline background
    mon4_spline_background = bspline_background(mon4_lambda,
                                                sc.Dim('wavelength'),
                                                smoothing_factor=70)

    # Smooth monitor
    mon4_smooth = smooth_data(mon4_spline_background,
                              dim='wavelength',
                              NPoints=40)
    # Delete intermediate data
    del mon4_lambda, mon4_spline_background

    # Correct data
    # 1. Normalize to monitor
    # Convert to wavelength (counts)
    sample_lambda = sc.neutron.convert(sample_data, 'tof', 'wavelength')

    # Rebin monitors' data
    lambda_min, lambda_max, number_bins = lambda_binning
    mon_rebin = sc.rebin(mon4_smooth,
                         'wavelength',
                         sc.Variable(['wavelength'],
                                     unit=sc.units.angstrom,
                                     values=np.linspace(lambda_min,
                                                        lambda_max,
                                                        num=number_bins)))
    sample_lambda /= mon_rebin

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

        # the 3 following lines of code are to place info about source and sample
        # position at the right place in the correction dataArray in order to
        # proceed to the normalization

        del correction.coords['source_position']
        del correction.coords['sample_position']
        del correction.coords['position']

        sample_lambda /= correction

        del correction

    del sample_data

    sample_tof = sc.neutron.convert(sample_lambda, 'wavelength', 'tof')

    del sample_lambda

    # 3. Convert to d-spacing taking calibration into account
    if calibration is None:
        # No calibration data, use standard convert algorithm
        sample_dspacing = sc.neutron.convert(sample_tof, 'tof', 'd-spacing')

    else:
        # Calculate dspacing from calibration file
        sample_dspacing = sc.neutron.diffraction.convert_with_calibration(sample_tof, cal_sample)
        del cal_sample

    # 4. Focus panels
    # Assuming sample is in d-spacing: Focus into groups

    focused = sc.groupby(sample_dspacing, group='group').flatten('spectrum')

    del sample_dspacing

    # 5. Vanadium correction (requires Vanadium and Empty instrument)
    if vanadium is not None and empty_instr is not None:
        print("Proceed with reduction of Vanadium data ")
        
        vana_red_focused = process_vanadium_data(vanadium,
                                                 empty_instr,
                                                 lambda_binning,
                                                 dspacing_binning,
                                                 calibration,
                                                 **absorp)

        # The following selection of groups depends on the loaded data for
        # Sample, Vanadium and Empty instrument

        focused = focused['group', 0:5].copy()

        vana_red_focused.coords['detector_info'] = focused.coords['detector_info'].copy()

        focused /= vana_red_focused

        del vana_red_focused

    return focused


# #################################################################
# Process event data for Vanadium and Empty instrument
# #################################################################
def process_event_data(file, lambda_binning):
    """
    Load and reduce event data for Vanadium and Empty instrument

    file: Nexus file to be loaded and its data reduced in this script

    lambda_binning: lambda_min, lamba_max, number_of_bins

    """

    # load nexus file
    event_data = sc.neutron.load(file,
                                 load_pulse_times=False,
                                 mantid_args={'LoadMonitors': True})

    # ################################
    # Monitor correction
    # extract monitor and convert from tof to wavelength
    mon4_lambda = sc.neutron.convert(event_data.attrs['monitor4'].values, 'tof', 'wavelength')

    mon4_smooth = smooth_data(mon4_lambda, dim='wavelength', NPoints=40)

    del mon4_lambda

    # ################################
    # vana and EC
    # convert to lambda
    event_lambda = sc.neutron.convert(event_data, 'tof', 'wavelength')

    # normalize to monitor
    lambda_min, lambda_max, number_bins = lambda_binning

    mon_rebin = sc.rebin(mon4_smooth,
                         'wavelength',
                         sc.Variable(['wavelength'],
                                     unit=sc.units.angstrom,
                                     values=np.linspace(lambda_min,
                                                        lambda_max,
                                                        num=number_bins)))

    del mon4_smooth

    event_lambda /= mon_rebin

    del mon_rebin

    edges_lambda = sc.Variable(['wavelength'],
                               values=np.linspace(lambda_min, lambda_max, number_bins),
                               unit=sc.units.angstrom)
    event_histo = sc.histogram(event_lambda, edges_lambda)

    return event_histo


def process_vanadium_data(vanadium, empty_instr,
                          lambda_binning, dspacing_binning,
                          calibration=None, **absorp):
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

    dspacing_binning: format=(d_min, d_min, number_of_bins)
                      d_min and d_max are in Angstroms

    calibration: calibration file
                 Mantid format

    **absorp: dictionary containing information to correct absorption for sample and vanadium
              only the inputs related to Vanadium will be selected to calculate the correction
              see docstrings of powder_reduction for more details
              see help of Mantid's algorithm CylinderAbsorption for details
              https://docs.mantidproject.org/nightly/algorithms/CylinderAbsorption-v1.html

    """
    vana_red = process_event_data(vanadium, lambda_binning)
    ec_red = process_event_data(empty_instr, lambda_binning)

    # vana - EC
    vana_red -= ec_red

    del ec_red

    # Absorption correction applied
    if bool(absorp):
        # The values of number_density, scattering and attenuation are hard-coded since they must
        # correspond to Vanadium. Only radius and height of the Vanadium cylindrical sample
        # shape can be set. The names of these inputs if present have to be renamed to match
        # the requirements of Mantid's algorithm CylinderAbsorption

        #  Create dictionary to calculate absorption correction for Vanadium.
        absorp_vana = {key.replace('Vanadium', 'Sample'): value for key, value in absorp.items()
                       if 'Vanadium' in key}
        absorp_vana['SampleNumberDensity'] = 0.07118
        absorp_vana['ScatteringXSection'] = 5.16
        absorp_vana['AttenuationXSection'] = 4.8756

        correction = absorption_correction(vanadium,
                                           lambda_binning,
                                           **absorp_vana)

        # the 3 following lines of code are to place info about source and sample
        # position at the right place in the correction dataArray in order to
        # proceed to the normalization

        del correction.coords['source_position']
        del correction.coords['sample_position']
        del correction.coords['position']

        correction = sc.rebin(correction,
                              'wavelength',
                              sc.Variable(['wavelength'],
                                          values=vana_red.coords['wavelength'].values,
                                          unit=sc.units.angstrom))

        vana_red /= correction

        del correction

    # convert to TOF
    vana_red_tof = sc.neutron.convert(vana_red, 'wavelength', 'tof')

    del vana_red

    # convert to d-spacing (no calibration applied)
    vana_dspacing = sc.neutron.convert(vana_red_tof, 'tof', 'd-spacing')

    del vana_red_tof

    # rebin vanadium
    dmin, dmax, number_bins_d = dspacing_binning
    dspacing_bins = sc.Variable(['d-spacing'],
                                values=np.linspace(dmin, dmax, number_bins_d),
                                unit=sc.units.angstrom)
    vana_rebin = sc.rebin(vana_dspacing, 'd-spacing', dspacing_bins)

    del vana_dspacing

    # Calibration
    # Load
    input_load_cal = {'InstrumentFilename': 'WISH_Definition.xml'}
    calvana = load_calibration(calibration, mantid_args=input_load_cal)
    # Merge table with detector->spectrum mapping from vanadium
    # (implicitly checking that detectors between vanadium and calibration are the same)
    cal_vana = sc.merge(calvana, vana_rebin.coords['detector_info'].value)

    # Compute spectrum mask from detector mask
    maskvana = sc.groupby(cal_vana['mask'], group='spectrum').any('detector')

    # Compute spectrum groups from detector groups
    gvana = sc.groupby(cal_vana['group'], group='spectrum')

    groupvana = gvana.min('detector')

    assert groupvana == gvana.max('detector'), \
        xs"Calibration table has mismatching group for detectors in same spectrum"

    vana_rebin.coords['group'] = groupvana.data
    vana_rebin.masks['mask'] = maskvana.data

    # Mask negative measured values
    vana_rebin.masks['negative_bins'] = sc.Variable(dims=['spectrum', 'd-spacing'],
                                                    values=np.less(vana_rebin.values, 0.0))

    # Focus
    focused_vana = sc.groupby(vana_rebin, group='group').flatten('spectrum')

    return focused_vana


if __name__ == "__main__":

    # The value 5615 for the number of bins in wavelength corresponds to the value in Mantid
    # after rebinning 
    sample_file = 'WISH00043525.nxs'
    cal_file = "WISH_cycle_15_4_noends_10to10_dodgytube_removed_feb2016.cal"
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
                               dspacing_binning=(1., 10., 9000),
                               **input_for_absorption)

    # #to plot the data, one could histogram in d-spacing
    # dmin, dmax, numbers_bins_d = (1., 10., 9000)
    # dspacing_bins = sc.Variable(['d-spacing'],
    #                              values = np.linspace(dmin, dmax, numbers_bins_d)
    #                              unit=sc.units.angstrom)
    # focused_hist = sc.histogram(focused, dspacing_bins)
    # from scipp.plot import plot
    # plot(focused_hist['group', 4]) # to plot data for the 5th group
