#!/usr/bin/env python
# coding: utf-8
import numpy as np
import scipp as sc
from scipp import Dim
from scipp.neutron.diffraction import load_calibration
import smooth_data


def get_lambda_range():
    return 0.7, 10.35


def powder_reduction(sample='sample.nxs', calibration=None):
    """
    Simple WISH reduction workflow

    Corrections applied:
    - Normalization by monitors
    - Conversion considering calibration
    - Masking and grouping detectors into panels

    Parameters
    ----------
    sample: Nexus event file

    calibration: .cal file following Mantid's standards
        The columns correspond to detectors' IDs, offset, selection of detectors
        and groups

    Returns
    -------
    Scipp dataset containing reduced data in d-spacing

    """

    # Load data
    sample = sc.neutron.load(sample,
                             load_pulse_times=False,
                             mantid_args={'LoadMonitors': True})

    # Load calibration
    if calibration is not None:
        input_load_cal = {"InstrumentName": "WISH"}
        cal = load_calibration(calibration, mantid_args=input_load_cal)
        # Merge table with detector->spectrum mapping from sample
        # (implicitly checking that detectors between sample and calibration are the same)
        cal = sc.merge(cal, sample.labels['detector_info'].value)
        # Compute spectrum mask from detector mask
        mask = sc.groupby(cal['mask'], group='spectrum', combine=Dim.Spectrum).any(Dim.Detector)

        # Compute spectrum groups from detector groups
        g = sc.groupby(cal['group'], group='spectrum', combine=Dim.Spectrum)

        group = g.min(Dim.Detector)

        assert group == g.max(Dim.Detector), \
            "Calibration table has mismatching group for detectors in same spectrum"

        sample.labels['group'] = group.data
        sample.masks['mask'] = mask.data

    # Correct 4th monitor spectrum
    # There are 5 monitors for WISH. Only one, the fourth one, is selected for
    # correction (like in the real WISH workflow).

    # Select fourth monitor
    mon4_selected = sample.attrs['monitor4'].values

    # Smooth monitor
    mon4_smooth = smooth_data.smooth_data(mon4_selected,
                                          dim=Dim.Tof,
                                          NPoints=40)
    # Delete intermediate data
    del mon4_selected

    # Correct data
    # 1. Normalize to monitor
    # Convert to wavelength (counts and monitor)
    sample_lambda = sc.neutron.convert(sample, Dim.Tof, Dim.Wavelength)

    mon_conv = sc.neutron.convert(mon4_smooth, Dim.Tof, Dim.Wavelength)

    # Rebin monitors' data
    # The value 5615 corresponds to the value in Mantid after rebinning
    lambda_min, lambda_max = get_lambda_range()
    mon_rebin = sc.rebin(mon_conv,
                         Dim.Wavelength,
                         sc.Variable([Dim.Wavelength],
                                     unit=sc.units.angstrom,
                                     values=np.linspace(lambda_min, lambda_max, num=5615)))
    sample_lambda /= mon_rebin

    del mon_rebin, mon_conv, sample

    sample_tof = sc.neutron.convert(sample_lambda, Dim.Wavelength, Dim.Tof)

    del sample_lambda

    # 2. Convert to d-spacing taking calibration into account
    if calibration is None:
        # No calibration data, use standard convert algorithm
        sample_dspacing = sc.neutron.convert(sample_tof, Dim.Tof, Dim.DSpacing)
    else:
        # Calculate dspacing from calibration file
        sample_dspacing = sc.neutron.diffraction.convert_with_calibration(sample_tof, cal)

    del sample_tof

    # 3. Focus panels
    # Assuming sample is in Dim.DSpacing: Focus into groups
    focused = sc.groupby(sample_dspacing, group='group', combine=Dim.Group).flatten(Dim.Spectrum)

    del sample_dspacing

    # Histogram to make nice plot
    dspacing_bins = sc.Variable([Dim.DSpacing],
                                values=np.arange(1., 10., 0.001),
                                unit=sc.units.angstrom)

    focused_hist = sc.histogram(focused, dspacing_bins)

    del focused

    return focused_hist


if __name__ == "__main__":

    from scipp.plot import plot

    focused_hist = powder_reduction(sample='WISH00043525.nxs',
                                    calibration="WISH_cycle_15_4_noends_10to10_dodgytube_removed_feb2016.cal")

    plot(focused_hist)
