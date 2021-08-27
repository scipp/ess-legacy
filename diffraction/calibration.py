# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2021 Scipp contributors (https://github.com/scipp)
# @author Mads Bertelsen (original version), Céline Durniak

import numpy as np
import scipp as sc
import scippneutron as scn
import re


def load_calibration(filename, mantid_args={}):
    """
    Function that loads calibration files using the Mantid algorithm
    LoadDiffCal. This algorithm produces up to three workspaces, a
    TableWorkspace containing conversion factors between TOF and d, a
    GroupingWorkspace with detector groups and a MaskWorkspace for masking.
    The information from the TableWorkspace and GroupingWorkspace is converted
    to a Scipp dataset and returned, while the MaskWorkspace is ignored for
    now. Only the keyword paramters Filename and InstrumentName are mandatory.

    Example of use:

      from scipp.neutron.diffraction import load
      input = {"InstrumentName": "WISH"}
      cal = loadCal('cal_file.cal', mantid_args=input')

    Note that this function requires mantid to be installed and available in
    the same Python environment as scipp.

    :param str filename: The name of the cal file to be loaded.
    :param dict mantid_args : Dictionary with arguments for the
                              LoadDiffCal Mantid algorithm.
                              Currently InstrumentName is required.
    :raises: If the InstrumentName given in mantid_args is not
             valid.
    :return: A Dataset containing the calibration data and grouping.
    :rtype: Dataset
    """

    if "WorkspaceName" not in mantid_args:
        mantid_args["WorkspaceName"] = "cal_output"

    with scn.mantid.run_mantid_alg('LoadDiffCal', 
                                   Filename=filename,
                                   **mantid_args) as output:

        cal_ws = output.OutputCalWorkspace
        cal_data = scn.mantid.convert_TableWorkspace_to_dataset(cal_ws)

        # Modify units of cal_data
        cal_data["difc"].unit = sc.units.us / sc.units.angstrom
        cal_data["difa"].unit = sc.units.us / sc.units.angstrom ** 2
        cal_data["tzero"].unit = sc.units.us

        # Note that despite masking and grouping stored in separate workspaces,
        # there is no need to handle potentially mismatching ordering: All
        # workspaces have been created by the same algorithm, which should
        # guarantee ordering.
        mask_ws = output.OutputMaskWorkspace
        group_ws = output.OutputGroupingWorkspace
        rows = mask_ws.getNumberHistograms()
        mask = np.fromiter((mask_ws.readY(i)[0] for i in range(rows)),
                           count=rows,
                           dtype=np.bool_)
        group = np.fromiter((group_ws.readY(i)[0] for i in range(rows)),
                            count=rows,
                            dtype=np.int32)

        # This is deliberately not stored as a mask since that would make
        # subsequent handling, e.g., with groupby, more complicated. The mask
        # is conceptually not masking rows in this table, i.e., it is not
        # marking invalid rows, but rather describes masking for other data.
        cal_data["mask"] = sc.Variable(['row'], values=mask)
        cal_data["group"] = sc.Variable(['row'], values=group)

        cal_data.rename_dims({'row': 'detector'})
        cal_data.coords['detector'] = sc.Variable(['detector'],
                                                   values=cal_data['detid'].values.astype(np.int32))
        del cal_data['detid']

        return cal_data


def validate_calibration(calibration):
    """ Check units stored in calibration """
    if calibration["tzero"].unit != sc.units.us:
        raise ValueError("tzero must have units of `us`")
    if calibration["difc"].unit != sc.units.us / sc.units.angstrom:
        raise ValueError("difc must have units of `us / angstrom`")


def validate_dataset(dataset):
    """ Check units stored in dataset  """
    # TODO improve handling of othe units
    check_units = re.split('[*/]', str(dataset.unit))
    list_units = [item for item in dir(sc.units) if not item.startswith('_')]
    # We check if the data is expressed as density 
    # i.e. counts divided or multiplied by other unit or
    #  1/other unit * counts
    if 'counts' in check_units and \
        all([item.isnumeric() or item in list_units for item in check_units]):
        raise TypeError("Expected non-count-density unit")


def convert_with_calibration(dataset, cal):
    """ Convert from tof to dspacing taking calibration into account  """
    validate_calibration(cal)
    validate_dataset(dataset)
    output = dataset.copy()

    # 1. There may be a grouping of detectors, in which case we need to
    # apply it to the cal information first.
    if "detector_info" in list(output.coords.keys()):
        # 1a. Merge cal with detector-info, which contains information on how
        # `dataset` groups its detectors. At the same time, the coord
        # comparison in `merge` ensures that detector IDs of `dataset` match
        # those of `calibration`.
        detector_info = output.coords["detector_info"].value
        cal = sc.merge(detector_info, cal)

        # Masking and grouping information in the calibration table interferes
        # with `groupby.mean`, dropping.

        for name in ("mask", "group"):
            if name in list(cal.keys()):
                del cal[name]

        # 1b. Translate detector-based calibration information into coordinates
        # of data. We are hard-coding some information here: the existence of
        # "spectra", since we require labels named "spectrum" and a
        # corresponding dimension. Given that this is in a branch that is
        # access only if "detector_info" is present this should probably be ok.
        cal = sc.groupby(cal, group="spectrum").mean('detector')

    elif cal["tzero"].dims not in dataset.dims:
        raise ValueError("Calibration depends on dimension " +
                         cal["tzero"].dims +
                         " that is not present in the converted data " +
                         dataset.dims + ". Missing detector information?")

    # 2. Convert to tof if input is in another dimension
    if 'tof' not in list(output.coords.keys()):
        list_of_dims_for_input = output.dims
        list_of_dims_for_input.remove('spectrum')
        # TODO what happens if there are more than 1 dimension left
        dim_to_convert = list_of_dims_for_input[0]
        output = scn.convert(output, dim_to_convert, 'tof', scatter=True)

    # 3. Transform coordinate
    # all values of DIFa are equal to zero: d-spacing = (tof - TZERO) / DIFc
    if np.all(cal['difa'].data.values==0):
        # dealing with 'bins'
        output.bins.coords['dspacing'] = \
            (output.bins.coords['tof'] - cal["tzero"].data) / cal["difc"].data
        # dealing with other part of dataset
        output.coords['tof'] = \
            (output.coords['tof'] - cal["tzero"].data) / cal["difc"].data

    else:
        # DIFa non zero: tof = DIFa * d**2 + DIFc * d + TZERO. 
        # d-spacing is the positive solution of this polynomials

        # dealing with 'bins'
        output.bins.coords['dspacing'] = \
            0.5 * (- cal["difc"].data + np.sqrt(cal["difc"].data**2
                                               + 4 * cal["difa"].data
                                                   * (output.coords['tof']
                                                - cal["tzero"].data))
                  ) / cal["difa"].data

        # dealing with other part of dataset
        output.coords['tof'] = 0.5 * (- cal["difc"].data 
                                                + np.sqrt(cal["difc"].data**2 
                                                        + 4 * cal["difa"].data 
                                                        * (output.coords['tof'] 
                                                          - cal["tzero"].data))
                                                ) / cal["difa"].data

    del output.events.coords['tof']
    output.rename_dims({'tof': 'dspacing'})

    # change units
    output.coords['dspacing'].unit = sc.units.angstrom

    # transpose d-spacing if tof dimension of input dataset has more
    # than 1 dimension
    if len(output.coords['dspacing'].shape) == 2:
        output.coords['dspacing'] = sc.transpose(output.coords['dspacing'],
                                                dims=['spectrum', 'dspacing'])

    # move `position`, `source_position` and `sample_position`
    # from coordinates to attributes
    if 'sample_position' in list(output.coords.keys()):
        output.attrs['sample_position'] = output.coords['sample_position']
        del output.coords['sample_position']

    if 'source_position' in list(output.coords.keys()):
        output.attrs['source_position'] = output.coords['source_position']
        del output.coords['source_position']

    if 'position' in list(output.coords.keys()):
        output.attrs['position'] = output.coords['position']
        del output.coords['position']

    return output
