import scipp as sc
import mantid.simpleapi as simpleapi


def absorption_correction(data_array, filename, lambda_binning=(0.7, 10.35, 5615), **mantid_args):
    """
    This method is a straightforward wrapper exposing CylinderAbsorption through scipp

    CylinderAbsorption calculates an approximation of the attenuation due to absorption
    and single scattering in a 'cylindrical'' shape.

    Requirements:
    - The instrument associated with the workspace must be fully defined.
      (This being a WISH-centric implementation is done with the predefined instr file)

    Parameters
    ----------
    data_array: Scipp dataset with sample defined
    filename: Path to the file with data
    lambda_binning: min, max and number of steps for binning in wavelength
    mantid_args: additional arguments to be passed to Mantid's CylinderAbsorption method.

    Returns
    -------
    Scipp dataset containing absorption correction in Wavelength units.

    """

    # Create empty workspace with proper dimensions.
    ws = simpleapi.LoadEventNexus(filename,
        MetaDataOnly=True, LoadMonitors=False, LoadLogs=False)
    ws.getAxis(0).setUnit('Wavelength')

    # Rebin the resulting correction based on default WISH binning
    lambda_min, lambda_max, number_bins = lambda_binning
    ws = simpleapi.Rebin(ws, params=[lambda_min, number_bins, lambda_max])

    ws_correction = simpleapi.CylinderAbsorption(ws, **mantid_args)

    return sc.neutron.from_mantid(ws_correction)


