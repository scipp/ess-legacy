import scipp as sc
import mantid.simpleapi as simpleapi


def absorption_correction(data_array, lambda_binning=(0.7, 10.35, 5615), **mantid_args):
    """
    This method is a straightforward wrapper exposing AbsorptionCorrection through scipp

    AbsorptionCorrection calculates an approximation of the attenuation due to absorption
    and single scattering in a generic sample shape.

    Requirements:
    - The instrument associated with the workspace must be fully defined.
      (This being a WISH-centric implementation is done with the predefined instr file)

    Parameters
    ----------
    data_array: Scipp dataset with sample defined.
    lambda_binning: min, max and number of steps for binning in wavelength
    mantid_args: additional arguments to be passed to Mantid's CylinderAbsorption method.

    Returns
    -------
    Scipp dataset containing absorption correction in Wavelength units.

    """
    # We need the sample defined so exit prematurely if no sample is present
    if 'sample' not in data_array.attrs:
        raise AttributeError("Data array contains no sample.")
    
    # Create empty workspace with the WISH instrument definitions
    ws = simpleapi.LoadEmptyInstrument(Filename="WISH_Definition.xml")

    ws.getAxis(0).setUnit('Wavelength')

    # Rebin the resulting correction based on default WISH binning
    lambda_min, lambda_max, number_bins = lambda_binning
    ws = simpleapi.Rebin(ws, params=[lambda_min, number_bins, lambda_max])

    if 'sample' in data_array.attrs:
        ws.setSample(data_array.attrs['sample'].value)

    ws_correction = simpleapi.CylinderAbsorption(ws, **mantid_args)

    # We return the dataset in wavelengths for consistency.
    return sc.neutron.from_mantid(ws_correction)
