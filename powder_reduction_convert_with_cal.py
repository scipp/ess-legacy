import scipp as sc
from scipp import Dim


def powder_reduction(data, calibration=None):
    """
    Takes a dataset with tof data and detector positions and converts to
    d-spacing. Can use calibration data which needs to contain the fields
    difc and tzero.

    Parameters
    ----------
        data: scipp Dataset
            The input data to be converted

    Keyword parameters
    ------------------
        calibration: scipp Dataset
            Dataset containing difc and tzero for all detectors

    """

    if calibration is None:
        # No calibration data, use standard convert algorithm
        dspacing = sc.neutron.convert(data, Dim.Tof, Dim.DSpacing)
    else:
        # Calculate dspacing from calibration file
        dspacing = sc.neutron.diffraction.convert_with_calibration(data,
                                                                   calibration)

    return dspacing

    # Conversion to histogram still does not work, even when data is realistic
    # histogram = sc.histogram(dspacing, dspacing.coords[Dim.DSpacing])

    # "DiffractionFocussing" == sum? (not available yet)
    # focussed = sc.sum(hist, Dim.Position)
