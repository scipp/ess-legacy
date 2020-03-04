import scipp as sc
import mantid.simpleapi as simpleapi

def ensure_sample_in_ws(ws):
    """
    test for shape in sample - if not there, add a default vanadium cylinder

    Parameters
    ----------
        ws: Mantid Workspace
            Workspace with Sample defined
    Returns
    -------
    Mantid workspace containing the original sample or a default cylinder sample.

    """
    sample = ws.sample()   
    shape_xml = sample.getShape().getShapeXML()
    import xml.etree.ElementTree as etree
    root = etree.ElementTree(etree.fromstring(shape_xml)).getroot()
    # if no sample defined, the content of the shapexml element is empty.
    if root.text == '  ':
        shape = '''<cylinder id="cyl">
        <centre-of-bottom-base x="0.0" y="0.0" z="0.0" />
        <axis x="0.0" y="0.2" z="0.0"/>
        <radius val="1"/>
        <height val="10"/>
        </cylinder>'''
        simpleapi.CreateSampleShape(ws, shape)
        simpleapi.SetSampleMaterial(ws, ChemicalFormula="V")
    return ws

def absorption_correction(input_variable, **kwargs):
    """
    This method is a straightforward wrapper exposing AbsorptionCorrection through scipp

    AbsorptionCorrection calculates an approximation of the attenuation due to absorption
    and single scattering in a generic sample shape.

    Requirements:
    - The input workspace must have units of wavelength
    - The instrument associated with the workspace must be fully defined.
      (This being a WISH-centric implementation is done with the predefined instr file)

    Parameters
    ----------
    input_variable: Scipp dataset with sample defined.

    kwargs: additional arguments to be passed to Mantid's CylinderAbsorption method.

    Returns
    -------
    Scipp dataset containing absorption correction in Wavelength units.

    """

    # WISH specific binning
    lambda_binning = (0.7, 10.35, 5615)

    # Extract sample information
    sample = input_variable.attrs['sample'].value

    # Create empty workspace with the WISH instrument definitions
    ws = simpleapi.LoadEmptyInstrument(Filename="WISH_Definition.xml")

    # Assign the unit to X.
    # Notice that assigning the unit to X is more complex than to Y:
    # for Y we would have `ws.setYUnit()`
    ws.getAxis(0).setUnit('Wavelength')

    # Rebin the resulting correction based on default WISH binning
    lambda_min, lambda_max, number_bins = lambda_binning
    ws = simpleapi.Rebin(ws, params=[lambda_min, number_bins, lambda_max])

    # Set sample on the workspace
    ws.setSample(sample)

    # Make sure a sample is defined
    ws = ensure_sample_in_ws(ws)

    # Run the absorption algorithm
    ws_correction = simpleapi.CylinderAbsorption(ws, **kwargs)

    # Convert correction to DataArray
    var_correction = sc.neutron.from_mantid(ws_correction)

    # We return the dataset in wavelengths for consistency.
    return var_correction
