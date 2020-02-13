import numpy as np
import scipp as sc
from scipp import Dim, Variable
from scipp.compat.mantid import load
import mantid.simpleapi as simpleapi

def absorption_dataset(input_variable, **kwargs):
    """
    This method is a straightforward wrapper exposing AbsorptionCorrection through scipp

    AbsorptionCorrection calculates an approximation of the attenuation due to absorption
    and single scattering in a generic sample shape.

    Requirements:
    - The input workspace must have units of wavelength
    - The instrument associated with the workspace must be fully defined.
      (This being a WISH-centric implementation is done with the predefined instr file)
    """

    var_x = input_variable.coords[Dim.Tof].values
    x_len = len(var_x)
    var_y = [0]*x_len
    
    # Extract sample information
    sample = input_variable.attrs['sample'].value

    # Create workspace from passed dataset
    ws = simpleapi.CreateWorkspace(DataX=var_x, DataY=var_y, UnitX="Wavelength")

    # rebin workspace
    #ws = simpleapi.Rebin(ws, input_variable.)

    # Set sample on the workspace
    ws.setSample(sample)

    # test for shape in sample - if not there, add a default vanadium sphere
    shape_xml = sample.getShape().getShapeXML()
    import xml.etree.ElementTree as etree
    root = etree.ElementTree(etree.fromstring(shape_xml)).getroot()
    if root.text == '  ':  #'<type name="userShape">  </type>'

        shape = '''<sphere id="V-sphere">
        <centre x="0.0" y="0.0" z="0.0" />
        <radius val="0.0025"/>
        </sphere>'''

        simpleapi.CreateSampleShape(ws, shape)
        simpleapi.SetSampleMaterial(ws, ChemicalFormula="V")


    # Add WISH instrument definition
    _ = simpleapi.LoadInstrument(ws, filename="WISH_Definition.xml", RewriteSpectraMap=True)

    # Run the absorption algorithm
    ws_correction = simpleapi.AbsorptionCorrection(ws, **kwargs)

    # Convert correction to Variable
    var_correction = sc.neutron.from_mantid(ws_correction)

    # The above fails with 
    # RuntimeError: ExtractMonitors is trying to return 1 output(s) but you have provided 2 variable(s).
    # These numbers must match.

    # Apply correction
    var_corrected = input_variable / var_correction

    return var_corrected

def absorption_ws(ws, **kwargs):
    """
    This method is a straightforward wrapper applying AbsorptionCorrection to the passed workspace

    AbsorptionCorrection calculates an approximation of the attenuation due to absorption
    and single scattering in a generic sample shape.

    Requirements:
    - The input workspace must have units of wavelength
    - The instrument associated with the workspace must be fully defined.
      (This being a WISH-centric implementation is done with the predefined instr file)
    """

    # Add WISH instrument definition
    _ = simpleapi.LoadInstrument(ws, filename="WISH_Definition.xml", RewriteSpectraMap=True)

    # convert to wavelength
    ws = simpleapi.ConvertUnits(ws, Target="Wavelength")

    # Run the absorption algorithm
    ws_correction = simpleapi.AbsorptionCorrection(ws, **kwargs)
    ws_corrected = ws / ws_correction

    # Convert result to Variable
    varOut = sc.neutron.from_mantid(ws_corrected)

    return ws_corrected


if __name__ == '__main__':
    # simple test
    #filename = "WISH00043525.nxs
    #filename = "164200.nxs"
    #filename = 'wish_test.nxs'
    #filename = '../TrainingCourseData/PG3_4871_event.nxs'
    filename = "164199.nxs"


    # ------------- scipp ----------------------

    # load the example file as Dataset
    dataset = sc.compat.mantid.load(filename=filename) 

      # call absorption on the scipp dataset
    absorption_dataset1 = absorption_dataset(dataset, ScatterFrom='Sample')#, ElementSize=0.5)




    # ------------- mantid ----------------------

    # load the example file as Mantid workspace directly
    workspace = simpleapi.Load(filename)
    
    # test shape
    shape = '''<sphere id="V-sphere">
    <centre x="0.0" y="0.0" z="0.0" />
    <radius val="0.0025"/>
    </sphere>'''

    simpleapi.CreateSampleShape(workspace, shape)

    simpleapi.SetSampleMaterial(workspace, ChemicalFormula="V")

    # call absorption on the mantid WS
    corrected_ws = absorption_ws(workspace, ScatterFrom='Sample')

    print("original Y = {}".format(workspace.readY(0)))
    print("corrected Y = {}".format(corrected_ws.readY(0)))



 