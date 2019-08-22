import numpy as np
import scipp as sc
from scipp import Dim

n_pixel = 1000


def make_component_info(source_pos, sample_pos):
    """
    Create a dataset containing basic information about the components in an instrument.

    Currently this contains only source and sample position.
    The structure of this is subject to change.
    """
    component_info = sc.Dataset({
        'position':
        sc.Variable(dims=[Dim.Row],
                    values=[source_pos, sample_pos],
                    unit=sc.units.m)
    })
    return component_info

def make_cylinder_coords(n_pixel, n_rows=1, height=0.1, radius=4, center=[0,0,0], ttheta_min=15, ttheta_max=160):
    """
    Create a scipp variable with positions of detector pixels on a
    cylindrical detector with symmetry around y axis.  The positions
    correspond to the center of the pixels.  If the total number of
    pixels can not be evenly split into the given number of rows, a
    scipp variable with a lower number of pixels is returned.
    
    Parameters
    ----------
        n_pixel: int
            Total number of pixels on the detector
    
    Keyword parameters
    ------------------
        n_rows: int
            Number of rows to split the pixels into
            
        height: double
            Total height between lowest pixel center and highest
            
        radius: double
            Distance from symmetry axis to each pixel
            
        center: array of 3 double
            Center position of detector array (usually sample position)
            
        ttheta_min: double
            Minimum value of two theta covered by the detector
        
        ttheta_max: double
            Maximum value of two theta covered by the detector
    """
    
    if not n_pixel % n_rows == 0:
        print("The number of pixels can not be evenly distributed into the nunmber of rows")
    
    n_pixel_per_row = int(n_pixel/n_rows)
    if n_rows == 1:
        heights = [center[1]]
    else:
        heights = np.linspace(-0.5*height, 0.5*height, n_rows) + center[1]
    
    # Create a numpy array that describes detector positions
    pixel_positions = np.zeros((n_pixel_per_row*n_rows, 3))
    ttheta_array = np.linspace(ttheta_min, ttheta_max, n_pixel_per_row)

    for row in range(n_rows):
        from_index = row*n_pixel_per_row
        to_index = (row+1)*n_pixel_per_row
        pixel_positions[from_index:to_index,0] = radius*np.sin(3.14159/180*ttheta_array) + center[0]
        pixel_positions[from_index:to_index,1] = heights[row]
        pixel_positions[from_index:to_index,2] = radius*np.cos(3.14159/180*ttheta_array) + center[2]

    # Use this to initialize the pixel coordinates
    pixel_coords = sc.Variable(dims=[Dim.Position],
                               values=pixel_positions,
                               unit=sc.units.m)

    return pixel_coords

# Create dataset with a position coordinate.
# Includes labels 'component_info', a special name that `scipp.neutron.convert` will inspect for instrument information.
d = sc.Dataset(
    coords={
        Dim.Position:
        make_cylinder_coords(n_pixel, 2, radius=4),
        # TOF is optional, Mantid always has this but not needed at this point
        Dim.Tof:
        sc.Variable(dims=[Dim.Tof], values=np.arange(10.0), unit=sc.units.us)
    },
    labels={
        'component_info':
        sc.Variable(
            make_component_info(source_pos=[0, 0, -20], sample_pos=[0, 0, 0]))
    })

# Add sparse TOF coord, i.e., the equivalent to event-TOF in Mantid
tofs = sc.Variable(dims=[Dim.Position, Dim.Tof],
                   shape=[n_pixel, sc.Dimensions.Sparse],
                   unit=sc.units.us)
d['sample'] = sc.DataArray(coords={Dim.Tof: tofs})

# check positions are reasonable
print(d.coords[Dim.Position].values)

# Add some events
# Note: d.coords[Dim.Tof] gives the "dense" TOF coord, not the event-TOFs
d['sample'].coords[Dim.Tof][Dim.Position, 0].values = np.arange(10)
# The following should be equivalent but does not work yet, see scipp/#290
# d['sample'].coords[Dim.Tof].values[1] = np.arange(10)
#print(d)

dspacing = sc.neutron.convert(d, Dim.Tof, Dim.DSpacing)
print(dspacing)

# Converting event data to histogram
#hist = sc.histogram(dspacing, dspacing.coords[Dim.DSpacing])
#print(hist)

# "DiffractionFocussing" == sum? (not available yet)
# focussed = sc.sum(hist, Dim.Position)
