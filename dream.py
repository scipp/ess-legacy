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


# Create dataset with a position coordinate.
# Includes labels 'component_info', a special name that `scipp.neutron.convert` will inspect for instrument information.
d = sc.Dataset(
    coords={
        Dim.Position:
        sc.Variable(dims=[Dim.Position],
                    shape=[n_pixel],
                    dtype=sc.dtype.vector_3_double,
                    unit=sc.units.m),
        # TOF is optional, Mantid always has this but not needed at this point
        Dim.Tof:
        sc.Variable(dims=[Dim.Tof], values=np.arange(1000.0), unit=sc.units.us)
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

# Set some positions
d.coords[Dim.Position].values[0] = [1, 2, 3]
print(d.coords[Dim.Position].values[0])

# Add some events
# Note: d.coords[Dim.Tof] gives the "dense" TOF coord, not the event-TOFs
d['sample'].coords[Dim.Tof][Dim.Position, 0].values = np.arange(10)
# The following should be equivalent but does not work yet, see scipp/#290
# d['sample'].coords[Dim.Tof].values[1] = np.arange(10)
print(d)

dspacing = sc.neutron.convert(d, Dim.Tof, Dim.DSpacing)
print(dspacing)

# Converting event data to histogram (not available yet)
# hist = sc.histogram(d)

# "DiffractionFocussing" == sum? (not available yet)
# focussed = sc.sum(hist, Dim.Position)
