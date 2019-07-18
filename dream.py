import numpy as np
import scipp as sc
from scipp import Dim

n_pixel = 1000

# Create dataset with a position coordinate.
d = sc.Dataset(coords={
    Dim.Position: sc.Variable(dims=[Dim.Position], shape=[n_pixel], dtype=sc.dtype.vector_3_double, unit=sc.units.m),
    Dim.Tof: sc.Variable(dims=[Dim.Tof], values=np.arange(1000.0), unit=sc.units.us) # TOF is optional, Mantid always has this but not needed at this point
    })

# Add sparse TOF coord, i.e., the equivalent to event-TOF in Mantid
d.set_sparse_coord('sample', sc.Variable(dims=[Dim.Position, Dim.Tof], shape=[n_pixel, sc.Dimensions.Sparse], unit=sc.units.us))

# Set some positions
d.coords[Dim.Position].values[0] = [1,2,3]
print(d.coords[Dim.Position].values[0])

# Add some events
# Note: d.coords[Dim.Tof] gives the "dense" TOF coord, not the event-TOFs
d['sample'].coords[Dim.Tof][Dim.Position, 0].values = np.arange(10)
# The following should be equivalent but does not work yet, see scipp/#290
# d['sample'].coords[Dim.Tof].values[1] = np.arange(10)
print(d)

# Convert units (will work soon, see scipp/#267)
# d.convert(Dim.Tof, Dim.DSpacing)

# Converting event data to histogram (not available yet)
# hist = sc.histogram(d)

# "DiffractionFocussing" == sum? (not available yet)
# focussed = sc.sum(hist, Dim.Position)
