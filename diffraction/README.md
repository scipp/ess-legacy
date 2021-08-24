# ess-legacy diffraction

This folder contains Python scripts and Jupyter notebooks to reduce diffraction 
data with `scipp` and `scippneutron`.

The main script is `powder_reduction.py` which uses event data from WISH at ISIS STFC.

The required files i.e. sample, vanadium, empty instrument and calibration files are not provided.

Note that it does not reproduce the official data reduction for this instrument. But it 
simply illustrates how to use `scipp` and `scippneutron` to perform some of the steps required
to reduce data from event NeXus files collected at a spallation source.

The Jupyter notebooks in the `demo_notebooks` folder are used to demonstrate the implementation 
of intermediate steps of the reduction and use datafiles available in `Mantid`.

The legacy folder contains older scripts, which will most likely not run with the current 
version of `scipp` and `scippneutron`.
