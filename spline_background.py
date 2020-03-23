import numpy as np
from scipy import interpolate

import scipp as sc


def bspline_background(variable, dim, smoothing_factor=None):
    """
    Use scipy bspline method to create a fitting function to data in `variable`.
    Knots and splines are evaluated internally.

    Parameters
    ----------
        variable: scipp DataArray
            DataArray container to which values the spline should be applied
        dim: scipp Dim
            The dimension along which the spline should be calculated
        smoothing_factor: float
            Positive smoothing factor used to choose the number of knots

    Returns
    -------
        scipp DataArray
        DataArray container with the spline
    """
    if dim is None:
        raise ValueError("bspline_background: dimension must be specified.")
    if not isinstance(dim, sc.Dim):
        raise ValueError("bspline_background: dimension must be of Dim type.")
    if smoothing_factor < 0:
        raise ValueError("bspline_background: smoothing_factor must be positive.")

    x_values = variable.coords[dim].values
    values = variable.values
    errors = variable.variances
    weights = 1.0/errors

    # flag marking 'bin-edge' status of input x values
    bin_edge_input = False

    # If input x values are bin-edges, use bin centres to spline

    if len(x_values) == len(values)+1:
        bin_edge_input = True

        new_x_values = [0.5*(x_values[i] + x_values[i+1])
                        for i in range(len(x_values)-1)]
        x_values = new_x_values

    # find out the knots and splines.
    knots, u = interpolate.splprep([x_values, values],
                                   s=smoothing_factor,
                                   k=5,
                                   w=weights)
    # perform the B-spline interpolation
    splined = interpolate.splev(u, knots)
    # splined is a [2][data_length] array for X and Y values

    # cast splined into DataArray type
    if bin_edge_input:
        # redefine output x_values to be 'bin-edges'
        len_splined = len(splined[0])
        bin_edge_output_x = np.zeros(len_splined + 1)
        bin_edge_output_x[0] = 0.5*(splined[0][0] - splined[0][1])
        bin_edge_output_x[len_splined] = 0.5*(
                                         3*splined[0][len_splined-1] -
                                         splined[0][len_splined-2])

        for i in range(len_splined-1):
            bin_edge_output_x[i+1] = 0.5*(splined[0][i+1] - splined[0][i])

        output_x = sc.Variable(dims=[dim], values=bin_edge_output_x)
    else:
        output_x = sc.Variable(dims=[dim], values=splined[0])

    output_y = sc.Variable(dims=[dim], values=splined[1])
    output_data = sc.DataArray(data=output_y, coords={dim: output_x})

    return output_data
