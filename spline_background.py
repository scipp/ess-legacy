import numpy as np
from scipy import interpolate

import scipp as sc


def bspline_background(variable, dim):
    """
    Use scipy bspline method to create a fitting function to data in `variable`.
    Knots and splines are evaluated internally.

    Parameters
    ----------
        variable: scipp DataArray
            DataArray container to which values the spline should be applied
        dim: scipp Dim
            The dimension along which the spline should be calculated
        output: scipp DataArray
            DataArray container with the spline
    """
    if dim is None:
        raise ValueError("bspline_background: dimension must be specified.")
    if not isinstance(dim, sc.Dim):
        raise ValueError("bspline_background: dimension must be of Dim type.")

    x_values = variable.coords[dim].values
    values = variable.values
    errors = variable.variances
    weights = 1.0/errors

    # spline smoothing from
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.UnivariateSpline.html
    s = len(values)*np.std(errors)*np.std(errors)
    # find out the knots and splines.
    knots, u = interpolate.splprep([x_values, values], s=s, k=5, w=weights)
    # perform the B-spline interpolation
    splined = interpolate.splev(u, knots)
    # splined is a [2][data_length] array for X and Y values

    # cast splined into DataArray type
    output_x = sc.Variable(dims=[sc.Dim.X], values=splined[0])
    output_y = sc.Variable(dims=[sc.Dim.X], values=splined[1])
    output_data = sc.DataArray(data=output_y, coords={sc.Dim.X: output_x})

    return output_data


if __name__ == '__main__':
    # simple test
    import matplotlib.pyplot as plt

    plt.rcParams['figure.figsize'] = [10, 8]

    # test 1
    N = 100
    x = np.arange(N)
    y0 = np.zeros(N)
    y = y0
    for i, item in enumerate(y):
        xi = (15.0 / (N-1)) * i
        y[i] = np.cos(xi) * np.exp(-0.1*xi)
    err = np.sqrt(y*y)

    # test 2
    # x = np.arange(30)
    # y0 = 20.+50.*np.exp(-(x-8.)**2./120.)
    # err = np.sqrt(y0)
    # y = 20.+50.*np.exp(-(x-8.)**2./120.)
    # y += err*np.random.normal(size=len(err))
    # err = np.sqrt(y)

    input_x = sc.Variable(dims=[sc.Dim.X], values=x)
    input_y = sc.Variable(dims=[sc.Dim.X], values=y, variances=err**2)
    input_data = sc.DataArray(data=input_y, coords={sc.Dim.X: input_x})

    output_array = bspline_background(input_data, sc.Dim.X)

    x_sp = output_array.coords[sc.Dim.X].values
    y_sp = output_array.values

    plt.figure()
    plt.plot(x, y, 'ro', x_sp, y_sp, 'b')
    plt.legend(['Points', 'Interpolated B-spline', 'True'], loc='best')
    plt.axis([min(x)-1, max(x)+1, min(y)-1, max(y)+1])
    plt.title('B-Spline interpolation')
    plt.show()
