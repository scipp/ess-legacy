import numpy as np
from scipy import interpolate


def bspline_background(variable, x_values, dim=None):
    """
    Use scipy bspline method to create a fitting function to data in `variable`.
    Knots and splines are evaluated internally.

    Parameters
    ----------
        variable: scipp variable
            The variable to which the spline should be applied
        x_values: nupmy array
            X values for the variable data
        dim: scipp Dim
            The dimension along which the spline should be calculated
    """

    if dim is None:
        raise ValueError("bspline_background: dimension must be specified.")

    values = variable[dim, :].values

    # find out the knots.
    # "Magic" s=0 for the smoothness parameter stands for interpolation,
    # since no weights are provided.
    knots, u = interpolate.splprep([x_values, values], s=0)
    # perform the B-spline interpolation
    splined_variable = interpolate.splev(u, knots)
    # splined_variable is a [2][data_length] array for X and Y values

    return splined_variable


if __name__ == '__main__':
    # simple test
    import matplotlib.pyplot as plt
    import scipp as sc
    from scipp import Dim

    plt.rcParams['figure.figsize'] = [10, 8]

    N = 100
    x = np.arange(N)
    y0 = np.zeros(N)
    err = np.sqrt(y0)
    y = y0
    y += err*np.random.normal(size=len(err))
    for i, item in enumerate(y):
        xi = (15.0 / (N-1)) * i
        y[i] = np.cos(xi) * np.exp(-0.1*xi)

    input_y = sc.Variable(dims=[Dim.Tof], values=y, variances=err**2, unit=sc.units.us)
    output = bspline_background(input_y, x, dim=Dim.Tof)

    plt.figure()
    plt.plot(x, y, 'ro', output[0], output[1], 'b')
    plt.legend(['Points', 'Interpolated B-spline', 'True'], loc='best')
    plt.axis([min(x)-1, max(x)+1, min(y)-1, max(y)+1])
    plt.title('B-Spline interpolation')
    plt.show()
