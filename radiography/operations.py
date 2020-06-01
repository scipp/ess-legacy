import glob
import os
import numpy as np
import scipp as sc
from astropy.io import fits


def _load_fits(fit_dir):
    if not os.path.isdir(fit_dir):
        raise RuntimeError(fit_dir + " is not directory")
    stack = []
    path_length = len(fit_dir) + 1
    filenames = sorted(glob.glob(fit_dir + "/*.fits"))
    nfiles = len(filenames)
    count = 0
    print(f"Loading {nfiles} files from '{fit_dir}'")
    for filename in filenames:
        count += 1
        print('\r{0}: Image {1}, of {2}'.format(filename[path_length:], count, nfiles), end="")
        with fits.open(os.path.join(fit_dir, filename)) as hdu:
            data = hdu[0].data
            stack.append(np.flipud(data))

    if len(stack) == 1:
        # Fold away a dim
        stack = stack[0]

    return np.array(stack, dtype=np.float64)


def fits_to_variable(fits_dir, average_data=False):
    """
    Loads all fits images from the directory into a scipp Variable.
    """
    stack = _load_fits(fits_dir)
    if average_data:
        stack = stack.mean(axis=0)

    if len(stack.shape) == 3:
        return sc.Variable(["slice", "x", "y"], values=stack, variances=stack)

    elif len(stack.shape) == 2:
        return sc.Variable(["x", "y"], values=stack, variances=stack)
    else:
        raise IndexError("Expected 2 or 3 dimensions,"
                         f" found {len(stack.shape)}")
