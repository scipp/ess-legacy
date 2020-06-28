from functools import partial

import numpy as np
import scipp as sc


def _shift(var, dim, forward, out_of_bounds):
    fill = var[dim, 0:1].copy()
    fill.values = np.full_like(fill.values, out_of_bounds)
    if forward:
        return sc.concatenate(fill, var[dim, :-1], dim)
    else:
        return sc.concatenate(var[dim, 1:], fill, dim)


def mask_from_adj_pixels(mask):
    """
    Checks if the adjacent pixels (in 8 directions) are masked to remove
    any noisy pixels which are erroneously masked or unmasked compared to
    it's neighbours

    If all adj. pixels are then the pixel considered is set to True
    If no adj. pixels are then the pixel considered is set to False
    If surrounding pixels have a mix of True/False the val is left as-is

    This function handles border pixels as if they aren't there. So that
    the following happens:
    ------------------------
    |F|T|     ->      |T|T|
    |T|T|             |T|T|
    -----------------------

    Parameters
    ----------
    mask: Existing mask with some positions masked

    Returns
    -------
    mask: Mask copy after completing the op. described above

    """

    mask = mask.copy()

    def make_flip(fill):
        flip = sc.Variable(dims=['neighbor', 'y', 'x'],
                           shape=[
                               8,
                           ] + mask.shape,
                           dtype=sc.dtype.bool)
        flip['neighbor', 0] = _shift(mask, "x", True, fill)
        flip['neighbor', 1] = _shift(mask, "x", False, fill)
        flip['neighbor', 2] = _shift(mask, "y", True, fill)
        flip['neighbor', 3] = _shift(mask, "y", False, fill)
        flip['neighbor', 4:6] = _shift(flip['neighbor', 0:2], "y", True, fill)
        flip['neighbor', 6:8] = _shift(flip['neighbor', 0:2], "y", False, fill)
        return flip

    # mask if all neighbors masked
    mask = mask | sc.all(make_flip(True), 'neighbor')
    # unmask if no neighbor masked
    mask = mask & sc.any(make_flip(False), 'neighbor')
    return mask


def mean_from_adj_pixels(data):
    """
    Applies a mean across 8 neighboring pixels (plus centre value)
    for data with 'x' and 'y' dimensions (at least).
    Result will calculate mean from slices across additional dimensions.
    
    For example if there is a tof dimension in addition to x, and y, 
    for each set of neighbours the returned mean will take the mean tof value in the neighbour group.
    """
    fill = np.finfo(data.values.dtype).min
    has_variances = not data.variances is None
    container = sc.Variable(['neighbor'] + data.dims,
                            shape=[
                                9,
                            ] + data.shape,
                            variances=has_variances)
    container['neighbor', 0] = data
    container['neighbor', 1] = _shift(data, "x", True, fill)
    container['neighbor', 2] = _shift(data, "x", False, fill)
    container['neighbor', 3] = _shift(data, "y", True, fill)
    container['neighbor', 4] = _shift(data, "y", False, fill)
    container['neighbor', 5:7] = _shift(container['neighbor', 1:3], "y", True,
                                        fill)
    container['neighbor', 7:9] = _shift(container['neighbor', 1:3], "y", False,
                                        fill)

    edges_mask = sc.less_equal(container, sc.Variable(value=fill))
    da = sc.DataArray(data=container, masks={'edges': edges_mask})
    return sc.mean(da, dim='neighbor').data


def _median(neighbors, edges_mask, dim):
    masked_values = np.ma.array(neighbors.values,
                                mask=edges_mask.values,
                                copy=False)
    masked_median_v = np.ma.median(masked_values, axis=0)
    if neighbors.variances is not None:
        masked_median_var = np.ma.median(neighbors.variances, axis=0)
        np.ma.array(neighbors.variances, mask=edges_mask.values, copy=False)
        return sc.Variable(dims=neighbors.dims[1:],
                           values=masked_median_v,
                           variances=masked_median_var)
    return sc.Variable(dims=neighbors.dims[1:], values=masked_median_v)


def median_from_adj_pixels(data):
    """
    Applies a median across 8 neighboring pixels (plus centre value)
    for data with 'x' and 'y' dimensions (at least).
    Result will calculate median from slices across additional dimensions.

    For example if there is a tof dimension in addition to x, and y,
    for each set of neighbours the returned median will take the median tof value in the neighbour group.
    """
    fill = np.finfo(data.values.dtype).min
    has_variances = not data.variances is None
    container = sc.Variable(['neighbor'] + data.dims,
                            shape=[
                                9,
                            ] + data.shape,
                            variances=has_variances)
    container['neighbor', 0] = data
    container['neighbor', 1] = _shift(data, "x", True, fill)
    container['neighbor', 2] = _shift(data, "x", False, fill)
    container['neighbor', 3] = _shift(data, "y", True, fill)
    container['neighbor', 4] = _shift(data, "y", False, fill)
    container['neighbor', 5:7] = _shift(container['neighbor', 1:3], "y", True,
                                        fill)
    container['neighbor', 7:9] = _shift(container['neighbor', 1:3], "y", False,
                                        fill)

    edges_mask = sc.less_equal(container, sc.Variable(value=fill,
                                                      variance=fill))
    return _median(container, edges_mask, dim='neighbor')


def _calc_adj_spectra(center_spec_num, bank_width, num_spectra):
    col_positions = [-1, 0, 1]  # Take relative positions of col

    # We calculate along a row at a time, rather than a col at a time
    top_row_center = (center_spec_num - bank_width)
    bottom_row_center = (center_spec_num + bank_width)

    neighbour_spec = []

    # Top / Middle / Bottom row
    neighbour_spec.extend(i + top_row_center for i in col_positions)
    neighbour_spec.extend(i + center_spec_num for i in col_positions)
    neighbour_spec.extend(i + bottom_row_center for i in col_positions)

    def is_valid(spec_num):
        # Do not include OOB and the centre spectra
        return 0 <= spec_num < num_spectra and spec_num != center_spec_num

    neighbour_spec = [i for i in neighbour_spec if is_valid(i)]
    return neighbour_spec


def generate_neighbouring_spectra(num_spectra: int, bank_width: int):
    # Generates neighbouring spectra, for example selecting
    # 5 in the following bank will return a dict with [1,2,3,4,6,7,8,9] where
    # 1,  2,  3
    # 3, [5], 6
    # 7,  8,  9

    # TODO drop this once Median filter uses an implementation that does not
    # require shifts
    from multiprocessing import Pool
    with Pool() as p:
        results = p.map(
            partial(_calc_adj_spectra,
                    bank_width=bank_width,
                    num_spectra=num_spectra), range(num_spectra))
    assert len(results) == num_spectra

    packed_pixels = {}
    for i, adj_pixels in enumerate(results):
        packed_pixels[i] = adj_pixels

    return packed_pixels
