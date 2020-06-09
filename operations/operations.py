import numpy as np
import scipp as sc


def mask_from_adj_pixels(mask, bank_width):
    """
    Checks if the adjacent pixels (in 8 directions) are masked to remove
    any noisy pixels which are erroneously masked or unmasked compared to
    it's neighbours

    If all adj. pixels are then the pixel considered is set to True
    If no adj. pixels are then the pixel considered is set to False
    If surrounding pixels have a mix of True/False the val is left as-is
    This function handles border pixels as if they are "wildcard" values

    Parameters
    ----------
    mask: Existing mask with some positions masked
    bank_width: The width of each bank to reshape into 2D

    Returns
    -------
    mask: Mask copy after completing the op. described above

    """

    def shift(var, dim, forward, out_of_bounds):
        fill = var[dim, 0:1].copy()
        fill.values = np.full_like(fill.values, out_of_bounds)
        if forward:
            return sc.concatenate(fill, var[dim, :-1], dim)
        else:
            return sc.concatenate(var[dim, 1:], fill, dim)

    mask = mask.copy()
    mask = sc.reshape(mask, dims=["y", "x"], shape=(bank_width, bank_width))

    def make_flip(fill):
        flip = sc.Variable(dims=['neighbor', 'y', 'x'],
                           shape=[8, ] + mask.shape, dtype=sc.dtype.bool)
        flip['neighbor', 0] = shift(mask, "x", True, fill)
        flip['neighbor', 1] = shift(mask, "x", False, fill)
        flip['neighbor', 2] = shift(mask, "y", True, fill)
        flip['neighbor', 3] = shift(mask, "y", False, fill)
        flip['neighbor', 4:6] = shift(flip['neighbor', 0:2], "y", True, fill)
        flip['neighbor', 6:8] = shift(flip['neighbor', 0:2], "y", False, fill)
        return flip

    # mask if all neighbors masked
    mask = mask | sc.all(make_flip(True), 'neighbor')
    # unmask if no neighbor masked
    mask = mask & sc.any(make_flip(False), 'neighbor')

    # Flatten using numpy, avoids #1192
    return sc.Variable(["spectrum"], values=mask.values.ravel())
