from functools import partial
from multiprocessing import Pool
from typing import Dict, List

import scipp as sc


def integrate(dataset: sc.Dataset):
    integrated = sc.Dataset()
    for dim in dataset.coords.keys():
        integrated.coords[dim] = dataset.coords[dim]

    for k in dataset.keys():
        integrated[k] = sc.sum(dataset[k], "tof")

    return integrated



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


def generate_neighbouring_spectra(num_spectra: int,
                                  bank_width: int) -> Dict[int, List[int]]:
    # Generates neighbouring spectra, for example selecting
    # 5 in the following bank will return a dict with [1,2,3,4,6,7,8,9] where
    # 1,  2,  3
    # 3, [5], 6
    # 7,  8,  9

    # Using multiprocessing as this can be slow for 10k spectra, but is
    # embarrassingly parallel
    with Pool() as p:
        results = p.map(partial(_calc_adj_spectra,
                                bank_width=bank_width,
                                num_spectra=num_spectra),
                        range(num_spectra))
    assert len(results) == num_spectra

    packed_pixels = {}
    for i, adj_pixels in enumerate(results):
        packed_pixels[i] = adj_pixels

    return packed_pixels
