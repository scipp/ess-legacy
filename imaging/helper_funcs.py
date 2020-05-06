import csv
import glob
import os

import fabio
import numpy as np
import scipp as sc
from tifffile import imsave


def read_x_values(tof_file):
    """
    Reads the TOF values from the CSV into a list
    """
    tof_values = []
    with open(tof_file) as fh:
        csv_reader = csv.reader(fh, delimiter='\t')
        next(csv_reader, None)  # skip header
        for row in csv_reader:
            tof_values.append(float(row[1]))
    return tof_values


def _load_tiffs(tiff_dir):
    if not os.path.isdir(tiff_dir):
        raise RuntimeError(tiff_dir + " is not directory")
    stack = []
    path_length = len(tiff_dir) + 1
    filenames = sorted(glob.glob(tiff_dir + "/*.tiff"))
    nfiles = len(filenames)
    count = 0
    print(f"Loading {nfiles} files from '{tiff_dir}'")
    for filename in filenames:
        count += 1
        print('\r{0}: Image {1}, of {2}'.format(filename[path_length:], count,
                                                nfiles), end="")
        img = fabio.open(os.path.join(tiff_dir, filename))
        stack.append(np.flipud(img.data))

    print()  # Print a newline to separate each load message

    return np.array(stack)


def export_tiff_stack(dataset, key, base_name, output_dir, x_len, y_len):
    to_save = dataset[key]

    num_bins = 1 if len(to_save.shape) == 1 else to_save.shape[0]
    stack_data = np.reshape(to_save.values, (x_len, y_len, num_bins))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Writing tiffs
    for i in range(stack_data.shape[2]):
        imsave(os.path.join(output_dir,
                            '{:s}_{:04d}.tiff'.format(base_name, i)),
               stack_data[:, :, i].astype(np.float32))
    print('Saved {:s}_{:04d}.tiff stack.'.format(base_name, 0))

    # Write out tofs as CSV

    if num_bins == 1:
        dataset_tof = dataset.coords["tof"].values
        tof_vals = [dataset_tof[0], dataset_tof[-1]]
    else:
        tof_vals = to_save.coords["tof"].values

    with open(os.path.join(output_dir, 'tof_of_tiff_{}.txt'.format(base_name)),
              'w') as tofs:
        writer = csv.writer(tofs, delimiter='\t')
        writer.writerow(['tiff_bin_nr', 'tof'])
        tofs = tof_vals
        tof_data = list(zip(list(range(len(tofs))), tofs))
        writer.writerows(tof_data)
    print('Saved tof_of_tiff_{}.txt.'.format(base_name))


def tiffs_to_variable(tiff_dir):
    """
    Loads all tiff images from the directory into a scipp Variable.
    """
    stack = _load_tiffs(tiff_dir)
    data = stack.astype(np.float64).reshape(stack.shape[0],
                                            stack.shape[1] * stack.shape[2])
    return sc.Variable(["tof", "spectrum"],
                       values=data, variances=data)


def stitch(data_array, frame_parameters, frame_shifts, rebin_parameters):
    """
    Stitches the 5 different frames data.

    It crops out each frame, then shifts it so that all frames align,
    and then rebins to the operations bins used for all frames.
    """
    frames = []

    rebin_params = sc.Variable(["tof"], values=np.arange(*rebin_parameters,
                                                           dtype=np.float64))

    for i, (slice_bins, shift_parameter) in enumerate(
            zip(frame_parameters, frame_shifts)):
        bins = sc.Variable(["tof"],
                           values=np.arange(*slice_bins, dtype=np.float64))
        # Rebins the whole data to crop it to frame bins
        rebinned = sc.rebin(data_array, "tof", bins)
        # Shift the frame backwards to make all frames overlap
        rebinned.coords["tof"] += shift_parameter
        # Rebin to overarching coordinates so that the frame coordinates align
        rebinned = sc.rebin(rebinned, "tof", rebin_params)

        frames.append(rebinned)

    for f in frames[1:]:
        frames[0] += f

    return frames[0]


def make_detector_groups(nx_original, ny_original, nx_target, ny_target):
    element_width_x = nx_original // nx_target
    element_width_y = ny_original // ny_target

    # To contain our new spectra mappings
    grid = np.zeros((nx_original, ny_original), dtype=np.float64)

    for i in range(0, nx_target):
        x_start = i * element_width_x
        x_end = (i + 1) * element_width_x

        for j in range(0, ny_target):
            y_start = j * element_width_y
            y_end = (j + 1) * element_width_y

            vals = np.full((element_width_x, element_width_y),
                           i + j * nx_target, dtype=np.float64)
            grid[x_start:x_end, y_start:y_end] = vals

    return sc.Variable(["spectrum"], values=grid.ravel())


def apply_median_filter(dataset, variable_key, bank_width):
    pass
