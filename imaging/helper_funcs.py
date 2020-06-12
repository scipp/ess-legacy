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
    return np.loadtxt(tof_file, delimiter='\t',
                      skiprows=1,  # Skip header on the first line
                      usecols=1)  # Only use the TOF vals, not index


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


def export_tiff_stack(dataset, key, base_name, output_dir, x_len, y_len,
                      tof_values):
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
    tof_vals = [tof_values[0], tof_values[-1]] if num_bins == 1 else tof_values

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

    rebin_params = sc.Variable(["tof"], unit=sc.units.us,
                               values=np.linspace(
                                   start=rebin_parameters["start"],
                                   stop=rebin_parameters["stop"],
                                   num=rebin_parameters["num_bins"],
                                   dtype=np.float64))

    for i, (slice_bins, shift_parameter) in enumerate(
            zip(frame_parameters, frame_shifts)):
        bins = sc.Variable(["tof"], unit=sc.units.us,
                           values=np.arange(*slice_bins, dtype=np.float64))
        # Rebins the whole data to crop it to frame bins
        rebinned = sc.rebin(data_array, "tof", bins)
        # Shift the frame backwards to make all frames overlap
        rebinned.coords["tof"] += sc.Variable(shift_parameter, unit=sc.units.us)
        # Rebin to overarching coordinates so that the frame coordinates align
        rebinned = sc.rebin(rebinned, "tof", rebin_params)

        frames.append(rebinned)

    for f in frames[1:]:
        frames[0] += f

    return frames[0]


def make_detector_groups(nx_original, ny_original, nx_target, ny_target):
    element_width_x = nx_original // nx_target
    element_width_y = ny_original // ny_target

    x = sc.Variable(dims=['x'],
                    values=np.arange(nx_original)) / element_width_x
    y = sc.Variable(dims=['y'],
                    values=np.arange(ny_original)) / element_width_y
    grid = x + nx_target * y
    return sc.Variable(["spectrum"], values=np.ravel(grid.values))
