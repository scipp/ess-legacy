import math
import scipp as sc
from scipp import Dim
import numpy as np
import matplotlib
from scipp.compat.mantid import load

def expand_data_file(filename, n_pixels, n_events=None, time_noise_us=200, verbose=False):
    """
    Function that loads an event based nexus data file using mantid through
    scipp and adds additional pixels / events. The pixels in the datafile are
    replicated until the required number is reached. For each pixel the original
    sparse data is added, but with a user defined random noise. If a number of
    events is specified, the original data is added multiple times with new
    random time noise each time. Exceptions are raised if n_pixels or n_events
    are too small. A scipp dataset is returned.

    Arguments
    ---------

    filename : str
        Filename of nexus event datafile to be loaded

    n_pixels : int
        Number of pixel for returned scipp dataset, must be larger than original

    Keyword Arguments
    -----------------

    n_events : int
        Number of events for returned scipp dataset, must be sufficiently large

    time_noise_us : float
        Width of gaussian noise added to time of flight data

    verbose : Boolean
        Sets verbose mode which prints additional information

    """
    # Make sure n_pixels and n_events are ints or can at least be converted
    n_pixels = int(n_pixels)
    if n_events is not None:
        n_events = int(n_events)

    d = sc.Dataset()
    d["loaded_data"] = load(filename=filename, load_pulse_times=False)
    if verbose:
        print("Loaded dataset")
        print(d)

    original_length = len(np.array(d.coords[Dim.Spectrum].values))
    original_events = 0
    for pixel_id in range(original_length):
        original_events += len(np.array(d["loaded_data"].coords[Dim.Tof].values[pixel_id]))

    if n_pixels < original_length:
        raise ValueError("n_pixel less than number of pixels in existing datafile")

    expected_events = n_pixels/original_length*original_events
    if n_events is not None and n_events < expected_events:
        print(n_pixels, n_events)
        print(original_length, original_events)
        raise ValueError("n_events too small, not all pixels will get events")

    # Adding pixels to dataset
    current_length = original_length
    positions_var = d.coords[Dim.Spectrum]
    combined_var = positions_var.copy()
    while (current_length <= n_pixels - original_length):
        current_length += original_length
        combined_var = sc.concatenate(combined_var, positions_var, Dim.Spectrum)

    remaining_length = n_pixels - current_length

    if remaining_length > 0:
        combined_var = sc.concatenate(combined_var, positions_var[Dim.Spectrum, 0:remaining_length], Dim.Spectrum)

    # Create sparse tof data
    tofs = sc.Variable(dims=[Dim.Spectrum, Dim.Tof], shape=[n_pixels, sc.Dimensions.Sparse], unit=sc.units.us)
    # Create new dataset with new positions and tof sparse data
    ds = sc.DataArray(coords={Dim.Spectrum: combined_var, Dim.Tof: tofs})
    # Keep detector_info from loaded data manually
    ds.labels["detector_info"] = d.labels["detector_info"]

    if verbose:
        print("Generated dataset before tof events are added")
        print(ds)

    added_events = 0
    while n_events is None or added_events < n_events: # keep adding sparse data until enough events reached
        original_id = 0
        for pixel_id in range(n_pixels):
            # For each pixel, grab the events from the original file
            #pos_n_events = len(np.array(d["loaded_data"].coords[Dim.Tof].values[original_id]))
            pos_n_events = len(d["loaded_data"].coords[Dim.Tof].values[original_id])

            # If any events present, add random noise and append them as sparse data
            if pos_n_events > 0:
                noise = np.random.normal(pos_n_events, scale=time_noise_us)
                new_values = np.array(d["loaded_data"].coords[Dim.Tof].values[original_id]) + noise

                ds.coords[Dim.Tof][Dim.Spectrum, pixel_id].values.extend(new_values)
                added_events += pos_n_events

            if n_events is not None and added_events >= n_events:
                break # The required number of events have been reached.

            # keep track of index in original dataset
            original_id += 1
            if original_id == original_length:
                original_id = 0

        # If n_events not given by user, just use natural events for each pixel instead of adding additional
        if n_events is None:
            break

    if verbose:
        final_pixel_length = len(np.array(ds.coords[Dim.Spectrum].values))
        total_events = 0
        for pixel_id in range(n_pixels):
            total_events += len(np.array(ds.coords[Dim.Tof].values[pixel_id]))

        print("Generated dataset after tof events are added")
        print(ds)

        print("Original data file had " + str(original_events) + " events, distributed over " + str(original_length) + " pixels.")
        if n_events is None:
            print("A new datafile with " + str(n_pixels) + " was requested.")
        else:
            print("A new datafile with " + str(n_pixels) + " with " + str(n_events) + " events was requested.")
        print("New data file has " + str(total_events) + " events, distributed over " + str(final_pixel_length) + " pixels.")

        if math.fabs(total_events - expected_events)/total_events > 0.1 and n_events is None:
            print("With linear scaling, it was expected to get " + str(expected_events) + ", yet the result was more than 10% off.")
        if n_events is not None:
            events_change = math.fabs(total_events - n_events)/total_events
            if events_change > 0.03:
                print("The number of events was " + str(events_change*100) + "% off the requested number.")

    return ds
