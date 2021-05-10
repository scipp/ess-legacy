import os
import sys
import getopt
import expand_load

def main(argv):
    """
    This python function loads an event based nexus file with scipp and mantid
    which is then expanded to a higher pixel and event count. Random noise is
    added to the time of flight information to avoid exact duplication of
    events.  Command line arguments are used to set the filename, number of
    pixels, number of events and the width of the random distribution used for
    the time noise.

    -f or --file= : sets the Filename
    -p or --pixels= : sets the number of pixels (scientific notation allowed)
    -e or --events= : sets the number of events (scientific notation allowed)
    -t or --noise= : sets the noise width in us (scientific notation allowed)
    -v : sets verbose mode
    -h : shows help in terminal

    Was tested with mantid 4.0.0 on python 3.6 and scipp
    The following command was used to create the necessary environment:
    conda create -n scipp_mantid_env -c scipp/label/dev -c mantid scipp mantid-framework python=3.6
    """

    filename = None # set with -f or --file=
    n_pixels = None # set with -p or --pixels=
    n_events = None # set with -e or --events=
    time_noise_us = None # set with -t or --noise
    verbose = False # set with -v

    try:
        opts, args = getopt.getopt(argv, "hf:p:e:t:v",["file=", "pixels=", "events=", "noise="])
    except getopt.GetoptError:
        print("Error in input, use -h for more information.")
        print("test_expand_load.py -f <filename> -p <n pixels> -e <n events> -t <time noise us>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == "-h":
            print("-"*5 + " HELP " + "-"*66)
            print("test_expand_load.py -f <filename> -p <n pixels> -e <n events> -t <time noise>")
            print("Long options: --file= --pixels= --events= --noise=")
            print("verbose mode: -v")
            print("filename and n pixels are required.")
            print("Can only increase number of pixels and number of events per pixel.")
            sys.exit()
        elif opt in ("-f", "--file"):
            filename = arg
        elif opt in ("-p", "--pixels"):
            n_pixels = arg
        elif opt in ("-e", "--events"):
            n_events = arg
        elif opt in ("-t", "--noise"):
            time_noise_us = arg
        elif opt == "-v":
            verbose = True

    if filename is None:
        print("test_expand_load.py -f <filename> -p <n pixels> -e <n events> -t <time noise>")
        raise ValueError("No filename is choosen, set -f or --file option.")

    if not os.path.isfile(filename):
        raise ValueError("No file with path: \"" + filename + "\" found.")

    if n_pixels is None:
        print("test_expand_load.py -f <filename> -p <n pixels> -e <n events> -t <time noise>")
        raise ValueError("No number of pixels choosen, set -p or --pixels option.")

    # By converting to float, scientific notation is allowed in input
    n_pixels = float(n_pixels)
    if time_noise_us is not None:
        time_noise_us = float(time_noise_us)
    if n_events is not None:
        n_events = float(n_events)

    if time_noise_us is None: # If time noise not given, use default of function
        data = expand_load.expand_data_file(filename, n_pixels, n_events=n_events, verbose=verbose)
    else:
        data = expand_load.expand_data_file(filename, n_pixels, n_events=n_events,
                                            time_noise_us=time_noise_us, verbose=verbose)

if __name__ == "__main__":
    main(sys.argv[1:]) # drop script name from arguments
