import expand_load

"""
This file contains examples for loading a WISH event based nexus file with scipp
and expanding it to higher pixel and event counts.

Was tested with mantid 4.0.0 on python 3.6 and scipp
The following command was used to create the necessary environment:
conda create -n scipp_mantid_env -c scipp/label/dev -c mantid scipp mantid-framework python=3.6
"""

data = expand_load.expand_data_file('../../wish_event_data/WISH00043525.nxs', 3E5, n_events=6E8, verbose=True)

# tested cases
#data = expand_load.expand_data_file('../../wish_event_data/WISH00043525.nxs', 6E5)
#data = expand_load.expand_data_file('../../wish_event_data/WISH00043525.nxs', 3E5, n_events=6E8, verbose=True)
#data = expand_load.expand_data_file('../../wish_event_data/WISH00043525.nxs', 3E5, n_events=6E8, time_noise_us=500, verbose=True)
#data = expand_load.expand_data_file('../../wish_event_data/WISH00043525.nxs', 3E5, n_events=6E8, time_noise_us=100, verbose=True)
