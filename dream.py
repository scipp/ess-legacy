import time
import numpy as np
import scipp as sc
from scipp import Dim
from scipy import constants as scipy_constants


class bragg_peak:
    """
    Class to hold a Bragg peak generator with a given d_spacing,
    width and strength. The strength is a unphysical description
    of its strength, and has no absolute meaning, but is relative
    to the total strength observed.
    The Bragg peak can generate a numpy array of d_spacing values
    with a length found from the strength of the peak and a given
    conversion factor between strength and expected events.
    The number of events is drawn from a poisson distribution.
    """
    def __init__(self, d_spacing, d_width, strength):
        """
        Initialize a Bragg peak

        Parameters
        ----------
            d_spacing: float [AA]
                Central d_spacing of the peak

            d_width: float [AA]
                Width of the gaussian d_spacing distribution

            strength: float [Arb]
                Strength of the Bragg peak (relative)
        """

        self.d_spacing = d_spacing
        self.d_width = d_width
        self.strength = strength

    def generate_events(self, events_per_strength):
        """
        Will generate a poisson distributed number of events with
        expectation value: strength*events_per_strength.
        The events is distributed around the central d_spacing value
        with the given d_spacing width.

        Parameters
        ----------
            events_per_strength: float [Arb]
                Number of events expected per strength of the Bragg peak
        """

        n_events = np.random.poisson(self.strength*events_per_strength)

        return_value = self.d_spacing + self.d_width*np.random.randn(n_events)

        if len(return_value) >0 and np.min(return_value) < 0:
            print("Negative d_spacing returned")

        return return_value

class DreamTest:
    """
    Class for setting up a simplified version of the DREAM instrument using scipp.
    """
    def __init__(self, n_pixel, source_sample_dist=76.55, detector_radius=1.25, n_rows=1):
        """
        Initialize the simplified DREAM instrument.
        Sets up a scipp dataset including:
            Source and sample position (along z direction)
            Detector pixels in cylinder around sample (symmetry in y direction)
            Dense tof dimension
            Sparse tof dimension

        Parameters
        ----------
            n_pixel: int
                Number of pixels in generated detector array

            source_sample_dist: double [m]
                Distance from source to sample

            detector_radius: double [m]
                Radius of detector (centered on sample)

            n_rows: int
                Number of rows in the detector
        """

        self.cal = sc.Dataset()

        # Includes labels 'component_info', a special name that `scipp.neutron.convert` will inspect for instrument information.
        self.data = sc.Dataset(
            coords={
                Dim.Position:
                self._make_cylinder_coords(n_pixel, n_rows, radius=detector_radius,
                                           source_sample_dist=source_sample_dist, write_calibration=True),
                # TOF is optional, Mantid always has this but not needed at this point
                Dim.Tof:
                sc.Variable(dims=[Dim.Tof], values=np.arange(10.0), unit=sc.units.us)
            },
            labels={
                'component_info':
                sc.Variable(
                    self._make_component_info(source_pos=[0, 0, -source_sample_dist], sample_pos=[0, 0, 0]))
            })

        # Add sparse TOF coord, i.e., the equivalent to event-TOF in Mantid
        tofs = sc.Variable(dims=[Dim.Position, Dim.Tof],
                           shape=[n_pixel, sc.Dimensions.Sparse],
                           unit=sc.units.us)
        self.data['sample'] = sc.DataArray(coords={Dim.Tof: tofs})

        # source_sample_dist needed in data generation
        self.source_sample_dist = source_sample_dist

        # Set up dspacing and hist variable
        self.dspacing = None
        self.hist = None

        # d spacings for sample
        self.sample_bragg_peaks = []
        self.sample_bragg_peaks_d = []

    def _make_component_info(self, source_pos, sample_pos):
        """
        Create a dataset containing basic information about the components in an instrument.

        Currently this contains only source and sample position.
        The structure of this is subject to change.
        """
        component_info = sc.Dataset({
            'position':
            sc.Variable(dims=[Dim.Row],
                        values=[source_pos, sample_pos],
                        unit=sc.units.m)
        })
        return component_info

    def _make_cylinder_coords(self, n_pixel, n_rows=1, height=0.1, radius=4,
                             center=[0,0,0], ttheta_min=15, ttheta_max=160,
                             write_calibration=False, source_sample_dist=None):
        """
        Create a scipp variable with positions of detector pixels on a
        cylindrical detector with symmetry around y axis and detection
        in the zx plane. The positions correspond to the center of the
        pixels.  If the total number of pixels can not be evenly split
        into the given number of rows, a scipp variable with a lower
        number of pixels is returned. The function can write a simple
        calibration Dataset if the keyword argument write_calibration
        is set to True, and in this case the source_sample_dist must
        be set as well.

        Parameters
        ----------
            n_pixel: int
                Total number of pixels on the detector

        Keyword parameters
        ------------------
            n_rows: int
                Number of rows to split the pixels into

            height: float [m]
                Total height between lowest pixel center and highest

            radius: float [m]
                Distance from symmetry axis to each pixel

            center: array of 3 float [m]
                Center position of detector array (usually sample position)

            ttheta_min: float [deg]
                Minimum value of two theta covered by the detector

            ttheta_max: float [deg]
                Maximum value of two theta covered by the detector

            write_calibration: bool
                True if a calibration Dataset should be written

            source_sample_dist: float [m]
                Distance from source to sample to use in calibration data
        """

        if not n_pixel % n_rows == 0:
            print("The number of pixels can not be evenly distributed into the nunmber of rows")

        n_pixel_per_row = int(n_pixel/n_rows)
        if n_rows == 1:
            heights = [center[1]]
        else:
            heights = np.linspace(-0.5*height, 0.5*height, n_rows) + center[1]

        # Create a numpy array that describes detector positions
        pixel_positions = np.zeros((n_pixel_per_row*n_rows, 3))
        ttheta_array = np.linspace(ttheta_min, ttheta_max, n_pixel_per_row)
        if write_calibration:
            if source_sample_dist is None:
                print("source_sample_dist necessary for creating calibration file!")
            difc = np.zeros((n_pixel_per_row*n_rows))
            #constant = unit_conversion*m_n/plancks
            # [s/m] -> [us/AA] 1E6/1E10 = 1E-4
            unit_conversion = 1E-4
            constant = unit_conversion*scipy_constants.m_n/scipy_constants.h

        for row in range(n_rows):
            from_index = row*n_pixel_per_row
            to_index = (row+1)*n_pixel_per_row
            pixel_positions[from_index:to_index,0] = radius*np.sin(np.pi/180*ttheta_array) + center[0]
            pixel_positions[from_index:to_index,1] = heights[row]
            pixel_positions[from_index:to_index,2] = radius*np.cos(np.pi/180*ttheta_array) + center[2]
            if write_calibration:
                source_to_detector = source_sample_dist + np.sqrt(radius**2 + heights[row]**2)
                difc[from_index:to_index] = 2*constant*source_to_detector*np.sin(np.pi/180*0.5*ttheta_array)



        # Use this to initialize the pixel coordinates
        pixel_coords = sc.Variable(dims=[Dim.Position],
                                   values=pixel_positions,
                                   unit=sc.units.m)

        if write_calibration:
            self.cal["tzero"] = sc.Variable(dims=[Dim.Position], values=np.zeros(n_pixel_per_row*n_rows), unit=sc.units.us)
            self.cal["difc"] = sc.Variable(dims=[Dim.Position], values=difc, unit=sc.units.us/sc.units.angstrom)

        return pixel_coords

    def generate_random_sample(self, n_peaks, d_range=[0.5, 10], deltad_over_d_range=[0.1, 0.5], strength_range=[1, 10]):
        """
        Generate a random series of Bragg peaks. They will be
        stored sorted with respect to d spacing to allow the
        data generation algorithm to search it faster.

        Parameters
        ----------
            n_peaks: int
                Number of Bragg peaks to generate

            d_range: List[2] [AA]
                Start and end of allowed d range interval

            deltad_over_d_range: List[2] [%]
                Start and end of allowed delta d over d range (in %, typical less than 1%)

            strength_range: List[2] [arb]
                Start and end of allowed strength interval (relative peak strength)
        """
        d_array = []
        for _ in range(n_peaks):
            d_array.append(np.random.uniform(d_range[0], d_range[1]))

        # Sort the generated d values, ascending
        d_array.sort()
        self.sample_bragg_peaks_d = d_array

        # Generate each bragg peak based on its d spacing, add random width and strength
        for d_value in d_array:
            dover_d_value = np.random.uniform(deltad_over_d_range[0], deltad_over_d_range[1])*d_value/100
            strength_value = np.random.uniform(strength_range[0], strength_range[1])
            self.sample_bragg_peaks.append(bragg_peak(d_value, dover_d_value, strength_value))

    def generate_data_pseudo(self, n_events, wavelength_width=3.6, wavelength_center=3.0,
                             effective_pulse_length=1E-4, progress_bar=True, verbose=True):
        """
        Generate pseudo realistic distribution based on DREAM parameters.
        The wavelength width needs to be a bit less than normal to contain
        the time distribution within the allowed frame due to no simulated
        chopper system and that Bragg peaks with a center within the
        wavelength range is sampled over their entire gaussian.
        The number of events generated will not exactly match n_events, as
        the events are drawn from many poisson distributions with an
        expectation value equal to n_events. For typical n_event numbers,
        the difference is much less than 1%.

        Parameters
        ----------
            n_events: int
                Number of events to generate

            wavelength_width: float [AA]
                Width of wavelength frame of the instrument, 3.6 Å should fill the frame with perfect source

            wavelength_center: float [AA]
                Center of wavelength frame of the instrument, typicall around 2.5-3.0 Å

            effective_pulse_length: float [s]
                Effective pulse length (after choppers have modified the pulse), between 10 us and 2.86 ms

            progress_bar: Boolean
                Set to True for a progress bar of data generation to be written to terminal

            verbose: Boolean
                Set to True for a short analysis of generated data to be written to terminal
        """

        if wavelength_center - wavelength_width/2.0 < 0:
            raise ValueError("Provided wavelength band with negative wavelengths.")

        # get pixel theta:
        positions = np.array(self.data.coords[Dim.Position].values)
        n_positions = (len(positions))
        z_dir = np.zeros((n_positions, 3))
        z_dir[:,2] = 1.0
        position_lengths = np.sqrt(np.sum(positions**2, axis=1))
        position_zdir_dots = np.sum(positions*z_dir, axis=1)
        tthetas = 180/np.pi*np.array(np.arccos(position_zdir_dots/position_lengths))
        thetas = 0.5*tthetas

        # Find allowed bragg peaks for each pixel and find total strength in order to normalize
        allowed = np.full((n_positions, len(self.sample_bragg_peaks)), False)
        pixel_strengths = np.zeros(n_positions)
        count = 0
        for bragg_peak in self.sample_bragg_peaks:
            wavelength = 2.0*bragg_peak.d_spacing*np.sin(np.pi/180*thetas)
            allowed[:,count] = np.logical_and(wavelength < wavelength_center + wavelength_width/2.0,
                                              wavelength > wavelength_center - wavelength_width/2.0)
            pixel_strengths[allowed[:,count]] += bragg_peak.strength
            count += 1

        # Now we have pixel_strength as indicator for relative intensity in each peak (assuming relatively narrow peaks)
        total_strength = np.sum(pixel_strengths)
        events_per_strength = n_events / total_strength

        # Calculate distance from source to each pixel when scattered in sample
        source_to_pixel = self.source_sample_dist + position_lengths

        # Prepare progress bar and timing of most time consuming loop
        progress_fraction = int(n_positions/10)
        progress_count = 0
        start_time = time.time()

        print("Generating sparse data for each pixel")
        # For each pixel we check all allowed bragg peaks and generate a poisson distributed number of events from each
        for pixel_id in range(n_positions):

            if progress_bar and pixel_id % progress_fraction == 0:
                print(str(progress_count*10) + "%")
                progress_count += 1

            allowed_bragg_peaks = allowed[pixel_id,:]
            wavelengths = []
            for bragg_peak_id in np.arange(len(allowed_bragg_peaks))[allowed_bragg_peaks]:
                returned_wavelengths = 2.0*np.sin(np.pi/180*thetas[pixel_id])*self.sample_bragg_peaks[bragg_peak_id].generate_events(events_per_strength)
                if len(returned_wavelengths) > 0:
                    wavelengths.append(returned_wavelengths)

            # Will add the events to the sparse Tof dimension if any are found
            if len(wavelengths) > 0:
                wavelengths = np.concatenate(wavelengths)
                # convert wavelengths to time
                neutron_speeds = 3956.0 / wavelengths # [Å] -> [m/s]
                # convert to travel time
                travel_time = source_to_pixel[pixel_id] / neutron_speeds # [m] / [m/s] -> [s]
                # add some time uncertainty from pulse length (realistically it will be reduced by chopper settings)
                event_times = travel_time + effective_pulse_length*np.random.randn(len(travel_time))
                # Write the events to the pixel with the current pixel_id
                self.data['sample'].coords[Dim.Tof][Dim.Position, pixel_id].values = event_times*1E6 # [s] -> [us] convert to us

        end_time = time.time()
        print("Data was generated in " +  "%.2f" % (end_time - start_time) + " seconds.")

        if verbose:
            sum_times = 0
            min_time = None
            max_time = 0
            for array in np.array(self.data["sample"].coords[Dim.Tof].values):
                sum_times += len(array)
                if len(array) > 0:
                    if min_time is None or np.min(array) < min_time:
                        min_time = np.min(array)
                    if np.max(array) > max_time:
                        max_time = np.max(array)

                    # Should never happen
                    if np.min(array) < 0:
                        print("Negative time found in array!")
                        print(array)

            #print(sum_times)
            n_pixels_read = len(np.array(self.data["sample"].coords[Dim.Tof].values))
            print("Distributed " + str(sum_times) + " events into " + str(n_pixels_read) + " pixels.")
            print("Minimum time recorded: " + "%.0f" % min_time + " us")
            print("Maximum time recorded: " + "%.0f" % max_time + " us")
            frame_time = max_time - min_time
            fraction_of_optimal = frame_time*14/1E6
            print("Corresponding to a frame of: " + "%.0f" % frame_time + " us, which is "
                  + "%.1f" % (fraction_of_optimal*100) + "% of the available frame")
