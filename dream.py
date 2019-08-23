import numpy as np
import scipp as sc
from scipp import Dim

class DreamTest:
    """
    Class for setting up a simplified version of the DREAM instrument using scipp.
    """
    def __init__(self, n_pixel, source_sample_dist=76.55, detector_radius=4):
        """
        Initialize the simplified DREAM instrument.
        Sets up a scipp dataset including:
            Source and sample position (along z direction)
            Detector pixels in cylinder arround sample (symmetry in y direction)
            Dense tof dimension
            Sparse tof dimension
            
        Parameters
        ----------
            n_pixel: int
                Number of pixels in generated detector array

            source_sample_dist: double
                Distance from source to sample

            detector_radius: double
                Radius of detector (centered on sample)
        """
    
        # Includes labels 'component_info', a special name that `scipp.neutron.convert` will inspect for instrument information.
        self.d = sc.Dataset(
            coords={
                Dim.Position:
                self._make_cylinder_coords(n_pixel, 1, radius=detector_radius),
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
        self.d['sample'] = sc.DataArray(coords={Dim.Tof: tofs})
    
        # Set up dspacing and hist variable
        self.dspacing = None
        self.hist = None

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
                             center=[0,0,0], ttheta_min=15, ttheta_max=160):
        """
        Create a scipp variable with positions of detector pixels on a
        cylindrical detector with symmetry around y axis and detection
        in the zx plane. The positions correspond to the center of the
        pixels.  If the total number of pixels can not be evenly split
        into the given number of rows, a scipp variable with a lower
        number of pixels is returned.
        
        Parameters
        ----------
            n_pixel: int
                Total number of pixels on the detector
        
        Keyword parameters
        ------------------
            n_rows: int
                Number of rows to split the pixels into
                
            height: double
                Total height between lowest pixel center and highest
                
            radius: double
                Distance from symmetry axis to each pixel
                
            center: array of 3 double
                Center position of detector array (usually sample position)
                
            ttheta_min: double
                Minimum value of two theta covered by the detector
            
            ttheta_max: double
                Maximum value of two theta covered by the detector
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

        for row in range(n_rows):
            from_index = row*n_pixel_per_row
            to_index = (row+1)*n_pixel_per_row
            pixel_positions[from_index:to_index,0] = radius*np.sin(np.pi/180*ttheta_array) + center[0]
            pixel_positions[from_index:to_index,1] = heights[row]
            pixel_positions[from_index:to_index,2] = radius*np.cos(np.pi/180*ttheta_array) + center[2]

        # Use this to initialize the pixel coordinates
        pixel_coords = sc.Variable(dims=[Dim.Position],
                                   values=pixel_positions,
                                   unit=sc.units.m)

        return pixel_coords

    def generate_data(self, n_events):
        """
        This method generates n_events of sparse tof data.
        To be expanded to generate plausible data into some Bragg peaks in the future
        
        Parameters
        ----------
            n_events: int
                Number of events to generate and add to current dataset self.d
        """

        # Add some events
        # Note: d.coords[Dim.Tof] gives the "dense" TOF coord, not the event-TOFs
        self.d['sample'].coords[Dim.Tof][Dim.Position, 0].values = np.arange(n_events)
        # The following should be equivalent but does not work yet, see scipp/#290
         # d['sample'].coords[Dim.Tof].values[1] = np.arange(10)

    def convert_to_dspacing(self):
        """
        This methods converts d to d_spacing using scipp-neutron convert
        """

        self.dspacing = sc.neutron.convert(self.d, Dim.Tof, Dim.DSpacing)
        return self.dspacing

    def convert_to_histogram(self):
        """
        Convert to histogram (not used yet)
        """
        self.hist = sc.histogram(self.dspacing, dspacing.coords[Dim.DSpacing])
        return self.hist

    #def data_reduction(self):
        # "DiffractionFocussing" == sum? (not available yet)
        # focussed = sc.sum(hist, Dim.Position)


n_pixel = 1000
n_events = 10

dream = DreamTest(n_pixel)
print("Dataset with pixels initialzied")
print(dream.d)

dream.generate_data(n_events)
print("Dataset with some sparse tof data")
print(dream.d)

dspacing = dream.convert_to_dspacing()
print("Dataset converted to dspacing")
print(dspacing)
