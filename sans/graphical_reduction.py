# Functions used in graphical reduction demo. Lifted from reduction.ipynb
import scipp as sc
import numpy as np
from contrib import to_bin_centers, to_bin_edges, map_to_bins
import ipywidgets as w

def load_and_return(run, path):
    print(f'Loading data for run {run}:')
    return sc.neutron.load(filename=f'{path}/LARMOR000{run}.nxs')

def run_reduction(sample, sample_trans, background,
                  background_trans, moderator_file_path, direct_beam_file_path,
                   l_collimation, r1, r2, dr, wavelength_bins):
    
    dtype = sample.coords['position'].dtype
    sample_pos_offset = sc.Variable(value=[0.0, 0.0, 0.30530], unit=sc.units.m, dtype=dtype)
    bench_pos_offset = sc.Variable(value=[0.0, 0.001, 0.0], unit=sc.units.m, dtype=dtype)
    for item in [sample, sample_trans, background, background_trans]:
        item.coords['sample-position'] += sample_pos_offset
        item.coords['position'] += bench_pos_offset

    print('Reducing sample data:')
    sample_q1d = q1d(data=sample, transmission=sample_trans, 
                     l_collimation=l_collimation, r1=r1, r2=r2, dr=dr, wavelength_bins=wavelength_bins,
                     direct_beam_file_path=direct_beam_file_path, moderator_file_path=moderator_file_path,
                     wavelength_bands=None)

    print('Reducing background data:')
    background_q1d = q1d(data=background, transmission=background_trans,
                         l_collimation=l_collimation, r1=r1, r2=r2, dr=dr, wavelength_bins=wavelength_bins,
                         direct_beam_file_path=direct_beam_file_path, moderator_file_path=moderator_file_path,
                         wavelength_bands=None)
    
    reduced = sample_q1d - background_q1d

    # reduced.coords['Transmission'] = sc.Variable(
    #     value=f'{sample_transmission_run_number}_trans_sample_0.9_13.5_unfitted')
    # reduced.coords['TransmissionCan'] = sc.Variable(
    #     value=f'{background_transmission_run_number}_trans_can_0.9_13.5_unfitted')
    
    print('Finished Reduction')
    return reduced, sample_q1d, background_q1d

def load_larmor(run_number):
    return sc.neutron.load(filename=f'{path}/LARMOR000{run_number}.nxs')

def load_rkh(filename):
    return sc.neutron.load(
           filename=filename,
           mantid_alg='LoadRKH',
           mantid_args={'FirstColumnValue':'Wavelength'})

def apply_masks(data):
    tof = data.coords['tof']
    data.masks['bins'] = sc.less(tof['tof',1:], 1500.0 * sc.units.us) | \
                         (sc.greater(tof['tof',:-1], 17500.0 * sc.units.us) & \
                          sc.less(tof['tof',1:], 19000.0 * sc.units.us))
    pos = sc.neutron.position(data)
    x = sc.geometry.x(pos)
    y = sc.geometry.y(pos)
    data.masks['beam-stop'] = sc.less(sc.sqrt(x*x+y*y), 0.045 * sc.units.m)
    data.masks['tube-ends'] = sc.greater(sc.abs(x), 0.36 * sc.units.m) # roughly all det IDs listed in original
    #MaskDetectorsInShape(Workspace=maskWs, ShapeXML=self.maskingPlaneXML) # irrelevant tiny wedge?


def background_mean(data, dim, begin, end):
    coord = data.coords[dim]
    assert (coord.unit == begin.unit) and (coord.unit == end.unit)
    i = np.searchsorted(coord, begin.value)
    j = np.searchsorted(coord, end.value) + 1
    return data - sc.mean(data[dim, i:j], dim)


def transmission_fraction(incident_beam, transmission, wavelength_bins):
    # Approximation based on equations in CalculateTransmission documentation
    # TODO proper implementation of mantid.CalculateTransmission
    return (transmission / transmission) * (incident_beam / incident_beam)
    #CalculateTransmission(SampleRunWorkspace=transWsTmp,
    #                      DirectRunWorkspace=transWsTmp,
    #                      OutputWorkspace=outWsName,
    #                      IncidentBeamMonitor=1,
    #                      TransmissionMonitor=4, RebinParams='0.9,-0.025,13.5',
    #                      FitMethod='Polynomial',
    #                      PolynomialOrder=3, OutputUnfittedData=True)

def extract_monitor_background(data, begin, end, wavelength_bins):
    background = background_mean(data, 'tof', begin, end)
    del background.coords['sample-position'] # ensure unit conversion treats this a monitor
    background = sc.neutron.convert(background, 'tof', 'wavelength')
    background = sc.rebin(background, 'wavelength', wavelength_bins)
    return background


def setup_transmission(data, wavelength_bins):
    incident_beam = extract_monitor_background(data['spectrum', 0], 40000.0*sc.units.us, 99000.0*sc.units.us, wavelength_bins)
    transmission = extract_monitor_background(data['spectrum', 3], 88000.0*sc.units.us, 98000.0*sc.units.us, wavelength_bins)
    return transmission_fraction(incident_beam, transmission, wavelength_bins)


def solid_angle(data):
    # TODO proper solid angle
    # [0.0117188,0.0075,0.0075] bounding box size
    pixel_size = 0.0075 * sc.units.m 
    pixel_length = 0.0117188 * sc.units.m
    L2 = sc.neutron.l2(data)
    return (pixel_size * pixel_length) / (L2 * L2)


def q_resolution(lam_edges, moderator, d, l_collimation, r1, r2, dr):
    moderator = sc.rebin(moderator, 'wavelength', lam_edges)
    
    d_lam = lam_edges['wavelength', 1:] - lam_edges['wavelength', :-1] # bin widths
    lam = 0.5 * (lam_edges['wavelength', 1:] + lam_edges['wavelength', :-1]) # bin centres
    
    l2 = sc.neutron.l2(d)
    theta = sc.neutron.scattering_angle(d)  
    inv_l3 = (l_collimation + l2) / (l_collimation * l2)
    
    # Terms in Mildner and Carpenter equation.
    # See https://docs.mantidproject.org/nightly/algorithms/TOFSANSResolutionByPixel-v1.html
    a1 = (r1/l_collimation)*(r1/l_collimation) * 3.0
    a2 = (r2*inv_l3)*(r2*inv_l3) * 3.0
    a3 = (dr/l2) * (dr/l2) 
    q_sq = 4.0 * np.pi * sc.sin(theta) * sc.reciprocal(lam) # keep in wav dim to prevent broadcast
    q_sq *= q_sq
    
    tof = moderator.data.copy()
    tof.variances = None # shortcoming of Mantid or Mantid converter?
    tof.rename_dims({'wavelength':'tof'}) # TODO overly restrictive check in convert (rename)
    tof.unit = sc.units.us
    mod = sc.Dataset(coords={
      'tof':tof,
      'position':sample.coords['position'],
      'source_position':sample.coords['source_position'],
      'sample_position':sample.coords['sample_position']})
    s = sc.neutron.convert(mod, 'tof', 'wavelength').coords['wavelength']
    
    std_dev_lam_sq = (d_lam * d_lam)/12 + s * s
    std_dev_lam_sq *= sc.reciprocal(lam * lam)
    f = (4 * np.pi * np.pi) * sc.reciprocal(12 * lam * lam)
   
    return sc.DataArray(f * (a1 + a2 + a3) + (q_sq * std_dev_lam_sq),
                        coords={'wavelength':lam, 'spectrum':d.coords['spectrum'].copy()})


def q1d(data, transmission, l_collimation, r1, r2, dr, wavelength_bins, direct_beam_file_path, moderator_file_path, wavelength_bands=None):
    transmission = setup_transmission(transmission, wavelength_bins)
    data = data.copy()
    apply_masks(data)
    data = sc.neutron.convert(data, 'tof', 'wavelength', out=data)
    data = sc.rebin(data, 'wavelength', wavelength_bins)

    monitor = data.coords['monitor1'].value
    monitor = background_mean(monitor, 'tof', 40000.0*sc.units.us, 99000.0*sc.units.us)
    monitor = sc.neutron.convert(monitor, 'tof', 'wavelength', out=monitor)
    monitor = sc.rebin(monitor, 'wavelength', wavelength_bins)

    # this factor seems to be a fudge factor. Explanation pending.
    data *= 100.0 / 176.71458676442586

    # Setup direct beam and normalise to monitor. I.e. adjust for efficiency of detector across the wavelengths.
    direct_beam = load_rkh(filename=direct_beam_file_path)
    # This would work assuming that there is a least one wavelength point per bin
    #direct_beam = sc.groupby(direct_beam, 'wavelength', bins=monitor.coords['wavelength']).mean('wavelength')
    direct_beam = map_to_bins(direct_beam, 'wavelength', monitor.coords['wavelength'])
    direct_beam = monitor * transmission * direct_beam

    # Estimate qresolution function
    moderator = load_rkh(filename=moderator_file_path)
    to_bin_edges(moderator, 'wavelength')

    q_bins = sc.Variable(
        dims=['Q'],
        unit=sc.units.one/sc.units.angstrom,
        values=np.geomspace(0.008, 0.6, num=55))
   
    d = sc.Dataset({'data':data, 'norm':solid_angle(data)*direct_beam})
    #dq_sq = q_resolution(d.coords['wavelength'], moderator, d, l_collimation, r1, r2, dr)
    to_bin_centers(d, 'wavelength')
    d = sc.neutron.convert(d, 'wavelength', 'Q', out=d) # TODO no gravity yet
    
    
    if wavelength_bands is None:
        d = sc.histogram(d, q_bins)
        d = sc.sum(d, 'spectrum')
        I = d['data']/d['norm']
    else:
        # Cut range into number of requested bands
        n_band = int(wavelength_bands)
        n_bin = len(wavelength_bins.values)-1
        bounds = np.arange(n_bin)[::n_bin//n_band]
        bounds[-1] = n_bin
        slices =  [slice(i, j) for i,j in zip(bounds[:-1],bounds[1:])]
        bands = None
        # Reduce by wavelength slice
        for s in slices:
            band = sc.histogram(d['Q', s].copy(), q_bins) # TODO fix scipp to avoid need for copy
            band = sc.sum(band, 'spectrum')
            bands = sc.concatenate(bands, band, 'wavelength') if bands is not None else band
        # Add coord for wavelength edges of bands
        bands.coords['wavelength'] = sc.Variable(
            dims=['wavelength'],
            unit=sc.units.angstrom,
            values=np.take(wavelength_bins.values, bounds))
        I = bands['data']/bands['norm']

    return I
