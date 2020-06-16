# SANS specific functions
import numpy as np
import scipp as sc
import contrib


def project_xy(data, nx=100, ny=20):
    z = sc.geometry.z(sc.neutron.position(data))
    x = sc.geometry.x(sc.neutron.position(data)) / z
    y = sc.geometry.y(sc.neutron.position(data)) / z
    data.coords['x/z'] = x
    data.coords['y/z'] = y
    x = sc.Variable(dims=['x/z'],
                    values=np.linspace(sc.min(x).value,
                                       sc.max(x).value,
                                       num=nx))
    y = sc.Variable(dims=['y/z'],
                    values=np.linspace(sc.min(y).value,
                                       sc.max(y).value,
                                       num=ny))
    return sc.realign(data, coords={'y/z': y, 'x/z': x})


def solid_angle(data):
    # TODO proper solid angle
    # [0.0117188,0.0075,0.0075] bounding box size
    pixel_size = 0.0075 * sc.units.m
    pixel_length = 0.0117188 * sc.units.m
    L2 = sc.neutron.l2(data)
    return (pixel_size * pixel_length) / (L2 * L2)


def background_mean(data, dim, begin, end):
    coord = data.coords[dim]
    assert (coord.unit == begin.unit) and (coord.unit == end.unit)
    i = np.searchsorted(coord, begin.value)
    j = np.searchsorted(coord, end.value) + 1
    return data - sc.mean(data[dim, i:j], dim)


def transmission_fraction(incident_beam, transmission):
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
    del background.coords[
        'sample-position']  # ensure unit conversion treats this a monitor
    background = sc.neutron.convert(background, 'tof', 'wavelength')
    background = sc.rebin(background, 'wavelength', wavelength_bins)
    return background


def setup_transmission(data, wavelength_bins):
    incident_beam = extract_monitor_background(data['spectrum',
                                                    0], 40000.0 * sc.units.us,
                                               99000.0 * sc.units.us,
                                               wavelength_bins)
    transmission = extract_monitor_background(data['spectrum',
                                                   3], 88000.0 * sc.units.us,
                                              98000.0 * sc.units.us,
                                              wavelength_bins)
    return transmission_fraction(incident_beam, transmission)


def to_wavelength(data, transmission, direct_beam, masks, wavelength_bins):
    transmission = setup_transmission(transmission, wavelength_bins)
    data = data.copy()
    for name, mask in masks.items():
        data.masks[name] = mask
    data = sc.neutron.convert(data, 'tof', 'wavelength', out=data)
    data = sc.rebin(data, 'wavelength', wavelength_bins)

    monitor = data.attrs['monitor1'].value
    monitor = background_mean(monitor, 'tof', 40000.0 * sc.units.us,
                              99000.0 * sc.units.us)
    monitor = sc.neutron.convert(monitor, 'tof', 'wavelength', out=monitor)
    monitor = sc.rebin(monitor, 'wavelength', wavelength_bins)

    direct_beam = contrib.map_to_bins(direct_beam, 'wavelength',
                                      monitor.coords['wavelength'])
    direct_beam = monitor * transmission * direct_beam

    d = sc.Dataset({'data': data, 'norm': solid_angle(data) * direct_beam})
    contrib.to_bin_centers(d, 'wavelength')
    return d


def reduce(data, q_bins):
    data = sc.histogram(data, q_bins)
    if 'layer' in data.coords:
        return sc.groupby(data, 'layer').sum('spectrum')
    else:
        return sc.sum(data, 'spectrum')


def reduce_by_wavelength(data, q_bins, groupby, wavelength_bands):
    slices = contrib.make_slices(
        contrib.midpoints(data.coords['wavelength'], 'wavelength'),
        'wavelength', wavelength_bands)
    data = sc.neutron.convert(data, 'wavelength', 'Q',
                              out=data)  # TODO no gravity yet
    bands = None
    for s in slices:
        band = sc.histogram(data['Q', s], q_bins)
        band = sc.groupby(band, group=groupby).sum('spectrum')
        bands = sc.concatenate(bands, band,
                               'wavelength') if bands is not None else band
    bands.coords['wavelength'] = wavelength_bands
    return bands
