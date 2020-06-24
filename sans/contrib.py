# Generic helpers that may end up as contributions to scipp if cleaned up
import scipp as sc


def midpoints(var, dim):
    return 0.5 * (var[dim, 1:] + var[dim, :-1])


def to_bin_centers(d, dim):
    edges = d.coords[dim].copy()
    del d.coords[dim]
    d.coords[dim] = 0.5 * (edges[dim, 1:] + edges[dim, :-1])


def to_bin_edges(d, dim):
    centers = d.coords[dim].copy()
    del d.coords[dim]
    first = 1.5 * centers[dim, 0] - 0.5 * centers[dim, 1]
    last = 1.5 * centers[dim, -1] - 0.5 * centers[dim, -2]
    bulk = 0.5 * (centers[dim, 1:] + centers[dim, :-1])
    edges = sc.concatenate(first, bulk, dim)
    edges = sc.concatenate(edges, last, dim)
    d.coords[dim] = edges


def map_to_bins(data, dim, edges):
    data = data.copy()
    to_bin_edges(data, dim)
    bin_width = data.coords[dim][dim, 1:] - data.coords[dim][dim, :-1]
    bin_width.unit = sc.units.one
    data *= bin_width
    data = sc.rebin(data, dim, edges)
    bin_width = edges[dim, 1:] - edges[dim, :-1]
    bin_width.unit = sc.units.one
    data /= bin_width
    return data


def select_bins(array, dim, start, end):
    coord = array.coords[dim]
    edges = coord.shape[0]
    # scipp treats bins as closed on left and open on right: [left, right)
    first = sc.sum(sc.less_equal(coord, start), dim).value - 1
    last = edges - sc.sum(sc.greater(coord, end), dim).value
    assert first >= 0
    assert last < edges
    return array[dim, first:last + 1]


def make_slices(var, dim, cutting_points):
    points = var.shape[0]
    slices = []
    for i in range(cutting_points.shape[0] - 1):
        start = cutting_points[dim, i]
        end = cutting_points[dim, i + 1]
        # scipp treats ranges as closed on left and open on right: [left, right)
        first = sc.sum(sc.less(var, start), dim).value
        last = points - sc.sum(sc.greater_equal(var, end), dim).value
        assert first >= 0
        assert last <= points
        slices.append(slice(first, last))
    return slices
