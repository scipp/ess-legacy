class LoKI:
    def __init__(self, ntube=32, nstraw=7, npixel=512, dim='spectrum'):
        self._ntube = ntube
        self._nstraw = nstraw
        self._npixel = npixel
        self._dim = dim

    def _assert_valid_tube(self, tube):
        assert tube >= 0 and tube < self._ntube

    def _assert_valid_straw(self, straw):
        assert straw >= 0 and straw < self._nstraw

    def tube(self, tube):
        self._assert_valid_tube(tube)
        n = self._nstraw * self._npixel
        return (self._dim, slice(n * tube, n * (tube + 1)))

    def straw(self, tube, straw):
        self._assert_valid_tube(tube)
        self._assert_valid_straw(straw)
        start = (tube * self._nstraw + straw) * self._npixel
        end = (tube * self._nstraw + straw + 1) * self._npixel
        return (self._dim, slice(start, end))

    def layers(self):
        import scipp as sc
        import numpy as np
        pixel_layers = 0
        straw_layers = 3
        tube_layers = 4
        pixel_layer = sc.Variable(dims=['pixel'],
                                  dtype=sc.dtype.int32,
                                  values=np.zeros(self._npixel))
        straw_layer = sc.Variable(dims=['straw'],
                                  dtype=sc.dtype.int32,
                                  values=[0, 0, 1, 1, 0, 1, 2])
        tube_layer = sc.Variable(dims=['tube'],
                                 dtype=sc.dtype.int32,
                                 values=np.arange(self._ntube) % tube_layers)
        layers = tube_layer * straw_layers + straw_layer + pixel_layer
        return sc.reshape(layers,
                          dims=[self._dim],
                          shape=(layers.values.size, )).copy()

    def to_logical_dims(self, data):
        """
        Reshape data to use ['tube','straw','pixel'] instead of ['spectrum'].
        """
        import scipp as sc
        dims = data.dims
        shape = data.shape
        if dims[0] != 'spectrum':
            raise RuntimeError("Expected 'spectrum' to be outer dim of data")
        dims[0:1] = ['tube', 'straw', 'pixel']
        shape[0:1] = [self._ntube, self._nstraw, self._npixel]
        coords = {
            dim: coord.copy()
            for dim, coord in data.coords.items()
            if not 'spectrum' in coord.dims
        }
        return sc.DataArray(data=sc.reshape(data.data,
                                            dims=dims,
                                            shape=tuple(shape)),
                            coords=coords)
