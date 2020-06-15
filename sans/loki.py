class LoKI:
    def __init__(self, ntube=32, nstraw=7, npixel=512):
        self._ntube = ntube
        self._nstraw = nstraw
        self._npixel = npixel

    def _assert_valid_tube(self, tube):
        assert tube >= 0 and tube < self._ntube

    def _assert_valid_straw(self, straw):
        assert straw >= 0 and straw < self._nstraw

    def tube(self, tube):
        self._assert_valid_tube(tube)
        n = self._nstraw * self._npixel
        return ('spectrum', slice(n * tube, n * (tube + 1)))

    def straw(self, tube, straw):
        self._assert_valid_tube(tube)
        self._assert_valid_straw(straw)
        start = (tube * self._nstraw + straw) * self._npixel
        end = (tube * self._nstraw + straw + 1) * self._npixel
        return ('spectrum', slice(start, end))
