from typing import Collection
import numpy as np


class MaskedSlices(Collection):
    def __init__(self, *slice_dims: np.ndarray):
        super().__init__()
        self._dims = slice_dims

    def __len__(self):
        return sum([len(d) for d in self._dims])

    def __iter__(self):
        for d in self._dims:
            for e in d:
                yield e

    def __contains__(self, item):
        for d in self._dims:
            if item in d:
                return True
        return False

    def __getitem__(self, idx):
        dim = 0
        while idx >= len(self._dims[dim]):
            idx -= len(self._dims[dim])
            dim += 1
        return self._dims[dim][idx]

    def __repr__(self):
        if len(self._dims) < 1:
            return ''
        representation = repr(self._dims[0])
        if len(self._dims) < 2:
            return representation
        for d in self._dims[1:]:
            representation += ', '
            representation += repr(d)
        return representation


    def components(self):
        return self._dims
