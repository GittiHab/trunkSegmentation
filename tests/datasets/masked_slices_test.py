import numpy as np
import pytest

from datasets.masked_slices import MaskedSlices


@pytest.fixture
def slice_dims():
    return [np.arange(3), np.arange(4)]

@pytest.fixture
def masked_slices(slice_dims):
    return MaskedSlices(*slice_dims)

class TestMaskedSlices:
    def test_length(self, masked_slices):
        assert len(masked_slices) == 7

    def test_iteration(self, masked_slices):
        assert list(masked_slices) == [0, 1, 2, 0, 1, 2, 3]

    def test_getitem(self, masked_slices):
        assert masked_slices[2] == 2
        assert masked_slices[6] == 3

    def test_contains(self, masked_slices):
        assert 2 in masked_slices
        assert 6 not in masked_slices

    def test_repr(self, masked_slices):
        assert repr(masked_slices) == 'array([0, 1, 2]), array([0, 1, 2, 3])'

    def test_components(self, masked_slices, slice_dims):
        assert masked_slices.components() == tuple(slice_dims)
