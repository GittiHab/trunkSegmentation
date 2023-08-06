import numpy as np
import pytest
from datasets.preprocessing import FilterLabeled


class TestFilterLabeled:
    @staticmethod
    def test_skip_condition_all_zeroes():
        # Test when the input array contains all zeroes
        mask_plane = np.zeros((4, 4))
        filter = FilterLabeled()
        assert filter._skip_condition(mask_plane)

    @staticmethod
    def test_skip_condition_single_stripe():
        # Test when the input array has a single stripe (row/column)
        mask_plane = np.zeros((4, 4))
        mask_plane[1, 0] = 1
        mask_plane[1, 3] = 1
        filter = FilterLabeled()
        assert filter._skip_condition(mask_plane)

    @staticmethod
    def test_skip_condition_multiple_stripes():
        # Test when the input array has multiple stripes
        mask_plane = np.zeros((4, 4))
        mask_plane[1, :] = 1
        mask_plane[3, 0] = 1
        mask_plane[3, 2] = 1
        filter = FilterLabeled()
        assert not filter._skip_condition(mask_plane)

    @staticmethod
    def test_skip_condition_stripes_in_different_directions():
        # Test when the input array has stripes in both rows and columns
        mask_plane = np.zeros((4, 4))
        mask_plane[1, :] = 1
        mask_plane[:, 2] = 1
        filter = FilterLabeled()
        assert not filter._skip_condition(mask_plane)

    @staticmethod
    def test_skip_condition_complex_stripes():
        # Test when the input array has complex stripe patterns
        mask_plane = np.zeros((5, 5))
        mask_plane[1, :] = 1
        mask_plane[3, :] = 1
        mask_plane[:, 2] = 1
        mask_plane[4, 4] = 1
        filter = FilterLabeled()
        assert not filter._skip_condition(mask_plane)
