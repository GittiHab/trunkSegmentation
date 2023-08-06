import numpy as np
from datasets.preprocessing import Padding


def test_padding_with_np_array():
    padding = Padding(((0, 0), (8, 8), (8, 8)), 'constant')
    images = np.random.random((5, 16, 16))
    labels = np.random.random((5, 16, 16))
    a_padded, b_padded = padding(images, labels, None, None)
    assert a_padded[0].shape == (32, 32)
    assert b_padded[0].shape == (32, 32)

    # Check the padding values
    assert np.all(a_padded[:, :8, :] == 0)
    assert np.all(a_padded[:, -8:, :] == 0)
    assert np.all(a_padded[:, :, :8] == 0)
    assert np.all(a_padded[:, :, -8:] == 0)
    assert np.all(b_padded[:, :8, :] == 0)
    assert np.all(b_padded[:, -8:, :] == 0)
    assert np.all(b_padded[:, :, :8] == 0)
    assert np.all(b_padded[:, :, -8:] == 0)


def test_padding_with_list_of_images_and_labels():
    padding = Padding(((8, 8), (8, 8)), 'constant')
    images = [np.random.random((16, 16)), np.random.random((64, 64)), np.random.random((128, 128))]
    labels = [np.random.random((16, 16)), np.random.random((64, 64)), np.random.random((128, 128))]
    a_padded, b_padded = padding(images, labels, None, None)
    assert a_padded[0].shape == (32, 32)
    assert b_padded[0].shape == (32, 32)

    # Check the padding values
    assert np.all(a_padded[0][:8, :] == 0)
    assert np.all(a_padded[0][-8:, :] == 0)
    assert np.all(a_padded[0][:, :8] == 0)
    assert np.all(a_padded[0][:, -8:] == 0)
    assert np.all(b_padded[0][:8, :] == 0)
    assert np.all(b_padded[0][-8:, :] == 0)
    assert np.all(b_padded[0][:, :8] == 0)
    assert np.all(b_padded[0][:, -8:] == 0)

def test_padding_scalar_width():
    padding = Padding(8, 'constant')
    images = np.random.random((5, 16, 16))
    labels = np.random.random((5, 16, 16))
    a_padded, b_padded = padding(images, labels, None, None)
    assert a_padded[0].shape == (32, 32)
    assert b_padded[0].shape == (32, 32)