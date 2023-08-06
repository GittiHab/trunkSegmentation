from utils.inference_utils import output_name


def test_output_name_with_valid_input():
    assert output_name('path/to/myfile.png') == 'path/to/myfile.out.png'


def test_output_name_with_empty_input():
    assert output_name('') == 'out.tiff'
