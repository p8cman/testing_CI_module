from main import extract_nifti_data, threshold_data, get_mean
import nibabel as nib
import numpy as np
import os

def test_extract_nifti_data(tmpdir):
	data = np.ones((32, 32, 15, 100), dtype=np.int16)
	img = nib.Nifti1Image(data, np.eye(4))
	path = os.path.join(tmpdir, "test_img.nii.gz")
	nib.save(img, path)
	loaded_data = extract_nifti_data(path)
	assert np.array_equal(data, loaded_data), "loading is incorrect"

def test_threshold_data():
	data = np.random.randn(4,4)
	threshold = 0.1
	thresholded_data = threshold_data(data, threshold)
	assert (thresholded_data > threshold).any(), "thresholding is incorrect"

def test_get_mean():
	data = np.ones((4,4))
	mean = get_mean(data)
	assert mean == 1, "mean is incorrect"