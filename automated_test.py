import pytest
import numpy as np
import fastmorph

def test_spherical_dilate():
	labels = np.zeros((10,10,10), dtype=bool)
	res = fastmorph.spherical_dilate(labels, radius=1000)
	assert np.all(res == False)

	labels[5,5,5] = True
	radius = 5 * np.sqrt(3)
	res = fastmorph.spherical_dilate(labels, radius=radius)
	assert np.all(res == True)	

	res = fastmorph.spherical_dilate(labels, radius=1)
	assert np.count_nonzero(res) == 7

	res = fastmorph.spherical_dilate(labels, radius=np.sqrt(2))
	assert np.count_nonzero(res) == 19

	res = fastmorph.spherical_dilate(labels, radius=np.sqrt(3))
	assert np.count_nonzero(res) == 27

def test_spherical_erode():
	labels = np.ones((10,10,10), dtype=bool)
	res = fastmorph.spherical_erode(labels, radius=1000)
	assert np.all(res == False)

def test_fill_holes():
	labels = np.ones((10,10,10), dtype=np.uint8)
	labels[:,:,:5] = 2

	labels[5,5,2] = 0
	labels[5,5,7] = 0

	assert np.count_nonzero(labels) == 998
	res, ct = fastmorph.fill_holes(labels, return_fill_count=True)
	assert np.count_nonzero(res) == 1000
	assert list(np.unique(res)) == [1,2]

	assert res[5,5,2] == 2
	assert res[5,5,7] == 1

	assert ct[1] == 1
	assert ct[2] == 1

	labels = np.ones((10,10,10), dtype=bool)
	labels[5,5,2] = 0

	res = fastmorph.fill_holes(labels, return_fill_count=False)
	assert np.all(res == True)

	res = fastmorph.fill_holes(labels, return_fill_count=True)
	assert np.all(res[0] == True)
	assert res[1] == 1

	res = fastmorph.fill_holes(labels, return_removed=True)
	assert np.all(res[0] == True)
	assert res[1] == set()

	res = fastmorph.fill_holes(labels, return_fill_count=True, return_removed=True)
	assert np.all(res[0] == True)
	assert res[1] == 1
	assert res[2] == set()

def test_spherical_open_close_run():
	labels = np.zeros((10,10,10), dtype=bool)
	res = fastmorph.spherical_open(labels, radius=1)
	res = fastmorph.spherical_close(labels, radius=1)

def test_spherical_close():
	labels = np.zeros((10,10,10), dtype=bool)
	labels[4:7,4:7,4:7] = True
	labels[5,5,5] = False

	res = fastmorph.spherical_close(labels, radius=1)
	assert res[5,5,5] == True

