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