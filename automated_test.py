import pytest
import numpy as np
import fastmorph

def test_spherical_dilate():
	labels = np.zeros((10,10,10), dtype=bool)
	res = fastmorph.spherical_dilate(labels, radius=1000)
	assert np.all(res == False)

	labels[5,5,5] = True
	radius = 5 * np.sqrt(3) + 0.000001
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


def test_multilabel_dilate_3d():
	labels = np.zeros((3,3,3), dtype=bool)

	out = fastmorph.dilate(labels)
	assert not np.any(out)

	labels[1,1,1] = True

	out = fastmorph.dilate(labels)
	assert np.all(out)

	labels = np.zeros((3,3,3), dtype=bool)
	labels[0,0,0] = True
	out = fastmorph.dilate(labels)

	ans = np.zeros((3,3,3), dtype=bool)
	ans[:2,:2,:2] = True

	assert np.all(out == ans)

	labels = np.zeros((3,3,3), dtype=int)
	labels[0,1,1] = 1
	labels[2,1,1] = 2

	out = fastmorph.dilate(labels)
	ans = np.ones((3,3,3), dtype=int)
	ans[2,:,:] = 2
	assert np.all(ans == out)

	labels = np.zeros((3,3,3), dtype=int, order="F")
	labels[0,1,1] = 1
	labels[1,1,1] = 2
	labels[2,1,1] = 2

	out = fastmorph.dilate(labels)
	ans = np.ones((3,3,3), dtype=int, order="F")
	ans[1:,:,:] = 2
	assert np.all(ans == out)

def test_multilabel_dilate_2d():
	labels = np.zeros((3,3), dtype=bool)

	out = fastmorph.dilate(labels)
	assert not np.any(out)

	labels[1,1] = True

	out = fastmorph.dilate(labels)
	assert np.all(out)

	labels = np.zeros((3,3), dtype=bool)
	labels[0,0] = True
	out = fastmorph.dilate(labels)

	ans = np.zeros((3,3), dtype=bool)
	ans[:2,:2] = True

	assert np.all(out == ans)

	labels = np.zeros((3,3), dtype=int)
	labels[0,1] = 1
	labels[2,1] = 2

	out = fastmorph.dilate(labels)
	ans = np.ones((3,3), dtype=int)
	ans[2,:] = 2
	assert np.all(ans == out)

	labels = np.zeros((3,3), dtype=int, order="F")
	labels[0,1] = 1
	labels[1,1] = 2
	labels[2,1] = 2

	out = fastmorph.dilate(labels)
	ans = np.ones((3,3), dtype=int, order="F")
	ans[1:,:] = 2
	assert np.all(ans == out)

def test_multilabel_erode_3d():
	labels = np.ones((3,3,3), dtype=bool)
	out = fastmorph.erode(labels)
	assert np.sum(out) == 1 and out[1,1,1] == True

	out = fastmorph.erode(out)
	assert not np.any(out)

	out = fastmorph.erode(out)
	assert not np.any(out)

	labels = np.ones((3,3,3), dtype=int, order="F")
	labels[0,:,:] = 1
	labels[1,:,:] = 2
	labels[2,:,:] = 3

	out = fastmorph.erode(labels)
	ans = np.zeros((3,3,3), dtype=int, order="F")

	assert np.all(ans == out)

	labels = np.zeros((5,5,5), dtype=bool)
	labels[1:4,1:4,1:4] = True
	out = fastmorph.erode(labels)
	assert np.sum(out) == 1 and out[2,2,2] == True

	labels = np.ones((5,5,5), dtype=bool)
	out = fastmorph.erode(labels)
	assert np.sum(out) == 27

def test_multilabel_erode_2d():
	labels = np.ones((3,3), dtype=bool)
	out = fastmorph.erode(labels)
	assert np.sum(out) == 1 and out[1,1] == True

	out = fastmorph.erode(out)
	assert not np.any(out)

	out = fastmorph.erode(out)
	assert not np.any(out)

	labels = np.ones((3,3), dtype=int, order="F")
	labels[0,:] = 1
	labels[1,:] = 2
	labels[2,:] = 3

	out = fastmorph.erode(labels)
	ans = np.zeros((3,3), dtype=int, order="F")

	assert np.all(ans == out)

	labels = np.zeros((5,5), dtype=bool)
	labels[1:4,1:4] = True
	out = fastmorph.erode(labels)
	assert np.sum(out) == 1 and out[2,2] == True

	labels = np.ones((5,5), dtype=bool)
	out = fastmorph.erode(labels)
	assert np.sum(out) == 9

@pytest.mark.parametrize('dtype', [
	np.uint8,np.uint16,np.uint32,np.uint64,
	np.int8,np.int16,np.int32,np.int64,
])
def test_grey_erode(dtype):
	labels = np.arange(27, dtype=dtype).reshape((3,3,3), order="F")
	out = fastmorph.erode(labels, mode=fastmorph.Mode.grey)

	ans = np.array([
		[
			[0, 0, 1],
			[0, 0, 1],
			[3, 3, 4],
		],
		[
			[0, 0, 1],
			[0, 0, 1],
			[3, 3, 4],
		],
		[
			[9, 9, 10],
			[9, 9, 10],
			[12, 12, 13],
		],
	], dtype=dtype).T

	assert np.all(out == ans)

	out = fastmorph.erode(out, mode=fastmorph.Mode.grey)
	assert np.all(out == 0)

@pytest.mark.parametrize('dtype', [
	np.uint8,np.uint16,np.uint32,np.uint64,
	np.int8,np.int16,np.int32,np.int64,
])
def test_grey_dilate(dtype):
	L = 5
	H = 10

	labels = np.zeros((3,3,3), dtype=dtype)
	labels[0,0,0] = L
	labels[2,2,2] = H

	out = fastmorph.dilate(labels, mode=fastmorph.Mode.grey)

	ans = np.array([
		[
			[L, L, 0],
			[L, L, 0],
			[0, 0, 0],
		],
		[
			[L, L, 0],
			[L, H, H],
			[0, H, H],
		],
		[
			[0, 0, 0],
			[0, H, H],
			[0, H, H],
		],
	], dtype=dtype).T

	assert np.all(out == ans)

	out = fastmorph.dilate(out, mode=fastmorph.Mode.grey)
	assert np.all(out == H)

def test_grey_dilate_bool():
	labels = np.zeros((3,3,3), dtype=bool)
	labels[1,1,1] = True

	out = fastmorph.dilate(labels, mode=fastmorph.Mode.grey)
	assert np.all(out == True)


