![Automated Tests](https://github.com/seung-lab/fastmorph/actions/workflows/run_tests.yml/badge.svg) [![PyPI version](https://badge.fury.io/py/fastmorph.svg)](https://badge.fury.io/py/fastmorph)

# fastmorph: multilabel 3D morphological image processing functions.

This is a collection of morphological 3D image operations that are tuned for working with dense 3D labeled images. 

We provide the following multithreaded operations:

- Multi-Label Stenciled Dilation, Erosion, Opening, Closing
- Multi-Label Spherical Erosion
- Binary Spherical Dilation, Opening, and Closing
- Multi-Label Fill Voids

Highlights compared to other libraries:

- Handles multi-labeled images
- Multithreaded
- High performance single-threaded
- Low memory usage
- Dilate computes mode of surrounding labels

Disadvantages versus other libraries:

- Not ideal for grayscale images
- Stencil (structuring element) is fixed size 3x3x3 and all on.


```python
import fastmorph

# may be binary or unsigned integer 2D or 3D image
labels = np.load("my_labels.npy")


# multi-label capable morphological operators
# they use a 3x3x3 all on structuring element
# dilate picks the mode of surrounding labels

# by default only background (0) labels are filled
morphed = fastmorph.dilate(labels, parallel=2)
# processes every voxel
morphed = fastmorph.dilate(labels, background_only=False, parallel=2)

morphed = fastmorph.erode(labels)
morphed = fastmorph.opening(labels, parallel=2)
morphed = fastmorph.closing(labels, parallel=2)

# Dilate only supports binary images at this time.
# Radius is specified in physical units, but
# by default anisotropy = (1,1,1) so it is the 
# same as voxels.
morphed = fastmorph.spherical_dilate(labels, radius=1, parallel=2, anisotropy=(1,1,1))

# open and close require dialate to work and so are binary only for now
morphed = fastmorph.spherical_open(labels, radius=1, parallel=2, anisotropy=(1,1,1))
morphed = fastmorph.spherical_close(labels, radius=1, parallel=2, anisotropy=(1,1,1))

# The rest support multilabel images.
morphed = fastmorph.spherical_erode(labels, radius=1, parallel=2, anisotropy=(1,1,1))

# Note: for boolean images, this function will directly call fill_voids
# and return a scalar for ct 
# For integer images, more processing will be done to deal with multiple labels.
# A dict of { label: num_voxels_filled } for integer images will be returned.
# Note that for multilabel images, by default, if a label is totally enclosed by another,
# a FillError will be raised. If remove_enclosed is True, the label will be overwritten.
filled_labels, ct = fastmorph.fill_holes(labels, return_fill_count=True, remove_enclosed=False)
```

## Performance

A test run on an M1 Macbook Pro on `connectomics.npy.ckl`, a 512<sup>3</sup> volume with over 2000 dense labels had the following results.

```
erode / 1 thread:  7.70 sec
erode / 2 threads:  4.10 sec
erode / 4 threads:  3.02 sec
dilate / background_only=True / 1 thread:  1.15 sec
dilate / background_only=True / 2 threads:  0.65 sec
dilate / background_only=True / 4 threads:  0.48 sec
dilate / background_only=False / 1 thread:  14.31 sec
dilate / background_only=False / 2 threads:  7.47 sec
dilate / background_only=False / 4 threads:  8.09 sec
dilate / background_only=False / 8 threads:  5.46 sec
skimage expand_labels / 1 thread:  75.20 sec
```

### Memory Profiles

<center>
<img src="https://github.com/seung-lab/fastmorph/blob/15c4c27ad3255c8ef959ceb67facd65e18eff2e4/memory-profile-dilate-bg-only-false.jpg" />
</center>

<center>
<img src="https://github.com/seung-lab/fastmorph/blob/15c4c27ad3255c8ef959ceb67facd65e18eff2e4/memory-profile-dilate-bg-only-true.jpg" />
</center>

<center>
<img src="https://github.com/seung-lab/fastmorph/blob/15c4c27ad3255c8ef959ceb67facd65e18eff2e4/memory-profile-skimage-expand_labels.jpg" />
</center>
