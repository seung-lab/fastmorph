![Automated Tests](https://github.com/seung-lab/fastmorph/actions/workflows/run_tests.yml/badge.svg)

# fastmorph: multilabel 3D morphological image processing functions.


```python
import fastmorph

# may be binary or unsigned integer 2D or 3D image
labels = np.load("my_labels.npy")

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

