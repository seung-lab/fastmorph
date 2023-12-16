![Automated Tests](https://github.com/seung-lab/fastmorph/actions/workflows/run_tests.yml/badge.svg)

# fastmorph: multilabel 3D morphological image processing functions.


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

