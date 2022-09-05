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
filled_labels, ct = fastmorph.fill_holes(labels, return_fill_count=True)
```

