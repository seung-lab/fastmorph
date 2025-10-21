![Automated Tests](https://github.com/seung-lab/fastmorph/actions/workflows/run_tests.yml/badge.svg) [![PyPI version](https://badge.fury.io/py/fastmorph.svg)](https://badge.fury.io/py/fastmorph)

# fastmorph: multilabel 3D morphological image processing functions.

This is a collection of morphological 3D image operations that are tuned for working with dense 3D labeled images. 

We provide the following multithreaded (except where noted) operations:

- Multi-Label Stenciled Dilation, Erosion, Opening, Closing
- Grayscale Stenciled Dilation, Erosion, Opening, Closing
- Multi-Label Spherical Erosion, Dilation, Opening, and Closing
- Multi-Label Fill Voids (mostly single threaded)

Highlights compared to other libraries:

- Handles multi-labeled images
- Multithreaded
- High performance single-threaded
- Low memory usage
- Dilate computes mode of surrounding labels

Disadvantages versus other libraries:

- Stencil (structuring element) is fixed size 3x3x3 and all on.

## Installation

The "spherical" extra dependency enables multilabel spherical dilation, opening, and closing by installing scipy.

```
pip install "fastmorph[spherical]"
```

## Examples


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

# You can select grayscale dilation, erosion, opening, and 
# closing by passing in a different Mode enum.
# The options are Mode.grey and Mode.multilabel
morphed = fastmorph.dilate(labels, mode=fastmorph.Mode.grey)
morphed = fastmorph.erode(labels, mode=fastmorph.Mode.grey)

# Radius is specified in physical units, but
# by default anisotropy = (1,1,1) so it is the 
# same as voxels.
morphed = fastmorph.spherical_dilate(labels, radius=1, parallel=2, anisotropy=(1,1,1))
morphed = fastmorph.spherical_open(labels, radius=1, parallel=2, anisotropy=(1,1,1))
morphed = fastmorph.spherical_close(labels, radius=1, parallel=2, anisotropy=(1,1,1))
morphed = fastmorph.spherical_erode(labels, radius=1, parallel=2, anisotropy=(1,1,1))

# Rapid multilabel hole filling. There are two versions that use different techniques
# and have different interfaces for their "aggressive" modes. Both modes fill
# holes appropriately by default.
#
# Generally speaking, fill_holes_v2 will be much faster. v2 uses a 
# mostly linear time contact graph analysis. v1 analyzes a sequence 
# of binary images. v2 exhibits much better scaling behavior and supports 
# returning the filled and hole labels as  CrackleArray compressed objects 
# to save memory.
# 
# The main advantage of v1 is that it includes a morphological closure mode
# that operates on voxels for closing small holes. The downside is that this
# can modify the surface of the object.
#
# v2 allows merging holes that are less than 100% closed, but if this
# threshold is set too high, holes won't be closed. If it is too low,
# improper merging can occur. 
# 
# In both methods, objects that contact the sides or more than one side
# (in the case of fix_borders) cannot be merged.

filled_labels, hole_labels = fastmorph.fill_holes_v2(labels)
# requires: pip install crackle-codec
# returns as compressed CrackleArrays that have speedy access to labels 
# in the compressed state (often hundreds of times smaller than the full array)
filled_labels, hole_labels = fastmorph.fill_holes_v2(labels, return_crackle=True)

# fix_borders runs hole filling for each object on the edge to reduce edge contacts
filled_labels, hole_labels = fastmorph.fill_holes_v2(labels, fix_borders=True)

# merge_threshold (range 0.0 - 1.0) controls how much surface area can be 
# "exposed" for a hole to still be filled. The default (1.0) means a hole
# must be perfectly sealed (typical for hole filling algorithms).
filled_labels, hole_labels = fastmorph.fill_holes_v2(labels, merge_threshold=0.97)

# Note: for boolean images, this function will directly call fill_voids
# and return a scalar for ct 
# For integer images, more processing will be done to deal with multiple labels.
# A dict of { label: num_voxels_filled } for integer images will be returned.
# Note that for multilabel images, by default, if a label is totally enclosed by another,
# a FillError will be raised. If remove_enclosed is True, the label will be overwritten.
filled_labels, ct = fastmorph.fill_holes_v1(labels, return_fill_count=True, remove_enclosed=False)

# If the holes in your segmentation are imperfectly sealed, consider
# using the following options.
filled_labels = fastmorph.fill_holes_v1(
	labels, 
	# runs 2d fill on the sides of the cube for each binary image
	fix_borders=True, 
	# does a dilate and then an erode after filling holes
	morphological_closing=True,
)

```

## Performance

A test run on an M1 Macbook Pro on `connectomics.npy.ckl`, a 512<sup>3</sup> volume with over 2000 dense labels had the following results for multilabel processing.

```
erode / 1 thread: 1.553 sec
erode / 2 threads: 0.885 sec
erode / 4 threads: 0.651 sec
dilate / background_only=True / 1 thread: 1.100 sec
dilate / background_only=True / 2 threads: 0.632 sec
dilate / background_only=True / 4 threads: 0.441 sec
dilate / background_only=False / 1 thread: 11.783 sec
dilate / background_only=False / 2 threads: 5.944 sec
dilate / background_only=False / 4 threads: 4.291 sec
dilate / background_only=False / 8 threads: 3.298 sec
scipy grey_dilation / 1 thread 14.648 sec
scipy grey_erode / 1 thread: 14.412 sec
skimage expand_labels / 1 thread: 62.248 sec
```

Test run on an M1 Macbook Pro with `ws.npy.ckl` a 512<sup>3</sup> volume with tens of thousands of components for multilabel processing.

```
erode / 1 thread: 2.380 sec
erode / 2 threads: 1.479 sec
erode / 4 threads: 1.164 sec
dilate / background_only=True / 1 thread: 1.598 sec
dilate / background_only=True / 2 threads: 1.011 sec
dilate / background_only=True / 4 threads: 0.805 sec
dilate / background_only=False / 1 thread: 25.182 sec
dilate / background_only=False / 2 threads: 13.513 sec
dilate / background_only=False / 4 threads: 8.749 sec
dilate / background_only=False / 8 threads: 6.640 sec
scipy grey_dilation / 1 thread 21.109 sec
scipy grey_erode / 1 thread: 20.305 sec
skimage expand_labels / 1 thread: 63.247 sec
```

Here is the performance on a completely zeroed 512<sup>3</sup> volume for multilabel processing.

```
erode / 1 thread: 0.462 sec
erode / 2 threads: 0.289 sec
erode / 4 threads: 0.229 sec
dilate / background_only=True / 1 thread: 2.337 sec
dilate / background_only=True / 2 threads: 1.344 sec
dilate / background_only=True / 4 threads: 1.021 sec
dilate / background_only=False / 1 thread: 2.267 sec
dilate / background_only=False / 2 threads: 1.251 sec
dilate / background_only=False / 4 threads: 0.944 sec
dilate / background_only=False / 8 threads: 0.718 sec
scipy grey_dilation / 1 thread 13.516 sec
scipy grey_erode / 1 thread: 13.326 sec
skimage expand_labels / 1 thread: 35.243 sec
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
