from enum import Enum
from typing import Optional, Sequence
import numpy as np
import edt
import fill_voids
import cc3d
import fastremap
import multiprocessing as mp

import fastmorphops

AnisotropyType = Optional[Sequence[int]]

class Mode(Enum):
  multilabel = 1
  grey = 2

def dilate(
  labels:np.ndarray,
  background_only:bool = True,
  parallel:int = 0,
  mode:Mode = Mode.multilabel,
  iterations:int = 1,
) -> np.ndarray:
  """
  Dilate forground labels using a 3x3x3 stencil with
  all elements "on".

  The mode of the voxels surrounding the stencil wins.

  labels: a 3D numpy array containing integer labels
    representing shapes to be dilated.

  background_only:
    True: Only evaluate background voxels for dilation.
    False: Allow labels to erode each other as they grow.

  parallel: how many pthreads to use in a threadpool

  mode: 
    Mode.multilabel: use mode of stencil for dilation
    Mode.grey: use grayscale image dilation (max value)

  iterations: number of times to iterate the result
  """
  if iterations <= 0:
    raise ValueError(f"iterations ({iterations}) must be a positive integer.")

  if parallel == 0:
    parallel = mp.cpu_count()
  parallel = min(parallel, mp.cpu_count())

  labels = np.asfortranarray(labels)
  while labels.ndim < 2:
    labels = labels[..., np.newaxis]
  
  output = labels

  for i in range(iterations):
    if mode == Mode.multilabel:
      output = fastmorphops.multilabel_dilate(output, background_only, parallel)
    else:
      output = fastmorphops.grey_dilate(output, parallel)

  return output.view(labels.dtype)

def erode(
  labels:np.ndarray, 
  parallel:int = 0,
  mode:Mode = Mode.multilabel,
  iterations:int = 1,
) -> np.ndarray:
  """
  Erodes forground labels using a 3x3x3 stencil with
  all elements "on".

  labels: a 3D numpy array containing integer labels
    representing shapes to be dilated.

  parallel: how many pthreads to use in a threadpool

  mode: 
    Mode.multilabel: are all surrounding pixels the same?
    Mode.grey: use grayscale image dilation (min value)

  iterations: number of times to iterate the result
  """
  if iterations <= 0:
    raise ValueError(f"iterations ({iterations}) must be a positive integer.")

  if parallel == 0:
    parallel = mp.cpu_count()

  parallel = min(parallel, mp.cpu_count())

  labels = np.asfortranarray(labels)
  while labels.ndim < 2:
    labels = labels[..., np.newaxis]

  output = labels

  for i in range(iterations):
    if mode == Mode.multilabel:
      output = fastmorphops.multilabel_erode(output, parallel)
    else:
      output = fastmorphops.grey_erode(output, parallel)

  return output.view(labels.dtype)

def opening(
  labels:np.ndarray, 
  background_only:bool = True,
  parallel:int = 0,
  mode:Mode = Mode.multilabel,
  iterations:int = 1,
) -> np.ndarray:
  """Performs morphological opening of labels.

  background_only is passed through to dilate.
    True: Only evaluate background voxels for dilation.
    False: Allow labels to erode each other as they grow.
  parallel: how many pthreads to use in a threadpool
  """
  return dilate(
    erode(labels, parallel, mode, iterations),
    background_only, parallel, mode, iterations
  )

def closing(
  labels:np.ndarray, 
  background_only:bool = True,
  parallel:int = 0,
  mode:Mode = Mode.multilabel,
  iterations:int = 1,
) -> np.ndarray:
  """Performs morphological closing of labels.

  background_only is passed through to dilate.
    True: Only evaluate background voxels for dilation.
    False: Allow labels to erode each other as they grow.
  parallel: how many pthreads to use in a threadpool
  """
  return erode(
    dilate(labels, background_only, parallel, mode, iterations), 
    parallel, mode, iterations
  )

def spherical_dilate(
  labels:np.ndarray, 
  radius:float = 1.0, 
  parallel:int = 0, 
  anisotropy:AnisotropyType = None,
  in_place:bool = False,
) -> np.ndarray:
  """
  Dilate foreground binary components.

  Uses the exact euclidean distance transform to determine voxel distances
  (hence spherical). Does not use a structuring element.

  labels: input labels (must be a boolean image)
  radius: physical distance (considering anisotropy) to dilate to (inclusive range)
  parallel: use this many threads to compute the distance transform
  anisotropy: voxel resolution in x, y, and z
  in_place: save memory by modifying labels directly instead of creating a new image

  Returns: dilated binary image
  """
  assert np.issubdtype(labels.dtype, bool), "Dilation is currently only supported for binary images."
  dt = edt.edt(labels == 0, parallel=parallel, anisotropy=anisotropy)
  
  binary_image = lambda: dt <= radius
  if in_place:
    labels |= binary_image()
    return labels

  return labels | binary_image()

def spherical_erode(
  labels:np.ndarray, 
  radius:float = 1.0, 
  parallel:int = 0, 
  anisotropy:AnisotropyType = None,
  in_place:bool = False,
) -> np.ndarray:
  """
  Erode foreground multi-label components.

  Uses the exact euclidean distance transform to determine voxel distances
  (hence spherical). Does not use a structuring element.

  labels: input labels (must be a boolean image)
  radius: physical distance (considering anisotropy) to dilate to (inclusive range)
  parallel: use this many threads to compute the distance transform
  anisotropy: voxel resolution in x, y, and z
  in_place: save memory by modifying labels directly instead of creating a new image

  Returns: eroded multi-label iamge  
  """
  dt = edt.edt(labels, parallel=parallel, anisotropy=anisotropy, black_border=True)

  binary_image = lambda: dt >= radius
  if in_place:
    labels *= binary_image()
    return labels
  return (labels * binary_image()).astype(labels.dtype, copy=False)

def spherical_open(
  labels:np.ndarray, 
  radius:float = 1.0,
  parallel:int = 0, 
  anisotropy:AnisotropyType = None,
  in_place:bool = False,
) -> np.ndarray:
  """Apply a spherical morphological open operation to a binary image."""
  assert np.issubdtype(labels.dtype, bool), "spherical_open is currently only supported for binary images."
  args = [ radius, parallel, anisotropy, in_place ]
  return spherical_dilate(spherical_erode(labels, *args), *args)

def spherical_close(
  labels:np.ndarray, 
  radius:float = 1.0,
  parallel:int = 0, 
  anisotropy:AnisotropyType = None,
  in_place:bool = False,
) -> np.ndarray:
  """Apply a spherical morphological close operation to a binary image."""
  assert np.issubdtype(labels.dtype, bool), "spherical_close is currently only supported for binary images."
  args = [ radius, parallel, anisotropy, in_place ]
  return spherical_erode(spherical_dilate(labels, *args), *args)


class FillError(Exception):
  pass

def fill_holes(
  labels:np.ndarray, 
  return_fill_count:bool = False,
  remove_enclosed:bool = False,
  return_removed:bool = False,
) -> np.ndarray:
  """
  For fill holes in toplogically closed objects.

  return_fill_count: return the total number of pixels filled in
    for boolean array: integer
    for integer array: { label: count }
  remove_enclosed: if one label totally encloses another, the interior label
    will be removed. Otherwise, raise a FillError.
  return_removed: returns the set of totally enclosed 
    labels that were eliminated

  Return value: (filled_labels, fill_count (if specified), removed_set (if specified))
  """
  assert np.issubdtype(labels.dtype, np.integer) or np.issubdtype(labels.dtype, bool), "fill_holes is currently only supported for integer or binary images."
  if np.issubdtype(labels.dtype, bool):
    filled_labels, filled_ct = fill_voids.fill(labels, return_fill_count=True)
    ret = [ filled_labels ]
    if return_fill_count:
      ret.append(filled_ct)
    if return_removed:
      ret.append(set())
    return (ret[0] if len(ret) == 1 else tuple(ret))

  cc_labels, N = cc3d.connected_components(labels, return_N=True)
  stats = cc3d.statistics(cc_labels)
  mapping = fastremap.component_map(cc_labels, labels)

  fill_counts = {}
  all_slices = stats["bounding_boxes"]

  labels = list(range(1, N+1))
  labels_set = set(labels)
  removed_set = set()

  for label in labels:
    if label not in labels_set:
      continue

    slices = all_slices[label]
    if slices is None:
      continue

    binary_image = (cc_labels[slices] == label)
    binary_image, pixels_filled = fill_voids.fill(
      binary_image, in_place=True, 
      return_fill_count=True
    )
    fill_counts[label] = pixels_filled
    if pixels_filled == 0:
      continue

    sub_labels = set(fastremap.unique(cc_labels[slices] * binary_image))
    sub_labels.remove(label)
    sub_labels.discard(0)
    if not remove_enclosed and sub_labels:
      raise FillError(f"{sub_labels} would have been deleted by this operation.")
    
    labels_set -= sub_labels
    removed_set |= sub_labels
    cc_labels[slices] = cc_labels[slices] * ~binary_image + mapping[label] * binary_image

  ret = [ cc_labels ]

  if return_fill_count:
    for label in removed_set:
      fill_counts.pop(label, None)
    ret.append(fill_counts)

  if return_removed:
    ret.append(removed_set)

  return (ret[0] if len(ret) == 1 else tuple(ret))

