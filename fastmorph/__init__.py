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
  if iterations < 0:
    raise ValueError(f"iterations ({iterations}) must be a positive integer.")
  elif iterations == 0:
    return np.copy(labels, order="F")

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
  erode_border:bool = True,
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

  erode_border: if True, the border is treated as background,
    else it is regarded as a value that would preserve the
    current value. Only has an effect for multilabel erosion.
  """
  if iterations < 0:
    raise ValueError(f"iterations ({iterations}) must be a positive integer.")
  elif iterations == 0:
    return np.copy(labels, order="F")

  if parallel == 0:
    parallel = mp.cpu_count()

  parallel = min(parallel, mp.cpu_count())

  labels = np.asfortranarray(labels)
  while labels.ndim < 2:
    labels = labels[..., np.newaxis]

  output = labels

  for i in range(iterations):
    if mode == Mode.multilabel:
      output = fastmorphops.multilabel_erode(output, erode_border, parallel)
    else:
      output = fastmorphops.grey_erode(output, parallel)

  return output.view(labels.dtype)

def opening(
  labels:np.ndarray, 
  background_only:bool = True,
  parallel:int = 0,
  mode:Mode = Mode.multilabel,
  erode_border:bool = True,
) -> np.ndarray:
  """Performs morphological opening of labels.

  This operator is idempotent with the exception
  of boundary effects.

  background_only is passed through to dilate.
    True: Only evaluate background voxels for dilation.
    False: Allow labels to erode each other as they grow.
  parallel: how many pthreads to use in a threadpool
  mode: 
    Mode.multilabel: are all surrounding pixels the same?
    Mode.grey: use grayscale image dilation (min value)
  erode_border: if True, the border is treated as background,
    else it is regarded as a value that would preserve the
    current value. Only has an effect for multilabel erosion.
  """
  return dilate(
    erode(labels, parallel, mode, iterations=1, erode_border=erode_border),
    background_only, parallel, mode, iterations=1
  )

def closing(
  labels:np.ndarray, 
  background_only:bool = True,
  parallel:int = 0,
  mode:Mode = Mode.multilabel,
  erode_border:bool = True,
) -> np.ndarray:
  """Performs morphological closing of labels.

  This operator is idempotent with the exception
  of boundary effects.

  background_only is passed through to dilate.
    True: Only evaluate background voxels for dilation.
    False: Allow labels to erode each other as they grow.
  parallel: how many pthreads to use in a threadpool
  mode: 
    Mode.multilabel: are all surrounding pixels the same?
    Mode.grey: use grayscale image dilation (min value)
  erode_border: if True, the border is treated as background,
    else it is regarded as a value that would preserve the
    current value. Only has an effect for multilabel erosion.
  """
  return erode(
    dilate(labels, background_only, parallel, mode, iterations=1), 
    parallel, mode, iterations=1, erode_border=erode_border,
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
  fix_borders:bool = False,
  morphological_closing:bool = False,
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
  if np.issubdtype(labels.dtype, bool) and not fix_borders and not morphological_closing:
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

  output = np.zeros(labels.shape, dtype=labels.dtype, order="F")

  removed_set = set()

  for label in range(1, N+1):
    if label in removed_set:
      continue

    slices = all_slices[label]
    if slices is None:
      continue

    binary_image = (cc_labels[slices] == label)

    pixels_filled = 0

    if morphological_closing:
      dilated_binary_image = dilate(binary_image)
      pixels_filled += np.sum(dilated_binary_image != binary_image)
      binary_image = dilated_binary_image
      del dilated_binary_image

    if fix_borders:
      binary_image[:,:,0], pf1 = fill_voids.fill(binary_image[:,:,0], return_fill_count=True)
      binary_image[:,:,-1], pf2 = fill_voids.fill(binary_image[:,:,-1], return_fill_count=True)
      binary_image[:,0,:], pf3 = fill_voids.fill(binary_image[:,0,:], return_fill_count=True)
      binary_image[:,-1,:], pf4 = fill_voids.fill(binary_image[:,-1,:], return_fill_count=True)
      binary_image[0,:,:], pf5 = fill_voids.fill(binary_image[0,:,:], return_fill_count=True)
      binary_image[-1,:,:], pf6 = fill_voids.fill(binary_image[-1,:,:], return_fill_count=True)
      pixels_filled += pf1 + pf2 + pf3 + pf4 + pf5 + pf6

    binary_image, pf7 = fill_voids.fill(
      binary_image, in_place=True, 
      return_fill_count=True
    )
    pixels_filled += pf7

    if morphological_closing:
      eroded_binary_image = erode(binary_image, erode_border=False)
      pixels_filled -= np.sum(eroded_binary_image != binary_image)
      binary_image = eroded_binary_image
      del eroded_binary_image

    fill_counts[label] = pixels_filled
    output[slices][binary_image] = mapping[label]

    if pixels_filled == 0:
      continue

    sub_labels = fastremap.unique(cc_labels[slices][binary_image])
    sub_labels = set(sub_labels)
    sub_labels.discard(label)
    sub_labels.discard(0)
    if not remove_enclosed and sub_labels:
      sub_labels = [ int(l) for l in sub_labels ]
      raise FillError(f"{sub_labels} would have been deleted by this operation.")

    removed_set |= sub_labels

  ret = [ output ]

  if return_fill_count:
    for label in removed_set:
      fill_counts.pop(label, None)
    ret.append(fill_counts)

  if return_removed:
    removed_set = set([ mapping[l] for l in removed_set ])
    ret.append(removed_set)

  return (ret[0] if len(ret) == 1 else tuple(ret))

