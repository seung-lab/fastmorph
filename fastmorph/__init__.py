from typing import Optional, Sequence
import numpy as np
import edt
import fill_voids
import cc3d
import fastremap

AnisotropyType = Optional[Sequence[int]]

def spherical_dilate(
  labels:np.ndarray, 
  radius:float = 1.0, 
  parallel:int = 1, 
  anisotropy:AnisotropyType = None
) -> np.ndarray:
  """
  Dilate foreground binary components.
  """
  assert np.issubdtype(labels.dtype, bool), "Dilation is currently only supported for binary images."
  dt = edt.edt(labels == 0, parallel=parallel, anisotropy=anisotropy)
  return labels | (dt <= radius)

def spherical_erode(
  labels:np.ndarray, 
  radius:float = 1.0, 
  parallel:int = 1, 
  anisotropy:AnisotropyType = None
) -> np.ndarray:
  """
  Erode foreground binary components.
  """
  dt = edt.edt(labels, parallel=parallel, anisotropy=anisotropy, black_border=True)
  return (labels * (dt >= radius)).astype(labels.dtype, copy=False)

def spherical_open(
  labels:np.ndarray, 
  radius:float = 1.0,
  parallel:int = 1, 
  anisotropy:AnisotropyType = None
) -> np.ndarray:
  """Apply a spherical morphological open operation to a binary image."""
  assert np.issubdtype(labels.dtype, bool), "spherical_open is currently only supported for binary images."
  args = [ radius, parallel, anisotropy ]
  return spherical_dilate(spherical_erode(labels, *args), *args)

def spherical_close(
  labels:np.ndarray, 
  radius:float = 1.0,
  parallel:int = 1, 
  anisotropy:AnisotropyType = None
) -> np.ndarray:
  """Apply a spherical morphological close operation to a binary image."""
  assert np.issubdtype(labels.dtype, bool), "spherical_close is currently only supported for binary images."
  args = [ radius, parallel, anisotropy ]
  return spherical_erode(spherical_dilate(labels, *args), *args)

def fill_holes(
  labels:np.ndarray, 
  return_fill_count:bool = False
) -> np.ndarray:
  """
  For fill holes in toplogically closed objects.
  """
  assert np.issubdtype(labels.dtype, (np.integer, bool)), "fill_holes is currently only supported for integer or binary images."
  if np.issubdtype(labels.dtype, bool):
    return fill_voids.fill(labels, return_fill_count=return_fill_count)

  cc_labels, N = cc3d.connected_components(labels, return_N=True)
  stats = cc3d.statistics(cc_labels)
  mapping = fastremap.component_map(cc_labels, labels)

  fill_counts = {}
  output = np.zeros(labels.shape, dtype=labels.dtype)
  slices = stats["bounding_boxes"]
  for label in range(1, N+1):
    binimg, ct = fill_voids.fill(cc_labels[slices[label]] == label, return_fill_count=True)
    output[slices[label]] = mapping[label] * binimg
    fill_counts[mapping[label]] = ct

  if return_fill_count:
    return (output, fill_counts)

  return output
