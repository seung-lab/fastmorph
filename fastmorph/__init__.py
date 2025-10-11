from collections import defaultdict
from enum import Enum
from typing import Optional, Sequence

import numpy as np
import edt
import fill_voids
import cc3d
import fastremap
import multiprocessing as mp

from tqdm import trange

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
  in_place:bool = False,
  progress:bool = False,
  parallel:int = 0,
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
  fix_borders: run 2d fill along the edges of the image on each binary image
  morphological_closing: perform a dilation first, and then erode at the end
    of the hole filling process
  in_place: allow modifications of labels array for reduced memory consumption
  progress: display a progress bar

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

  renumbered_labels, mapping = fastremap.renumber(labels, in_place=in_place)
  mapping = { v:k for k,v in mapping.items() }

  if renumbered_labels.dtype == bool:
    renumbered_labels = renumbered_labels.view(np.uint8)

  while renumbered_labels.ndim > 3 and renumbered_labels.shape[-1] == 1:
    renumbered_labels = renumbered_labels[...,0]

  stats = cc3d.statistics(renumbered_labels)

  N = len(stats["voxel_counts"])

  fill_counts = {}
  all_slices = stats["bounding_boxes"]
  total_counts = stats["voxel_counts"]

  output = np.zeros(labels.shape, dtype=labels.dtype, order="F")

  removed_set = set()

  with trange(1, N, disable=(not progress), desc="Filling Holes") as pbar:
    for label in pbar:
      if label in removed_set:
        continue

      slices = all_slices[label]
      if slices is None:
        continue

      pbar.set_postfix(label=str(mapping[label]))

      binary_image = (renumbered_labels[slices] == label)

      pixels_filled = 0

      if morphological_closing:
        dilated_binary_image = dilate(binary_image, parallel=parallel)
        # pixels_filled += np.sum(dilated_binary_image != binary_image)
        pixels_filled += fastmorphops.count_differences(dilated_binary_image, binary_image)
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
        return_fill_count=True,
      )
      pixels_filled += pf7

      if morphological_closing:
        eroded_binary_image = erode(binary_image, erode_border=False, parallel=parallel)
        # pixels_filled -= np.sum(eroded_binary_image != binary_image)
        pixels_filled -= fastmorphops.count_differences(eroded_binary_image, binary_image)
        binary_image = eroded_binary_image
        del eroded_binary_image

      fill_counts[label] = pixels_filled

      # This function call is equivalent to:
      # output[slices][binary_image] = mapping[label]
      # 
      # It avoids an Fortran vs C order impedance mismatch
      # that was causing a large performance hit.
      fastmorphops.draw_with_mask_f_order(
        output, 
        int(slices[0].start), int(slices[0].stop),
        int(slices[1].start), int(slices[1].stop),
        int(slices[2].start), int(slices[2].stop),
        binary_image,
        mapping[label]
      )

      if pixels_filled == 0:
        continue

      flat_binary_image = binary_image.ravel('F')
      flat_renumber = renumbered_labels[slices].ravel('F')
      mask = flat_renumber[flat_binary_image]
      sub_labels, sub_counts = fastremap.unique(mask, return_counts=True)
      del mask

      if morphological_closing:
        sub_counts = { l:c for l,c in zip(sub_labels, sub_counts) }
        sub_labels = set([ lbl for lbl in sub_labels if sub_counts[lbl] == total_counts[lbl] ])
      else:
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

def _pairs_to_connection_list(itr):
  tmp = defaultdict(set)
  for l1, l2 in itr:
    tmp[l1].add(l2)
    tmp[l2].add(l1)
  return tmp

def _fix_holes_2d(slice_labels:np.ndarray, merge_threshold:float) -> set[int]:

  holes2d = set()
  edge_surface_area = cc3d.contacts(slice_labels, connectivity=4)
  connections2d = _pairs_to_connection_list(edge_surface_area.keys())      

  slices2d = [
    np.s_[0,:], np.s_[-1,:],
    np.s_[:,0], np.s_[:,-1],
  ]

  # The edges of the slice are touching other borders and
  # it is impossible to say if they are holes or not so
  # protect those labels that touch the edge
  protected = set()
  for slc in slices2d:
    protected.update(fastremap.unique(slice_labels[slc]))

  for segid, neighbors in connections2d.items():
    neighbors.discard(0)
    neighbors = [ 
      (n, edge_surface_area[tuple(sorted([ segid, n ]))]) 
      for n in neighbors
    ]
    total_edge_area = sum([ x[1] for x in neighbors ])

    if total_edge_area == 0:
      continue

    neighbors = [ 
      n for n, area in neighbors 
      if (area/total_edge_area) >= merge_threshold 
    ]

    if len(neighbors) == 1:
      holes2d.add(segid)
    holes2d -= protected

  return holes2d

def fill_holes_v2(
  labels:np.ndarray,
  fix_borders:bool = False,
  merge_threshold:float = 1.0,
  anisotropy:tuple[float,float,float] = (1.0, 1.0, 1.0),
  parallel:int = 0,
) -> tuple["CrackleArray", "CrackleArray"]:
  import crackle

  # Ensure bg 0 gets treated as a connected component
  cc_labels, N = cc3d.connected_components(
    labels + 1, 
    return_N=True, 
    connectivity=26,
  )

  sentinel = np.iinfo(labels.dtype).max

  orig_map = fastremap.component_map(cc_labels, labels)
  orig_map[0] = 0
  orig_map[sentinel] = sentinel

  surface_areas = cc3d.contacts(
    cc_labels, 
    connectivity=26, 
    surface_area=True, 
    anisotropy=tuple(anisotropy),
  )

  connections = _pairs_to_connection_list(surface_areas.keys())

  slices = [
    np.s_[:,:,0],
    np.s_[:,:,-1],
    np.s_[0,:,:],
    np.s_[-1,:,:],
    np.s_[:,0,:],
    np.s_[:,-1,:],
  ]

  edge_labels = set()
  bg_edge_labels = set()

  for slc in slices:
    slice_labels = cc_labels[slc]
    uniq = set(fastremap.unique(slice_labels))

    for u in uniq:
      if orig_map[u] == 0:
        bg_edge_labels.add(u)

    if fix_borders:
      uniq -= _fix_holes_2d(slice_labels, merge_threshold)

    edge_labels.update(uniq)

  for bgl in bg_edge_labels:
    for lbl in connections[bgl]:
      edge_labels.add(lbl)

  del uniq
  del slice_labels

  candidate_holes = set(range(1,N+1))
  holes = candidate_holes.difference(edge_labels)
  del candidate_holes

  def best_contact(segid, edges):
    if not len(edges):
      return sentinel

    contact_surfaces = [ 
      (contact, surface_areas[tuple(sorted((hole, contact)))]) 
      for contact in edges
    ]
    total_area = sum([ x[1] for x in contact_surfaces ])
    
    if total_area == 0:
      if len(contact_surfaces) == 1:
        return contact_surfaces[0][0]
      else:
        return segid

    contact_surfaces.sort(key=lambda x: x[1])
    max_contact, max_area = contact_surfaces[-1]
    max_area /= total_area

    if max_area >= merge_threshold:
      return max_contact
    else:
      return segid

  remap = { i:i for i in range(N+1) }

  for hole in list(holes):
    if len(connections[hole]):
      edges = connections[hole].intersection(edge_labels)
      if not len(edges):
        edges = connections[hole]
      remap[hole] = best_contact(hole, edges)
    else:
      holes.discard(hole)

  del connections
  del edge_labels

  remap = { k: orig_map[v] for k,v in remap.items()  }

  filled_labels = fastremap.remap(
    cc_labels, remap, in_place=False,
  ).astype(labels.dtype, copy=False)
  del remap

  hole_labels = fastremap.mask_except(cc_labels, list(holes))
  hole_labels = np.where(hole_labels, labels, 0)

  return (filled_labels, hole_labels)
