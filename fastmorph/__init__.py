from typing import Union, Iterator

from collections import defaultdict
from enum import Enum
from typing import Optional, Sequence

import numpy as np
import numpy.typing as npt
import edt
import fill_voids
import cc3d
import fastremap
import multiprocessing as mp

try:
  import crackle
  HAS_CRACKLE = True
except ImportError:
  HAS_CRACKLE = False

from tqdm import trange

import fastmorphops

AnisotropyType = Optional[Iterator[float]]

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
  if np.issubdtype(labels.dtype, bool):
    dt = edt.edt(labels == 0, parallel=parallel, anisotropy=anisotropy)
    binary_image = dt <= radius
    del dt

    if in_place:
      labels |= binary_image
      return labels

    binary_image |= labels
    return binary_image

  import scipy.ndimage
  dt, indices = scipy.ndimage.distance_transform_edt(
    labels == 0, 
    return_distances=True, 
    return_indices=True,
    sampling=anisotropy,
  )
  mask = dt <= radius
  del dt
  dilated = labels[tuple(indices)]
  del indices
  dilated *= mask
  del mask

  if in_place:
    labels |= dilated
    return labels
  else:
    dilated |= labels
    return dilated

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
  binary_image = dt >= radius
  del dt

  if not in_place:
    labels = labels.copy()

  labels *= binary_image
  return labels

def spherical_open(
  labels:np.ndarray, 
  radius:float = 1.0,
  parallel:int = 0, 
  anisotropy:AnisotropyType = None,
  in_place:bool = False,
) -> np.ndarray:
  """Apply a spherical morphological open operation."""
  args = [ radius, parallel, anisotropy, in_place ]
  return spherical_dilate(spherical_erode(labels, *args), *args)

def spherical_close(
  labels:np.ndarray, 
  radius:float = 1.0,
  parallel:int = 0, 
  anisotropy:AnisotropyType = None,
  in_place:bool = False,
) -> np.ndarray:
  """Apply a spherical morphological close operation."""
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
  For filling holes in toplogically closed objects.

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

fill_holes_v1 = fill_holes

def _pairs_to_connection_list(itr:Iterator[tuple[int,int]]) -> defaultdict[set[int]]:
  tmp = defaultdict(set)
  for l1, l2 in itr:
    tmp[l1].add(l2)
    tmp[l2].add(l1)
  return tmp

def _fill_holes_2d(
  slice_labels:npt.NDArray[np.number], 
  zero_map:dict[int,int],
) -> npt.NDArray[np.number]:

  output = np.zeros(slice_labels.shape, dtype=slice_labels.dtype, order="F")

  cc_labels, N = cc3d.connected_components(
    slice_labels,
    connectivity=4,
    return_N=True,
  )

  if N <= 1:
    return slice_labels, set()

  orig_map = fastremap.component_map(cc_labels, slice_labels)

  stats = cc3d.statistics(cc_labels)
  bboxes = stats["bounding_boxes"]

  sublabels = set()

  for segid in range(1,N):
    if segid in sublabels:
      continue
    
    slc = bboxes[segid]
    binary_image = cc_labels[slc] == segid
    
    if zero_map[orig_map[segid]] == 0:
      output[slc][binary_image] = orig_map[segid]
      continue
    
    binary_image = fill_voids.fill(binary_image)

    enclosed = set(fastremap.unique(cc_labels[slc][binary_image]))
    enclosed.discard(0)
    enclosed.discard(segid)
    sublabels.update(enclosed)
    output[slc][binary_image] = orig_map[segid]

  sublabels = set([ orig_map[sid] for sid in sublabels ])
  return output, sublabels

def _true_label(
  hole:int, 
  edges:set[int], 
  connections:dict[int,list[int]],
  surface_areas:dict[tuple[int,int],float],
  merge_threshold:float,
  visited:np.ndarray,
) -> tuple[int, set[int]]:
  assert 0.0 <= merge_threshold <= 1.0

  if hole in edges:
    return hole, set()

  stack = [ hole ]
  found_edges = set()
  hole_group = set()

  while stack:
    label = stack.pop()

    if label in edges:
      found_edges.add(label)
      visited[label] = True
      continue

    hole_group.add(label)

    if visited[label]:
      continue

    visited[label] = True

    for next_label in connections[label]:
      stack.append(next_label)

  if merge_threshold > 0.0 and len(found_edges) > 1:
    areas = defaultdict(int)
    total_area = 0

    if len(hole_group) * len(found_edges) < len(surface_areas):
      for edge in found_edges:
        for hole_i in hole_group:
          area = surface_areas.get(tuple(sorted([edge, hole_i])), 0)
          areas[edge] += area
          total_area += area
    else:
      subset = [ 
        pair for pair in surface_areas.keys() 
        if (
          (pair[0] in hole_group and pair[1] in found_edges)
          or (pair[1] in hole_group and pair[0] in found_edges)
        )
      ]

      for pair in subset:
        area = surface_areas[pair]
        edge = pair[0] if pair[0] in found_edges else pair[1]
        areas[edge] += area
        total_area += area

    for edge in list(found_edges):
      if areas[edge] / total_area < merge_threshold:
        found_edges.discard(edge)

  if len(found_edges) == 1:
    return next(iter(found_edges)), hole_group
  
  return hole, hole_group

def _fill_binary_image(
  binary_image:npt.NDArray[np.bool_],
  fix_borders:bool,
  return_crackle:bool,
  parallel:int,
) -> Union[
  tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]],
  tuple["CrackleArray", "CrackleArray"]
]:
  if fix_borders:
    binary_image[:,:,0] = fill_voids.fill(binary_image[:,:,0])
    binary_image[:,:,-1] = fill_voids.fill(binary_image[:,:,-1])
    binary_image[:,0,:] = fill_voids.fill(binary_image[:,0,:])
    binary_image[:,-1,:] = fill_voids.fill(binary_image[:,-1,:])
    binary_image[0,:,:] = fill_voids.fill(binary_image[0,:,:])
    binary_image[-1,:,:] = fill_voids.fill(binary_image[-1,:,:])
  
  binary_image = fill_voids.fill(binary_image)

  order = "F" if binary_image.flags.f_contiguous else "C"

  if return_crackle:
    filled = crackle.compressa(binary_image, parallel=parallel)
    holes = crackle.CrackleArray(crackle.zeros(binary_image.shape, dtype=bool, order=order))
    holes.parallel = parallel
    return filled, holes

  return binary_image, np.zeros(binary_image.shape, dtype=bool, order=order)

def fill_holes_v2(
  labels:npt.NDArray[np.number],
  fix_borders:bool = False,
  merge_threshold:float = 1.0,
  anisotropy:AnisotropyType = (1.0, 1.0, 1.0),
  parallel:int = 0,
  return_crackle:bool = False,
) -> Union[
  tuple[npt.NDArray[np.number], npt.NDArray[np.number]], 
  tuple["CrackleArray", "CrackleArray"]
]:
  """
  For filling holes in toplogically closed objects using a faster method
  for multilabel objects.

  Base Mulit-Label Fill Algorithm:

    1. Color all 6-connected components (even background)
    2. Compute surface area of adjacent regions (6-connected)
    3. Compute connection list { label: [neighbor labels], ... }
    4. Create a set of candidate holes from all connected components
      and then filter them by the set difference with labels touching
      the edges of the cutout.
        - To account for e.g. a foreground object surrounded by 
          background, also label as edges all objects touching
          background that touches the image border.
    5. For each hole, walk through the region graph to identify 
      all neighboring holes and edges. 
    6. If there is more than one edge label, do nothing. If 
      there is only one edge label, all the neighboring holes
      are part of a hole group, label them with that edge label.

  For boolean images, simply use the fill_voids algorithm since
  that is faster.

  NOTE: if crackle-codec is installed, this function will have reduced memory usage.

  return_fill_count: return the total number of pixels filled in
    for boolean array: integer
    for integer array: { label: count }
  fix_borders: along each edge of the image, try filling each label as a binary
    image and consider labels removed to be holes and fill them. holes will be
    returned without this filling artifact
  anisotropy: distortion along each axis, used for calculating contact surfaces
  merge_threshold: by default, only objects that are totally enclosed by a single
    label are considered holes, but sometimes they peek out slightly (e.g. an
    organelle touching the membrane of a cell). You can set merge_threshold from
    0.0 to 1.0. This is the fraction of surface area that must be enclosed by a 
    single label to be considered mergable. You probably want this to be set
    above 0.85 at the outside (more likely above 0.95). Set it too low, and you'll 
    introduce external mergers.
  parallel: pass this value to any internal algorithms that support parallel operation
  return_crackle: you can reduce memory usage by returning the outputs as compressed
    CrackleArrays.

  Return value: (filled_labels, hole_labels)
  """
  if not HAS_CRACKLE and return_crackle:
    raise ImportError("crackle not found. Try pip install crackle-codec or set return_crackle=False")

  if np.issubdtype(labels.dtype, bool):
    # This is for speed, not because the below code is incorrect for bool
    # merge_threshold does nothing for a binary image anyway, so ignore it.
    return _fill_binary_image(labels, fix_borders, return_crackle, parallel)

  # Ensure bg 0 gets treated as a connected component
  if labels.flags.writeable:
    labels += 1
    cc_labels, N = cc3d.connected_components(
      labels, 
      return_N=True, 
      connectivity=6,
    )
    labels -= 1
  else:
    cc_labels, N = cc3d.connected_components(
      labels + 1, 
      return_N=True, 
      connectivity=6,
    )

  orig_map = fastremap.component_map(cc_labels, labels)
  orig_map[0] = 0

  slices = [
    np.s_[:,:,0],
    np.s_[:,:,-1],
    np.s_[0,:,:],
    np.s_[-1,:,:],
    np.s_[:,0,:],
    np.s_[:,-1,:],
  ]

  if HAS_CRACKLE:
    orig_cc_labels = crackle.compressa(cc_labels, parallel=parallel)
  elif fix_borders:
    orig_cc_labels = np.copy(cc_labels, order="F")
  else:
    orig_cc_labels = cc_labels

  enclosed_2d = set()
  if fix_borders:
    for slc in slices:
      cc_labels[slc], enclosed_2d_slice = _fill_holes_2d(cc_labels[slc], orig_map)
      enclosed_2d.update(enclosed_2d_slice)

  surface_areas = cc3d.contacts(
    cc_labels,
    connectivity=6,
    surface_area=True,
    anisotropy=tuple(anisotropy),
  )

  connections = _pairs_to_connection_list(surface_areas.keys())

  edge_labels = set()
  bg_edge_labels = set()

  for slc in slices:
    slice_labels = cc_labels[slc]
    uniq = set(fastremap.unique(slice_labels))

    for u in uniq:
      if orig_map[u] == 0:
        bg_edge_labels.add(u)

    edge_labels.update(uniq)

  for bgl in bg_edge_labels:
    for lbl in connections[bgl]:
      if surface_areas[tuple(sorted([bgl, lbl]))] > 0:
        edge_labels.add(lbl)

  del uniq
  del slice_labels

  if HAS_CRACKLE:
    cc_labels = crackle.compressa(cc_labels, parallel=parallel)
    enclosed_2d.difference_update(cc_labels.labels())
  elif fix_borders:
    enclosed_2d.difference_update(fastremap.unique(cc_labels))

  candidate_holes = set(range(1,N+1))
  holes = candidate_holes.difference(edge_labels)
  del candidate_holes

  remap = { i:i for i in range(N+1) }

  visited = np.zeros(N + 1, dtype=bool)
  for hole in list(holes):
    if visited[hole]:
      continue

    parent_label, group = _true_label(
      hole, edge_labels, 
      connections, surface_areas, 
      merge_threshold, visited,
    )
    if hole == parent_label:
      holes.discard(hole)
      holes.difference_update(group)
      continue

    for hole_i in group:
      remap[hole_i] = parent_label

  del connections
  del edge_labels

  remap = { k: orig_map[v] for k,v in remap.items()  }
  holes.update(enclosed_2d)

  if HAS_CRACKLE:
    filled_labels = cc_labels.remap(remap).astype(labels.dtype)
    del cc_labels
    del remap

    hole_labels = orig_cc_labels.mask_except(list(holes))
    hole_labels = hole_labels.remap(orig_map).astype(labels.dtype)

    if return_crackle:
      return (filled_labels, hole_labels)
    else:
      return (filled_labels.numpy(), hole_labels.numpy())
  else:
    filled_labels = fastremap.remap(
      cc_labels, remap, in_place=False,
    ).astype(labels.dtype, copy=False)
    del cc_labels
    del remap

    hole_labels = fastremap.mask_except(orig_cc_labels, list(holes), in_place=True)
    hole_labels = np.where(hole_labels, labels, 0).astype(labels.dtype, copy=False)

    return (filled_labels, hole_labels)
