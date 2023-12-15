#define PYBIND11_DETAILED_ERROR_MESSAGES

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>
#include <cstdlib>

namespace py = pybind11;


template <typename LABEL>
py::array to_numpy(
	LABEL* output,
	const uint64_t sx, const uint64_t sy, const uint64_t sz
) {
	py::capsule capsule(output, [](void* ptr) {
		if (ptr) {
			delete[] static_cast<LABEL*>(ptr);
		}
	});

	uint64_t width = sizeof(LABEL);

	return py::array_t<LABEL>(
		{sx,sy,sz},
		{width, sx * width, sx * sy * width},
		output,
		capsule
	);
}

template <typename LABEL>
py::array dilate_helper(
	LABEL* labels, LABEL* output,
	const uint64_t sx, const uint64_t sy, const uint64_t sz,
	const bool background_only
) {

	// assume a 3x3x3 stencil with all voxels on
	const uint64_t sxy = sx * sy;

	// 3x3 sets of labels, as index advances 
	// right is leading edge, middle becomes left, 
	// left gets deleted
	std::vector<LABEL> left, middle, right;

	auto fill_partial_stencil_fn = [&](
		const uint64_t xi, const uint64_t yi, const uint64_t zi, 
		std::vector<LABEL> &square
	) {
		square.clear();

		if (xi < 0 || xi >= sx) {
			return;
		}

		const uint64_t loc = xi + sx * (yi + sy * zi);

		if (labels[loc] != 0) {
			square.push_back(labels[loc]);
		}

		if (yi > 0 && labels[loc-sx] != 0) {
			square.push_back(labels[loc-sx]);
		}
		if (yi < sy - 1 && labels[loc+sx] != 0) {
			square.push_back(labels[loc+sx]);
		}
		if (zi > 0 && labels[loc-sxy] != 0) {
			square.push_back(labels[loc-sxy]);
		}
		if (zi < sz - 1 && labels[loc+sxy] != 0) {
			square.push_back(labels[loc+sxy]);
		}
		if (yi > 0 && zi > 0 && labels[loc-sx-sxy] != 0) {
			square.push_back(labels[loc-sx-sxy]);
		}
		if (yi < sy -1 && zi > 0 && labels[loc+sx-sxy] != 0) {
			square.push_back(labels[loc+sx-sxy]);
		}
		if (yi > 0 && zi < sz - 1 && labels[loc-sx+sxy] != 0) {
			square.push_back(labels[loc-sx+sxy]);
		}
		if (yi < sy - 1 && zi < sz - 1 && labels[loc+sx+sxy] != 0) {
			square.push_back(labels[loc+sx+sxy]);
		}
	};

	auto advance_stencil = [&](uint64_t x, uint64_t y, uint64_t z) {
		left = middle;
		middle = right;
		fill_partial_stencil_fn(x+2,y,z,right);
	};

	bool stale_stencil = true;

	for (uint64_t z = 0; z < sz; z++) {
		for (uint64_t y = 0; y < sy; y++) {
			stale_stencil = true;
			for (uint64_t x = 0; x < sx; x++) {
				uint64_t loc = x + sx * (y + sy * z);

				if (background_only && labels[loc] != 0) {
					output[loc] = labels[loc];
					stale_stencil = true;
					continue;
				}

				if (stale_stencil) {
					fill_partial_stencil_fn(x-1,y,z,left);
					fill_partial_stencil_fn(x,y,z,middle);
					fill_partial_stencil_fn(x+1,y,z,right);
					stale_stencil = false;
				}

				if (left.size() + middle.size() + right.size() == 0) {
					advance_stencil(x,y,z);
					continue;
				} 

				std::vector<LABEL> neighbors;
				neighbors.reserve(26);

				neighbors.insert(neighbors.end(), left.begin(), left.end());
				neighbors.insert(neighbors.end(), middle.begin(), middle.end());
				neighbors.insert(neighbors.end(), right.begin(), right.end());

				std::sort(neighbors.begin(), neighbors.end());

				LABEL mode_label = neighbors[0];
				int ct = 1;
				int max_ct = 1;
				int size = neighbors.size();
				for (int i = 1; i < size; i++) {
					if (neighbors[i] != neighbors[i-1]) {
						if (ct > max_ct) {
							mode_label = neighbors[i-1];
							max_ct = ct;
						}
						ct = 1;

						if (size - i < max_ct) {
							break;
						}
					}
					else {
						ct++;
					}
				}

				output[loc] = mode_label;

				advance_stencil(x,y,z);
			}
		}
	}

	return to_numpy(output, sx, sy, sz);
}

// assumes fortran order
py::array dilate(const py::array &labels, const bool background_only) {
	int width = labels.dtype().itemsize();

	const uint64_t sx = labels.shape()[0];
	const uint64_t sy = labels.shape()[1];
	const uint64_t sz = labels.shape()[2];

	void* labels_ptr = const_cast<void*>(labels.data());
	uint8_t* output_ptr = new uint8_t[sx * sy * sz * width]();

	py::array output;

	if (width == 1) {
		output = dilate_helper(
			reinterpret_cast<uint8_t*>(labels_ptr),
			reinterpret_cast<uint8_t*>(output_ptr),
			sx, sy, sz,
			background_only
		);
	}
	else if (width == 2) {
		output = dilate_helper(
			reinterpret_cast<uint16_t*>(labels_ptr),
			reinterpret_cast<uint16_t*>(output_ptr),
			sx, sy, sz,
			background_only
		);
	}
	else if (width == 4) {
		output = dilate_helper(
			reinterpret_cast<uint32_t*>(labels_ptr),
			reinterpret_cast<uint32_t*>(output_ptr),
			sx, sy, sz,
			background_only
		);
	}
	else if (width == 8) {
		output = dilate_helper(
			reinterpret_cast<uint64_t*>(labels_ptr),
			reinterpret_cast<uint64_t*>(output_ptr),
			sx, sy, sz,
			background_only
		);
	}

	return output;
}

template <typename LABEL>
py::array erode_helper(
	LABEL* labels, LABEL* output,
	const uint64_t sx, const uint64_t sy, const uint64_t sz
) {

	// assume a 3x3x3 stencil with all voxels on
	const uint64_t sxy = sx * sy;

	// 3x3 sets of labels, as index advances 
	// right is leading edge, middle becomes left, 
	// left gets deleted
	std::vector<LABEL> left, middle, right;
	bool pure_left = false;
	bool pure_middle = false;
	bool pure_right = false;

	auto fill_partial_stencil_fn = [&](
		const uint64_t xi, const uint64_t yi, const uint64_t zi, 
		std::vector<LABEL> &square
	) {
		square.clear();

		if (xi < 0 || xi >= sx) {
			return;
		}

		const uint64_t loc = xi + sx * (yi + sy * zi);

		if (labels[loc] != 0) {
			square.push_back(labels[loc]);
		}

		if (yi > 0 && labels[loc-sx] != 0) {
			square.push_back(labels[loc-sx]);
		}
		if (yi < sy - 1 && labels[loc+sx] != 0) {
			square.push_back(labels[loc+sx]);
		}
		if (zi > 0 && labels[loc-sxy] != 0) {
			square.push_back(labels[loc-sxy]);
		}
		if (zi < sz - 1 && labels[loc+sxy] != 0) {
			square.push_back(labels[loc+sxy]);
		}
		if (yi > 0 && zi > 0 && labels[loc-sx-sxy] != 0) {
			square.push_back(labels[loc-sx-sxy]);
		}
		if (yi < sy -1 && zi > 0 && labels[loc+sx-sxy] != 0) {
			square.push_back(labels[loc+sx-sxy]);
		}
		if (yi > 0 && zi < sz - 1 && labels[loc-sx+sxy] != 0) {
			square.push_back(labels[loc-sx+sxy]);
		}
		if (yi < sy - 1 && zi < sz - 1 && labels[loc+sx+sxy] != 0) {
			square.push_back(labels[loc+sx+sxy]);
		}
	};

	auto is_pure = [](std::vector<LABEL> &square){
		if (square.size() < 9) {
			return false;
		}

		for (int i = 1; i < 9; i++) {
			if (square[i] != square[i-1]) {
				return false;
			}
		}

		return true;
	};

	auto advance_stencil = [&](uint64_t x, uint64_t y, uint64_t z) {
		left = middle;
		middle = right;
		pure_left = pure_middle;
		pure_middle = pure_right;
		fill_partial_stencil_fn(x+2,y,z,right);
		pure_right = is_pure(right);
	};

	bool stale_stencil = true;

	for (uint64_t z = 0; z < sz; z++) {
		for (uint64_t y = 0; y < sy; y++) {
			stale_stencil = true;
			for (uint64_t x = 0; x < sx; x++) {
				uint64_t loc = x + sx * (y + sy * z);

				if (labels[loc] == 0) {
					stale_stencil = true;
					continue;
				}

				if (stale_stencil) {
					fill_partial_stencil_fn(x-1,y,z,left);
					fill_partial_stencil_fn(x,y,z,middle);
					fill_partial_stencil_fn(x+1,y,z,right);
					pure_left = is_pure(left);
					pure_middle = is_pure(middle);
					pure_right = is_pure(right);
					stale_stencil = false;
				}

				if (pure_left && pure_middle && pure_right) {
					output[loc] = labels[loc];
				}

				advance_stencil(x,y,z);
			}
		}
	}

	return to_numpy(output, sx, sy, sz);
}

// assumes fortran order
py::array erode(const py::array &labels) {
	int width = labels.dtype().itemsize();

	const uint64_t sx = labels.shape()[0];
	const uint64_t sy = labels.shape()[1];
	const uint64_t sz = labels.shape()[2];

	void* labels_ptr = const_cast<void*>(labels.data());
	uint8_t* output_ptr = new uint8_t[sx * sy * sz * width]();

	py::array output;

	if (width == 1) {
		output = erode_helper(
			reinterpret_cast<uint8_t*>(labels_ptr),
			reinterpret_cast<uint8_t*>(output_ptr),
			sx, sy, sz
		);
	}
	else if (width == 2) {
		output = erode_helper(
			reinterpret_cast<uint16_t*>(labels_ptr),
			reinterpret_cast<uint16_t*>(output_ptr),
			sx, sy, sz
		);
	}
	else if (width == 4) {
		output = erode_helper(
			reinterpret_cast<uint32_t*>(labels_ptr),
			reinterpret_cast<uint32_t*>(output_ptr),
			sx, sy, sz
		);
	}
	else if (width == 8) {
		output = erode_helper(
			reinterpret_cast<uint64_t*>(labels_ptr),
			reinterpret_cast<uint64_t*>(output_ptr),
			sx, sy, sz
		);
	}

	return output;
}

PYBIND11_MODULE(fastmorphops, m) {
	m.doc() = "Accelerated fastmorph functions."; 
	m.def("dilate", &dilate, "Morphological dilation of a multilabel volume using a 3x3x3 structuring element.");
	m.def("erode", &erode, "Morphological erosion of a multilabel volume using a 3x3x3 structuring element.");
}