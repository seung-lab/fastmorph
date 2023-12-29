#define PYBIND11_DETAILED_ERROR_MESSAGES

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdlib>
#include <cmath>

#include "fastmorph.hpp"

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

// assumes fortran order
py::array dilate(
	const py::array &labels, 
	const bool background_only, 
	const int threads
) {
	int width = labels.dtype().itemsize();

	const uint64_t sx = labels.shape()[0];
	const uint64_t sy = labels.shape()[1];
	const uint64_t sz = labels.shape()[2];

	void* labels_ptr = const_cast<void*>(labels.data());
	uint8_t* output_ptr = new uint8_t[sx * sy * sz * width]();

	py::array output;

#define DILATE_HELPER(uintx_t)\
	fastmorph::dilate(\
			reinterpret_cast<uintx_t*>(labels_ptr),\
			reinterpret_cast<uintx_t*>(output_ptr),\
			sx, sy, sz,\
			background_only, threads\
		);\
		return to_numpy(reinterpret_cast<uintx_t*>(output_ptr), sx, sy, sz);

	if (width == 1) {
		DILATE_HELPER(uint8_t)
	}
	else if (width == 2) {
		DILATE_HELPER(uint16_t)
	}
	else if (width == 4) {
		DILATE_HELPER(uint32_t)
	}
	else {
		DILATE_HELPER(uint64_t)
	}

#undef DILATE_HELPER
}

// assumes fortran order
py::array erode(const py::array &labels, const uint64_t threads) {
	int width = labels.dtype().itemsize();

	const uint64_t sx = labels.shape()[0];
	const uint64_t sy = labels.shape()[1];
	const uint64_t sz = labels.shape()[2];

	void* labels_ptr = const_cast<void*>(labels.data());
	uint8_t* output_ptr = new uint8_t[sx * sy * sz * width]();

	py::array output;

#define ERODE_HELPER(uintx_t)\
	fastmorph::erode(\
		reinterpret_cast<uintx_t*>(labels_ptr),\
		reinterpret_cast<uintx_t*>(output_ptr),\
		sx, sy, sz,\
		threads\
	);\
	return to_numpy(reinterpret_cast<uintx_t*>(output_ptr), sx, sy, sz);

	if (width == 1) {
		ERODE_HELPER(uint8_t)
	}
	else if (width == 2) {
		ERODE_HELPER(uint16_t)
	}
	else if (width == 4) {
		ERODE_HELPER(uint32_t)
	}
	else {
		ERODE_HELPER(uint64_t)
	}
#undef ERODE_HELPER
}

PYBIND11_MODULE(fastmorphops, m) {
	m.doc() = "Accelerated fastmorph functions."; 
	m.def("dilate", &dilate, "Morphological dilation of a multilabel volume using a 3x3x3 structuring element.");
	m.def("erode", &erode, "Morphological erosion of a multilabel volume using a 3x3x3 structuring element.");
}