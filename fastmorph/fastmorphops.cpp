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
py::array multilabel_dilate(
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
	fastmorph::multilabel_dilate(\
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
py::array multilabel_erode(const py::array &labels, const uint64_t threads) {
	int width = labels.dtype().itemsize();

	const uint64_t sx = labels.shape()[0];
	const uint64_t sy = labels.shape()[1];
	const uint64_t sz = labels.shape()[2];

	void* labels_ptr = const_cast<void*>(labels.data());
	uint8_t* output_ptr = new uint8_t[sx * sy * sz * width]();

	py::array output;

#define ERODE_HELPER(uintx_t)\
	fastmorph::multilabel_erode(\
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

// assumes fortran order
py::array grey_dilate(const py::array &labels, const uint64_t threads) {
	py::dtype dt = labels.dtype();
	int width = dt.itemsize();

	const uint64_t sx = labels.shape()[0];
	const uint64_t sy = labels.shape()[1];
	const uint64_t sz = labels.shape()[2];

	void* labels_ptr = const_cast<void*>(labels.data());
	uint8_t* output_ptr = new uint8_t[sx * sy * sz * width]();

	py::array output;

#define GREY_DILATE_HELPER(int_t)\
	fastmorph::grey_dilate(\
		reinterpret_cast<int_t*>(labels_ptr),\
		reinterpret_cast<int_t*>(output_ptr),\
		sx, sy, sz,\
		threads\
	);\
	return to_numpy(reinterpret_cast<int_t*>(output_ptr), sx, sy, sz);

	if (dt.kind() == 'i') {
		if (width == 1) {
			GREY_DILATE_HELPER(int8_t)
		}
		else if (width == 2) {
			GREY_DILATE_HELPER(int16_t)
		}
		else if (width == 4) {
			GREY_DILATE_HELPER(int32_t)
		}
		else {
			GREY_DILATE_HELPER(int64_t)
		}
	}
	else if (dt.kind() == 'b') {
		GREY_DILATE_HELPER(uint8_t)
	}
	else {
		if (width == 1) {
			GREY_DILATE_HELPER(uint8_t)
		}
		else if (width == 2) {
			GREY_DILATE_HELPER(uint16_t)
		}
		else if (width == 4) {
			GREY_DILATE_HELPER(uint32_t)
		}
		else {
			GREY_DILATE_HELPER(uint64_t)
		}
	}
#undef GREY_DILATE_HELPER
}

// assumes fortran order
py::array grey_erode(const py::array &labels, const uint64_t threads) {
	py::dtype dt = labels.dtype();
	int width = dt.itemsize();

	const uint64_t sx = labels.shape()[0];
	const uint64_t sy = labels.shape()[1];
	const uint64_t sz = labels.shape()[2];

	void* labels_ptr = const_cast<void*>(labels.data());
	uint8_t* output_ptr = new uint8_t[sx * sy * sz * width]();

	py::array output;

#define GREY_ERODE_HELPER(int_t)\
	fastmorph::grey_erode(\
		reinterpret_cast<int_t*>(labels_ptr),\
		reinterpret_cast<int_t*>(output_ptr),\
		sx, sy, sz,\
		threads\
	);\
	return to_numpy(reinterpret_cast<int_t*>(output_ptr), sx, sy, sz);

	if (dt.kind() == 'i') {
		if (width == 1) {
			GREY_ERODE_HELPER(int8_t)
		}
		else if (width == 2) {
			GREY_ERODE_HELPER(int16_t)
		}
		else if (width == 4) {
			GREY_ERODE_HELPER(int32_t)
		}
		else {
			GREY_ERODE_HELPER(int64_t)
		}
	}
	else if (dt.kind() == 'b') {
		GREY_ERODE_HELPER(uint8_t)
	}
	else {
		if (width == 1) {
			GREY_ERODE_HELPER(uint8_t)
		}
		else if (width == 2) {
			GREY_ERODE_HELPER(uint16_t)
		}
		else if (width == 4) {
			GREY_ERODE_HELPER(uint32_t)
		}
		else {
			GREY_ERODE_HELPER(uint64_t)
		}
	}
#undef GREY_ERODE_HELPER
}

PYBIND11_MODULE(fastmorphops, m) {
	m.doc() = "Accelerated fastmorph functions."; 
	m.def("multilabel_dilate", &multilabel_dilate, "Morphological dilation of a multilabel volume using mode of a 3x3x3 structuring element.");
	m.def("grey_dilate", &grey_dilate, "Morphological dilation of a grayscale volume using max of a 3x3x3 structuring element.");
	m.def("multilabel_erode", &multilabel_erode, "Morphological erosion of a multilabel volume using edge contacts of a 3x3x3 structuring element.");
	m.def("grey_erode", &grey_erode, "Morphological erosion of a grayscale volume using min of a 3x3x3 structuring element.");
}