#define PYBIND11_DETAILED_ERROR_MESSAGES

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <concepts>
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

template <typename LABEL>
py::array to_numpy(
	LABEL* output,
	const uint64_t sx, const uint64_t sy
) {
	py::capsule capsule(output, [](void* ptr) {
		if (ptr) {
			delete[] static_cast<LABEL*>(ptr);
		}
	});

	uint64_t width = sizeof(LABEL);

	return py::array_t<LABEL>(
		{sx,sy},
		{width, sx * width},
		output,
		capsule
	);
}

#define DISPATCH_TO_TYPES(FUNCTION_MACRO)\
	if (dt.kind() == 'i') {\
		if (width == 1) {\
			FUNCTION_MACRO(int8_t)\
		}\
		else if (width == 2) {\
			FUNCTION_MACRO(int16_t)\
		}\
		else if (width == 4) {\
			FUNCTION_MACRO(int32_t)\
		}\
		else {\
			FUNCTION_MACRO(int64_t)\
		}\
	}\
	else if (dt.kind() == 'b') {\
		FUNCTION_MACRO(uint8_t)\
	}\
	else {\
		if (width == 1) {\
			FUNCTION_MACRO(uint8_t)\
		}\
		else if (width == 2) {\
			FUNCTION_MACRO(uint16_t)\
		}\
		else if (width == 4) {\
			FUNCTION_MACRO(uint32_t)\
		}\
		else {\
			FUNCTION_MACRO(uint64_t)\
		}\
	}


// assumes fortran order
py::array multilabel_dilate(
	const py::array &labels, 
	const bool background_only, 
	const int threads
) {
	py::dtype dt = labels.dtype();
	int width = dt.itemsize();

	const uint64_t sx = labels.shape()[0];
	const uint64_t sy = labels.shape()[1];
	const uint64_t sz = labels.ndim() > 2 
		? labels.shape()[2] 
		: 1;

	void* labels_ptr = const_cast<void*>(labels.data());
	uint8_t* output_ptr = new uint8_t[sx * sy * sz * width]();

	py::array output;

#define DILATE_HELPER_3D(uintx_t)\
	fastmorph::multilabel_dilate(\
			reinterpret_cast<uintx_t*>(labels_ptr),\
			reinterpret_cast<uintx_t*>(output_ptr),\
			sx, sy, sz,\
			background_only, threads\
		);\
		return to_numpy(reinterpret_cast<uintx_t*>(output_ptr), sx, sy, sz);

#define DILATE_HELPER_2D(uintx_t)\
	fastmorph::multilabel_dilate(\
			reinterpret_cast<uintx_t*>(labels_ptr),\
			reinterpret_cast<uintx_t*>(output_ptr),\
			sx, sy,\
			background_only, threads\
		);\
		return to_numpy(reinterpret_cast<uintx_t*>(output_ptr), sx, sy);


	if (labels.ndim() > 2) {
		DISPATCH_TO_TYPES(DILATE_HELPER_3D)
	}
	else {
		DISPATCH_TO_TYPES(DILATE_HELPER_2D)
	}

#undef DILATE_HELPER_3D
#undef DILATE_HELPER_2D
}

// assumes fortran order
py::array multilabel_erode(
	const py::array &labels, 
	const bool erode_border,
	const uint64_t threads
) {
	py::dtype dt = labels.dtype();
	int width = dt.itemsize();

	const uint64_t sx = labels.shape()[0];
	const uint64_t sy = labels.shape()[1];
	const uint64_t sz = labels.ndim() > 2 
		? labels.shape()[2] 
		: 1;

	void* labels_ptr = const_cast<void*>(labels.data());
	uint8_t* output_ptr = new uint8_t[sx * sy * sz * width]();

	py::array output;

#define ERODE_HELPER_3D(uintx_t)\
	fastmorph::multilabel_erode(\
		reinterpret_cast<uintx_t*>(labels_ptr),\
		reinterpret_cast<uintx_t*>(output_ptr),\
		sx, sy, sz,\
		erode_border,\
		threads\
	);\
	return to_numpy(reinterpret_cast<uintx_t*>(output_ptr), sx, sy, sz);

#define ERODE_HELPER_2D(uintx_t)\
	fastmorph::multilabel_erode(\
		reinterpret_cast<uintx_t*>(labels_ptr),\
		reinterpret_cast<uintx_t*>(output_ptr),\
		sx, sy,\
		erode_border,\
		threads\
	);\
	return to_numpy(reinterpret_cast<uintx_t*>(output_ptr), sx, sy);

	if (labels.ndim() > 2) {
		DISPATCH_TO_TYPES(ERODE_HELPER_3D)
	}
	else {
		DISPATCH_TO_TYPES(ERODE_HELPER_2D)
	}

#undef ERODE_HELPER_3D
#undef ERODE_HELPER_2D
}

// assumes fortran order
py::array grey_dilate(const py::array &labels, const uint64_t threads) {
	py::dtype dt = labels.dtype();
	int width = dt.itemsize();

	const uint64_t sx = labels.shape()[0];
	const uint64_t sy = labels.shape()[1];
	const uint64_t sz = labels.ndim() > 2 
		? labels.shape()[2] 
		: 1;

	void* labels_ptr = const_cast<void*>(labels.data());
	uint8_t* output_ptr = new uint8_t[sx * sy * sz * width]();

	py::array output;

#define GREY_DILATE_HELPER_3D(int_t)\
	fastmorph::grey_dilate(\
		reinterpret_cast<int_t*>(labels_ptr),\
		reinterpret_cast<int_t*>(output_ptr),\
		sx, sy, sz,\
		threads\
	);\
	return to_numpy(reinterpret_cast<int_t*>(output_ptr), sx, sy, sz);

#define GREY_DILATE_HELPER_2D(int_t)\
	fastmorph::grey_dilate(\
		reinterpret_cast<int_t*>(labels_ptr),\
		reinterpret_cast<int_t*>(output_ptr),\
		sx, sy,\
		threads\
	);\
	return to_numpy(reinterpret_cast<int_t*>(output_ptr), sx, sy);

	if (labels.ndim() > 2) {
		DISPATCH_TO_TYPES(GREY_DILATE_HELPER_3D)
	}
	else {
		DISPATCH_TO_TYPES(GREY_DILATE_HELPER_2D)
	}

#undef GREY_DILATE_HELPER_3D
#undef GREY_DILATE_HELPER_2D
}

// assumes fortran order
py::array grey_erode(const py::array &labels, const uint64_t threads) {
	py::dtype dt = labels.dtype();
	int width = dt.itemsize();

	const uint64_t sx = labels.shape()[0];
	const uint64_t sy = labels.shape()[1];
	const uint64_t sz = labels.ndim() > 2 
		? labels.shape()[2] 
		: 1;

	void* labels_ptr = const_cast<void*>(labels.data());
	uint8_t* output_ptr = new uint8_t[sx * sy * sz * width]();

	py::array output;

#define GREY_ERODE_HELPER_3D(int_t)\
	fastmorph::grey_erode(\
		reinterpret_cast<int_t*>(labels_ptr),\
		reinterpret_cast<int_t*>(output_ptr),\
		sx, sy, sz,\
		threads\
	);\
	return to_numpy(reinterpret_cast<int_t*>(output_ptr), sx, sy, sz);

#define GREY_ERODE_HELPER_2D(int_t)\
	fastmorph::grey_erode(\
		reinterpret_cast<int_t*>(labels_ptr),\
		reinterpret_cast<int_t*>(output_ptr),\
		sx, sy,\
		threads\
	);\
	return to_numpy(reinterpret_cast<int_t*>(output_ptr), sx, sy);

	if (labels.ndim() > 2) {
		DISPATCH_TO_TYPES(GREY_ERODE_HELPER_3D)
	}
	else {
		DISPATCH_TO_TYPES(GREY_ERODE_HELPER_2D)
	}

#undef GREY_ERODE_HELPER_3D
#undef GREY_ERODE_HELPER_2D
}

#undef DISPATCH_TO_TYPES

// avoid numpy behavior that draws in C order 
// which severely impacts performance
template <typename T>
void draw_with_mask_f_order(
	const py::array_t<T> &output,
	const uint64_t x_start, const uint64_t x_end,
	const uint64_t y_start, const uint64_t y_end,
	const uint64_t z_start, const uint64_t z_end,
	const py::array_t<bool>& binary_image,
	const int64_t value
) {
	const uint64_t sx = output.shape()[0];
	const uint64_t sy = output.shape()[1];

    auto buf = output.request();  // buffer info
    auto ptr = static_cast<T*>(buf.ptr);

    auto binary_buf = binary_image.request();
    const bool* bool_ptr = static_cast<const bool*>(binary_buf.ptr);

    uint64_t i = 0;
	for (uint64_t z = z_start; z < z_end; z++) {
		for (uint64_t y = y_start; y < y_end; y++) {
			for (uint64_t x = x_start; x < x_end; x++, i++) {
				uint64_t loc = x + sx * (y + sy * z);
				if (bool_ptr[i]) {
					ptr[loc] = value;
				}
			}
		}
	}
}

uint64_t count_differences(
	const py::array_t<bool> &img1,
	const py::array_t<bool>& img2
) {
    auto buf1 = img1.request(); // buffer info
    auto ptr1 = static_cast<uint8_t*>(buf1.ptr);

    auto buf2 = img2.request(); // buffer info
    auto ptr2 = static_cast<uint8_t*>(buf2.ptr);

    const uint64_t size = img1.size();
	uint64_t count = 0;

    for (uint64_t i = 0; i < size; i++) {
    	count += (ptr1[i] != ptr2[i]);
    }

    return count;
}

// vcg defintion located in: 
// https://github.com/seung-lab/connected-components-3d/blob/master/cc3d/cc3d_graphs.hpp#L76-L95
template <std::integral LABEL>
void erode_autapses(
	py::array_t<LABEL, py::array::f_style>& py_labels,
	const py::array_t<uint32_t, py::array::f_style>& py_vcg
) {
	if (py_labels.ndim() != 3) {
		throw std::invalid_argument("labels must be a 3d array.");
	}
	if (py_vcg.ndim() != 3) {
		throw std::invalid_argument("vcg must be a 3d array.");
	}
	if (py_labels.size() != py_vcg.size()) {
		throw std::invalid_argument("labels must match vcg in size.");
	}


	LABEL* labels = const_cast<LABEL*>(py_labels.data());
	const uint32_t* vcg = const_cast<uint32_t*>(py_vcg.data());

	const int64_t sx = py_labels.shape(0);
	const int64_t sy = py_labels.shape(1);
	const int64_t sz = py_labels.shape(2);

	// This mapping thanks to DeepSeek's assistance with interpolating
	// the cc3d VCG bit position map
	static constexpr uint32_t vcg_mask[27] = {
		// zi = -1
		1u << 25, // (-x,-y,-z) bit 26
		1u << 17, // ( 0,-y,-z) bit 18
		1u << 24, // (+x,-y,-z) bit 25
		1u << 15, // (-x, 0,-z) bit 16
		1u << 5,  // ( 0, 0,-z) bit 6
		1u << 14, // (+x, 0,-z) bit 15
		1u << 23, // (-x,+y,-z) bit 24
		1u << 16, // ( 0,+y,-z) bit 17
		1u << 22, // (+x,+y,-z) bit 23

		// zi = 0
		1u << 9,  // (-x,-y, 0) bit 10
		1u << 3,  // ( 0,-y, 0) bit 4
		1u << 8,  // (+x,-y, 0) bit 9
		1u << 1,  // (-x, 0, 0) bit 2
		0u,       // ( 0, 0, 0) unused
		1u << 0,  // (+x, 0, 0) bit 1
		1u << 7,  // (-x,+y, 0) bit 8
		1u << 2,  // ( 0,+y, 0) bit 3
		1u << 6,  // (+x,+y, 0) bit 7

		// zi = +1
		1u << 21, // (-x,-y,+z) bit 22
		1u << 13, // ( 0,-y,+z) bit 14
		1u << 20, // (+x,-y,+z) bit 21
		1u << 11, // (-x, 0,+z) bit 12
		1u << 4,  // ( 0, 0,+z) bit 5
		1u << 10, // (+x, 0,+z) bit 11
		1u << 19, // (-x,+y,+z) bit 20
		1u << 12, // ( 0,+y,+z) bit 13
		1u << 18, // (+x,+y,+z) bit 19
	};

	for (int64_t z = 0; z < sz; z++) {
		for (int64_t y = 0; y < sy; y++) {
			for (int64_t x = 0; x < sx; x++) {
				const int64_t loc = x + sx * (y + sy * z);
				const LABEL cur = labels[loc];

				if (cur == 0) {
					continue;
				}

				int64_t zi = z == 0 ? 0 : -1;
				int64_t yi = y == 0 ? 0 : -1;
				int64_t xi = x == 0 ? 0 : -1;

				const int64_t zf = z == sz - 1 ? 0 : 1;
				const int64_t yf = y == sy - 1 ? 0 : 1;
				const int64_t xf = x == sx - 1 ? 0 : 1;

				const uint32_t vc = vcg[loc];

				for (; zi <= zf; zi++) {
					for (; yi <= yf; yi++) {
						for (; xi <= xf; xi++) {
							if (xi == 0 && yi == 0 && zi == 0) {
								continue;
							}

							const int64_t loc_i = loc + (xi + sx * (yi + sy * zi));
							const LABEL neighbor = labels[loc_i];

							if (cur != neighbor) {
								continue;
							}

							const int64_t mask_pos = (xi+1) + 3 * (yi+1) + 9 * (zi+1);

							if ((vc & vcg_mask[mask_pos]) == 0) {
								labels[loc] = 0;
								goto NEXT_LOOP;
							}

						}
					}
				}

				NEXT_LOOP: ;
			}
		}
	}
}

PYBIND11_MODULE(fastmorphops, m) {
	m.doc() = "Accelerated fastmorph functions."; 
	m.def("multilabel_dilate", &multilabel_dilate, "Morphological dilation of a multilabel volume using mode of a 3x3x3 structuring element.");
	m.def("grey_dilate", &grey_dilate, "Morphological dilation of a grayscale volume using max of a 3x3x3 structuring element.");
	m.def("multilabel_erode", &multilabel_erode, "Morphological erosion of a multilabel volume using edge contacts of a 3x3x3 structuring element.");
	m.def("grey_erode", &grey_erode, "Morphological erosion of a grayscale volume using min of a 3x3x3 structuring element.");
	
	// Performance optimization for output[slices][boolean_image] = value

	m.def("draw_with_mask_f_order", &draw_with_mask_f_order<bool>);
	m.def("draw_with_mask_f_order", &draw_with_mask_f_order<uint8_t>);
	m.def("draw_with_mask_f_order", &draw_with_mask_f_order<uint16_t>);
	m.def("draw_with_mask_f_order", &draw_with_mask_f_order<uint32_t>);
	m.def("draw_with_mask_f_order", &draw_with_mask_f_order<uint64_t>);
	m.def("draw_with_mask_f_order", &draw_with_mask_f_order<int8_t>);
	m.def("draw_with_mask_f_order", &draw_with_mask_f_order<int16_t>);
	m.def("draw_with_mask_f_order", &draw_with_mask_f_order<int32_t>);
	m.def("draw_with_mask_f_order", &draw_with_mask_f_order<int64_t>);

	m.def("count_differences", &count_differences);

	m.def("erode_autapses", &erode_autapses<uint8_t>, "Erode self-touches in a volume with a voxel-connectivity graph.");
	m.def("erode_autapses", &erode_autapses<uint16_t>, "Erode self-touches in a volume with a voxel-connectivity graph.");
	m.def("erode_autapses", &erode_autapses<uint32_t>, "Erode self-touches in a volume with a voxel-connectivity graph.");
	m.def("erode_autapses", &erode_autapses<uint64_t>, "Erode self-touches in a volume with a voxel-connectivity graph.");
	m.def("erode_autapses", &erode_autapses<int8_t>, "Erode self-touches in a volume with a voxel-connectivity graph.");
	m.def("erode_autapses", &erode_autapses<int16_t>, "Erode self-touches in a volume with a voxel-connectivity graph.");
	m.def("erode_autapses", &erode_autapses<int32_t>, "Erode self-touches in a volume with a voxel-connectivity graph.");
	m.def("erode_autapses", &erode_autapses<int64_t>, "Erode self-touches in a volume with a voxel-connectivity graph.");
}