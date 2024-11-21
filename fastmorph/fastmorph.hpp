#ifndef __FASTMORPH_HXX__
#define __FASTMORPH_HXX__

#include <vector>
#include <cstdlib>
#include <cmath>
#include <functional>
#include "threadpool.h"

namespace fastmorph {

void parallelize_blocks(
	const std::function<void(
		const uint64_t, const uint64_t, 
		const uint64_t, const uint64_t, 
		const uint64_t, const uint64_t
	)> &process_block,
	const uint64_t sx, const uint64_t sy, const uint64_t sz,
	const uint64_t threads, const uint64_t offset
) {
	const uint64_t block_size = (sz > 1) ? 64 : 512;

	const uint64_t grid_x = std::max(static_cast<uint64_t>((sx + block_size - 1) / block_size), static_cast<uint64_t>(1));
	const uint64_t grid_y = std::max(static_cast<uint64_t>((sy + block_size - 1) / block_size), static_cast<uint64_t>(1));
	const uint64_t grid_z = std::max(static_cast<uint64_t>((sz + block_size - 1) / block_size), static_cast<uint64_t>(1));

	const int real_threads = std::max(std::min(threads, grid_x * grid_y * grid_z), static_cast<uint64_t>(0));

	ThreadPool pool(real_threads);

	for (uint64_t gz = 0; gz < grid_z; gz++) {
		for (uint64_t gy = 0; gy < grid_y; gy++) {
			for (uint64_t gx = 0; gx < grid_x; gx++) {
				pool.enqueue([=]() {
					process_block(
						std::max(offset, gx * block_size), std::min((gx+1) * block_size, sx - offset),
						std::max(offset, gy * block_size), std::min((gy+1) * block_size, sy - offset),
						std::max(offset, gz * block_size), std::min((gz+1) * block_size, sz - offset)
					);
				});
			}
		}
	}

	pool.join();
}


template <typename LABEL>
void multilabel_dilate(
	LABEL* labels, LABEL* output,
	const uint64_t sx, const uint64_t sy, const uint64_t sz,
	const bool background_only, const uint64_t threads
) {

	// assume a 3x3x3 stencil with all voxels on
	const uint64_t sxy = sx * sy;

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

	auto fill_partial_stencil_fast_fn = [&](
		const uint64_t xi, const uint64_t yi, const uint64_t zi, 
		std::vector<LABEL> &square
	) {
		square.clear();

		if (xi < 0 || xi >= sx) {
			return;
		}

		const uint64_t loc = xi + sx * (yi + sy * zi);

		if (zi < sz - 1 && labels[loc+sxy] != 0) {
			square.push_back(labels[loc+sxy]);
		}
		if (yi > 0 && zi < sz - 1 && labels[loc-sx+sxy] != 0) {
			square.push_back(labels[loc-sx+sxy]);
		}
		if (yi < sy - 1 && zi < sz - 1 && labels[loc+sx+sxy] != 0) {
			square.push_back(labels[loc+sx+sxy]);
		}
	};

	auto process_block = [&](
		const uint64_t xs, const uint64_t xe, 
		const uint64_t ys, const uint64_t ye, 
		const uint64_t zs, const uint64_t ze
	){
		// 3x3 sets of labels, as index advances 
		// right is leading edge, middle becomes left, 
		// left gets deleted
		std::vector<LABEL> left, middle, right, tmp;
		left.reserve(9);
		middle.reserve(9);
		right.reserve(9);
		tmp.reserve(9);

		auto advance_stencil = [&](uint64_t x, uint64_t y, uint64_t z) {
			tmp = std::move(left);
			left = std::move(middle);
			middle = std::move(right);
			right = std::move(tmp);
			fill_partial_stencil_fn(x+2,y,z,right);
		};

		int stale_stencil = 3;


		std::vector<LABEL> neighbors;
		neighbors.reserve(27);

		for (uint64_t z = zs; z < ze; z++) {
			for (uint64_t y = ys; y < ye; y++) {
				stale_stencil = 3;
				for (uint64_t x = xs; x < xe; x++) {
					uint64_t loc = x + sx * (y + sy * z);

					if (background_only && labels[loc] != 0) {
						output[loc] = labels[loc];
						stale_stencil++;
						continue;
					}

					if (z > zs && output[loc-sxy] == 0) {
						if (stale_stencil == 1) {
							tmp = std::move(left);
							left = std::move(middle);
							middle = std::move(right);
							right = std::move(tmp);
							fill_partial_stencil_fast_fn(x+1,y,z,right);
						}
						else if (stale_stencil == 2) {
							std::swap(left, right);
							fill_partial_stencil_fast_fn(x,y,z,middle);
							fill_partial_stencil_fast_fn(x+1,y,z,right);					
						}
						else if (stale_stencil >= 3) {
							fill_partial_stencil_fast_fn(x-1,y,z,left);
							fill_partial_stencil_fast_fn(x,y,z,middle);
							fill_partial_stencil_fast_fn(x+1,y,z,right);
						}
					}
					else {
						if (stale_stencil == 1) {
							advance_stencil(x-1,y,z);
						}
						else if (stale_stencil == 2) {
							std::swap(left, right);
							fill_partial_stencil_fn(x,y,z,middle);
							fill_partial_stencil_fn(x+1,y,z,right);					
						}
						else if (stale_stencil >= 3) {
							fill_partial_stencil_fn(x-1,y,z,left);
							fill_partial_stencil_fn(x,y,z,middle);
							fill_partial_stencil_fn(x+1,y,z,right);
						}
					}

					stale_stencil = 0;

					if (left.size() + middle.size() + right.size() == 0) {
						stale_stencil = 1;
						continue;
					} 

					std::sort(middle.begin(), middle.end());
					std::sort(right.begin(), right.end());

					if ((right.size() + middle.size() >= 14)
						&& right[0] == right[right.size() - 1]
						&& middle[0] == middle[middle.size() - 1]
						&& right[0] == middle[0]) {

						output[loc] = right[0];
						if (x < sx - 1) {
							output[loc+1] = right[0];
							stale_stencil = 2;
							x++;
						}
						else {
							stale_stencil = 1;
						}
						continue;
					}

					neighbors.clear();

					neighbors.insert(neighbors.end(), left.begin(), left.end());
					neighbors.insert(neighbors.end(), middle.begin(), middle.end());
					neighbors.insert(neighbors.end(), right.begin(), right.end());

					std::sort(neighbors.begin(), neighbors.end());

					int size = neighbors.size();

					// the middle and right will be the next
					// left and middle and will dominate the
					// right so we can skip some calculation.
					if (neighbors[0] == neighbors[size - 1]) {
						output[loc] = neighbors[0];
						if (size >= 23 && x < sx - 1) {
							output[loc+1] = neighbors[0];
							stale_stencil = 2;
							x++;
						}
						else {
							stale_stencil = 1;
						}
						continue;
					}

					LABEL mode_label = neighbors[0];
					int ct = 1;
					int max_ct = 1;
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

					if (ct > max_ct) {
						mode_label = neighbors[size - 1];
					}

					output[loc] = mode_label;

					if (ct >= 23 && x < sx - 1) {
						output[loc+1] = mode_label;
						stale_stencil = 2;
						x++;
						continue;
					}

					stale_stencil = 1;
				}
			}
		}
	};

	parallelize_blocks(
		std::function<void(
			const uint64_t,const uint64_t,const uint64_t,
			const uint64_t,const uint64_t,const uint64_t
		)>(process_block), 
		sx, sy, sz, threads, /*offset=*/0
	);
}

template <typename LABEL>
void multilabel_dilate(
	LABEL* labels, LABEL* output,
	const uint64_t sx, const uint64_t sy,
	const bool background_only, const uint64_t threads
) {

	// assume a 3x3 stencil with all voxels on
	auto fill_partial_stencil_fn = [&](
		const uint64_t xi, const uint64_t yi,
		std::vector<LABEL> &column
	) {
		column.clear();

		if (xi < 0 || xi >= sx) {
			return;
		}

		const uint64_t loc = xi + sx * yi;

		if (labels[loc] != 0) {
			column.push_back(labels[loc]);
		}
		if (yi > 0 && labels[loc-sx] != 0) {
			column.push_back(labels[loc-sx]);
		}
		if (yi < sy - 1 && labels[loc+sx] != 0) {
			column.push_back(labels[loc+sx]);
		}
	};

	auto process_block = [&](
		const uint64_t xs, const uint64_t xe, 
		const uint64_t ys, const uint64_t ye, 
		const uint64_t zs, const uint64_t ze
	){
		// sets of labels representing a column of 3, as index advances 
		// right is leading edge, middle becomes left, 
		// left gets deleted
		std::vector<LABEL> left, middle, right, tmp;
		left.reserve(3);
		middle.reserve(3);
		right.reserve(3);
		tmp.reserve(3);

		auto advance_stencil = [&](uint64_t x, uint64_t y) {
			tmp = std::move(left);
			left = std::move(middle);
			middle = std::move(right);
			right = std::move(tmp);
			fill_partial_stencil_fn(x+2,y,right);
		};

		int stale_stencil = 3;

		std::vector<LABEL> neighbors;
		neighbors.reserve(9);

		for (uint64_t y = ys; y < ye; y++) {
			stale_stencil = 3;
			for (uint64_t x = xs; x < xe; x++) {
				uint64_t loc = x + sx * y;

				if (background_only && labels[loc] != 0) {
					output[loc] = labels[loc];
					stale_stencil++;
					continue;
				}

				if (stale_stencil == 1) {
					advance_stencil(x-1,y);
				}
				else if (stale_stencil == 2) {
					std::swap(left, right);
					fill_partial_stencil_fn(x,y,middle);
					fill_partial_stencil_fn(x+1,y,right);					
				}
				else if (stale_stencil >= 3) {
					fill_partial_stencil_fn(x-1,y,left);
					fill_partial_stencil_fn(x,y,middle);
					fill_partial_stencil_fn(x+1,y,right);
				}

				stale_stencil = 0;

				if (left.size() + middle.size() + right.size() == 0) {
					stale_stencil = 1;
					continue;
				} 

				std::sort(middle.begin(), middle.end());
				std::sort(right.begin(), right.end());

				if ((right.size() + middle.size() >= 5)
					&& right[0] == right[right.size() - 1]
					&& middle[0] == middle[middle.size() - 1]
					&& right[0] == middle[0]) {

					output[loc] = right[0];
					if (x < sx - 1) {
						output[loc+1] = right[0];
						stale_stencil = 2;
						x++;
					}
					else {
						stale_stencil = 1;
					}
					continue;
				}

				neighbors.clear();

				neighbors.insert(neighbors.end(), left.begin(), left.end());
				neighbors.insert(neighbors.end(), middle.begin(), middle.end());
				neighbors.insert(neighbors.end(), right.begin(), right.end());

				std::sort(neighbors.begin(), neighbors.end());

				int size = neighbors.size();

				LABEL mode_label = neighbors[0];
				int ct = 1;
				int max_ct = 1;
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

				if (ct > max_ct) {
					mode_label = neighbors[size - 1];
				}

				output[loc] = mode_label;

				if (ct >= 8 && x < sx - 1) {
					output[loc+1] = mode_label;
					stale_stencil = 2;
					x++;
					continue;
				}

				stale_stencil = 1;
			}
		}
	};

	parallelize_blocks(
		std::function<void(
			const uint64_t,const uint64_t,const uint64_t,
			const uint64_t,const uint64_t,const uint64_t
		)>(process_block), 
		sx, sy, /*sz=*/1, threads, /*offset=*/0
	);
}

template <typename LABEL>
void multilabel_erode(
	LABEL* labels, LABEL* output,
	const uint64_t sx, const uint64_t sy, const uint64_t sz,
	const uint64_t threads
) {

	// assume a 3x3x3 stencil with all voxels on
	const uint64_t sxy = sx * sy;

	auto is_pure = [&](
		const uint64_t xi, const uint64_t yi, const uint64_t zi
	) {
		const uint64_t loc = xi + sx * (yi + sy * zi);

		return static_cast<LABEL>(labels[loc] * (
			(xi >= 0 && xi < sx)
			&& (labels[loc] != 0)
			&& (yi > 0 && labels[loc-sx] == labels[loc])
			&& (yi < sy - 1 && labels[loc+sx] == labels[loc])
			&& (zi > 0 && labels[loc-sxy] == labels[loc])
			&& (zi < sz - 1 && labels[loc+sxy] == labels[loc])
			&& (yi > 0 && zi > 0 && labels[loc-sx-sxy] == labels[loc])
			&& (yi < sy -1 && zi > 0 && labels[loc+sx-sxy] == labels[loc])
			&& (yi > 0 && zi < sz - 1 && labels[loc-sx+sxy] == labels[loc])
			&& (yi < sy - 1 && zi < sz - 1 && labels[loc+sx+sxy] == labels[loc])
		));
	};

	auto is_pure_fast_z = [&](
		const uint64_t xi, const uint64_t yi, const uint64_t zi
	) {
		const uint64_t loc = xi + sx * (yi + sy * zi);

		return static_cast<LABEL>(labels[loc] * (
			(xi >= 0 && xi < sx)
		 && (zi < sz - 1 && labels[loc+sxy] == labels[loc])
		 && (yi > 0 && zi < sz - 1 && labels[loc-sx+sxy] == labels[loc])
		 && (yi < sy - 1 && zi < sz - 1 && labels[loc+sx+sxy] == labels[loc])
		));
	};

	auto is_pure_fast_y = [&](
		const uint64_t xi, const uint64_t yi, const uint64_t zi
	) {

		const uint64_t loc = xi + sx * (yi + sy * zi);

		return static_cast<LABEL>(labels[loc] * (
			    (xi >= 0 && xi < sx)
			&& (yi < sy - 1 && labels[loc+sx] == labels[loc])
			&& (yi < sy - 1 && zi > 0 && labels[loc+sx-sxy] == labels[loc])
			&& (yi < sy - 1 && zi < sz - 1 && labels[loc+sx+sxy] == labels[loc]))
		);
	};

	auto process_block = [&](
		const uint64_t xs, const uint64_t xe, 
		const uint64_t ys, const uint64_t ye, 
		const uint64_t zs, const uint64_t ze
	){
		LABEL pure_left = 0;
		LABEL pure_middle = 0;
		LABEL pure_right = 0;

		int stale_stencil = 3;

#define FILL_STENCIL(is_pure_fn) \
	if (stale_stencil == 1) {\
		pure_left = pure_middle;\
		pure_middle = pure_right;\
		pure_right = is_pure_fn(x+1,y,z);\
	}\
	else if (stale_stencil >= 3) {\
		pure_right = is_pure_fn(x+1,y,z);\
		if (!pure_right) {\
			x += 2;\
			stale_stencil = 3;\
			continue;\
		}\
		pure_middle = is_pure_fn(x,y,z);\
		if (!pure_middle) {\
			x++;\
			stale_stencil = 2;\
			continue;\
		}\
		pure_left = is_pure_fn(x-1,y,z);\
	}\
	else if (stale_stencil == 2) {\
		pure_left = pure_right;\
		pure_right = is_pure_fn(x+1,y,z);\
		if (!pure_right) {\
			x += 2;\
			stale_stencil = 3;\
			continue;\
		}\
		pure_middle = is_pure_fn(x,y,z);\
	}

		for (uint64_t z = zs; z < ze; z++) {
			for (uint64_t y = ys; y < ye; y++) {
				stale_stencil = 3;
				for (uint64_t x = xs; x < xe; x++) {
					uint64_t loc = x + sx * (y + sy * z);

					if (labels[loc] == 0) {
						x++;
						stale_stencil += 2;
						continue;
					}

					if (z > zs && output[loc-sxy] == labels[loc]) {
						FILL_STENCIL(is_pure_fast_z)
					}
					else if (y > ys && output[loc-sx] == labels[loc]) {
						FILL_STENCIL(is_pure_fast_y)
					}
					else {
						FILL_STENCIL(is_pure)
					}
					
					stale_stencil = 0;

					if (!pure_right) {
						x += 2;
						stale_stencil = 3;
						continue;
					}
					else if (!pure_middle) {
						x++;
						stale_stencil = 2;
						continue;
					}
					else if (pure_left == pure_middle && pure_middle == pure_right) {
						output[loc] = labels[loc];
					}

					stale_stencil = 1;
				}
			}
		}
	};

#undef FILL_STENCIL

	parallelize_blocks(
		std::function<void(
			const uint64_t,const uint64_t,const uint64_t,
			const uint64_t,const uint64_t,const uint64_t
		)>(process_block), 
		sx, sy, sz, threads, /*offset=*/1
	);
}

template <typename LABEL>
void multilabel_erode(
	LABEL* labels, LABEL* output,
	const uint64_t sx, const uint64_t sy,
	const uint64_t threads
) {

	// assume a 3x3 stencil with all voxels on

	auto is_pure = [&](const uint64_t xi, const uint64_t yi) {
		const uint64_t loc = xi + sx * yi;

		return static_cast<LABEL>(labels[loc] * (
			(xi >= 0 && xi < sx)
			&& (labels[loc] != 0)
			&& (yi > 0 && labels[loc-sx] == labels[loc])
			&& (yi < sy - 1 && labels[loc+sx] == labels[loc])
		));
	};

	auto process_block = [&](
		const uint64_t xs, const uint64_t xe, 
		const uint64_t ys, const uint64_t ye, 
		const uint64_t zs, const uint64_t ze
	){
		LABEL pure_left = 0;
		LABEL pure_middle = 0;
		LABEL pure_right = 0;

		int stale_stencil = 3;
	
		for (uint64_t y = ys; y < ye; y++) {
			stale_stencil = 3;
			for (uint64_t x = xs; x < xe; x++) {
				uint64_t loc = x + sx * y;

				if (labels[loc] == 0) {
					x++;
					stale_stencil += 2;
					continue;
				}

				if (stale_stencil == 1) {
					pure_left = pure_middle;
					pure_middle = pure_right;
					pure_right = is_pure(x+1,y);
				}
				else if (stale_stencil >= 3) {
					pure_right = is_pure(x+1,y);
					if (!pure_right) {
						x += 2;
						stale_stencil = 3;
						continue;
					}
					pure_middle = is_pure(x,y);
					if (!pure_middle) {
						x++;
						stale_stencil = 2;
						continue;
					}
					pure_left = is_pure(x-1,y);
				}
				else if (stale_stencil == 2) {
					pure_left = pure_right;
					pure_right = is_pure(x+1,y);
					if (!pure_right) {
						x += 2;
						stale_stencil = 3;
						continue;
					}
					pure_middle = is_pure(x,y);
				}
				
				stale_stencil = 0;

				if (!pure_right) {
					x += 2;
					stale_stencil = 3;
					continue;
				}
				else if (!pure_middle) {
					x++;
					stale_stencil = 2;
					continue;
				}
				else if (pure_left == pure_middle && pure_middle == pure_right) {
					output[loc] = labels[loc];
				}

				stale_stencil = 1;
			}
		}
	};

	parallelize_blocks(
		std::function<void(
			const uint64_t,const uint64_t,const uint64_t,
			const uint64_t,const uint64_t,const uint64_t
		)>(process_block), 
		sx, sy, /*sz=*/1, threads, /*offset=*/1
	);
}

template <typename LABEL>
void grey_dilate(
	LABEL* labels, LABEL* output,
	const uint64_t sx, const uint64_t sy, const uint64_t sz,
	const uint64_t threads
) {
	// assume a 3x3x3 stencil with all voxels on
	const uint64_t sxy = sx * sy;
	constexpr LABEL MAX_LABEL = std::numeric_limits<LABEL>::max();

	auto get_max = [&](
		const uint64_t xi, const uint64_t yi, const uint64_t zi
	) {
		const uint64_t loc = xi + sx * (yi + sy * zi);

		LABEL maxval = std::numeric_limits<LABEL>::min();

		if (xi < 0 || xi >= sx) {
			return maxval;
		}

		maxval = std::max(maxval, labels[loc]);

		if (yi > 0) {
			 maxval = std::max(maxval, labels[loc-sx]);
		}
		if (yi < sy - 1) {
			maxval = std::max(maxval, labels[loc+sx]);
		}
		if (zi > 0) {
			maxval = std::max(maxval, labels[loc-sxy]);
		}
		if (zi < sz - 1) {
			maxval = std::max(maxval, labels[loc+sxy]);
		}
		if (yi > 0 && zi > 0) {
			maxval = std::max(maxval, labels[loc-sx-sxy]);
		}
		if (yi < sy -1 && zi > 0) {
			maxval = std::max(maxval, labels[loc+sx-sxy]);
		}
		if (yi > 0 && zi < sz - 1) {
			maxval = std::max(maxval, labels[loc-sx+sxy]);
		}
		if (yi < sy - 1 && zi < sz - 1) {
			maxval = std::max(maxval, labels[loc+sx+sxy]);
		}

		return maxval;
	};

	auto process_block = [&](
		const uint64_t xs, const uint64_t xe, 
		const uint64_t ys, const uint64_t ye, 
		const uint64_t zs, const uint64_t ze
	){
		LABEL max_left = MAX_LABEL;
		LABEL max_middle = MAX_LABEL;
		LABEL max_right = MAX_LABEL;

		int stale_stencil = 3;

		for (uint64_t z = zs; z < ze; z++) {
			for (uint64_t y = ys; y < ye; y++) {
				stale_stencil = 3;
				for (uint64_t x = xs; x < xe; x++) {
					uint64_t loc = x + sx * (y + sy * z);

					if (labels[loc] == MAX_LABEL) {
						x++;
						stale_stencil += 2;
						continue;
					}

					if (stale_stencil == 1) {
						max_left = max_middle;
						max_middle = max_right;
						max_right = get_max(x+1,y,z);
					}
					else if (stale_stencil >= 3) {
						max_right = get_max(x+1,y,z);
						if (max_right == MAX_LABEL) {
							x += 2;
							stale_stencil = 3;
							continue;
						}
						max_middle = get_max(x,y,z);
						if (max_middle == MAX_LABEL) {
							x++;
							stale_stencil = 2;
							continue;
						}
						max_left = get_max(x-1,y,z);
					}
					else if (stale_stencil == 2) {
						max_left = max_right;
						max_right = get_max(x+1,y,z);
						if (max_right == MAX_LABEL) {
							x += 2;
							stale_stencil = 3;
							continue;
						}
						max_middle = get_max(x,y,z);
					}					

					stale_stencil = 0;

					if (max_right == MAX_LABEL) {
						x += 2;
						stale_stencil = 3;
						continue;
					}
					else if (max_middle == MAX_LABEL) {
						x++;
						stale_stencil = 2;
						continue;
					}
					
					output[loc] = std::max(
						std::max(max_left, max_middle), 
						max_right
					);

					stale_stencil = 1;
				}
			}
		}
	};

	parallelize_blocks(
		std::function<void(
			const uint64_t,const uint64_t,const uint64_t,
			const uint64_t,const uint64_t,const uint64_t
		)>(process_block), 
		sx, sy, sz, threads, /*offset=*/0
	);
}

template <typename LABEL>
void grey_dilate(
	LABEL* labels, LABEL* output,
	const uint64_t sx, const uint64_t sy,
	const uint64_t threads
) {
	// assume a 3x3 stencil with all voxels on
	constexpr LABEL MAX_LABEL = std::numeric_limits<LABEL>::max();

	auto get_max = [&](const uint64_t xi, const uint64_t yi) {
		const uint64_t loc = xi + sx * yi;

		LABEL maxval = std::numeric_limits<LABEL>::min();

		if (xi < 0 || xi >= sx) {
			return maxval;
		}

		maxval = std::max(maxval, labels[loc]);

		if (yi > 0) {
			 maxval = std::max(maxval, labels[loc-sx]);
		}
		if (yi < sy - 1) {
			maxval = std::max(maxval, labels[loc+sx]);
		}

		return maxval;
	};

	auto process_block = [&](
		const uint64_t xs, const uint64_t xe, 
		const uint64_t ys, const uint64_t ye, 
		const uint64_t zs, const uint64_t ze
	){
		LABEL max_left = MAX_LABEL;
		LABEL max_middle = MAX_LABEL;
		LABEL max_right = MAX_LABEL;

		int stale_stencil = 3;

		for (uint64_t y = ys; y < ye; y++) {
			stale_stencil = 3;
			for (uint64_t x = xs; x < xe; x++) {
				uint64_t loc = x + sx * y;

				if (labels[loc] == MAX_LABEL) {
					x++;
					stale_stencil += 2;
					continue;
				}

				if (stale_stencil == 1) {
					max_left = max_middle;
					max_middle = max_right;
					max_right = get_max(x+1,y);
				}
				else if (stale_stencil >= 3) {
					max_right = get_max(x+1,y);
					if (max_right == MAX_LABEL) {
						x += 2;
						stale_stencil = 3;
						continue;
					}
					max_middle = get_max(x,y);
					if (max_middle == MAX_LABEL) {
						x++;
						stale_stencil = 2;
						continue;
					}
					max_left = get_max(x-1,y);
				}
				else if (stale_stencil == 2) {
					max_left = max_right;
					max_right = get_max(x+1,y);
					if (max_right == MAX_LABEL) {
						x += 2;
						stale_stencil = 3;
						continue;
					}
					max_middle = get_max(x,y);
				}					

				stale_stencil = 0;

				if (max_right == MAX_LABEL) {
					x += 2;
					stale_stencil = 3;
					continue;
				}
				else if (max_middle == MAX_LABEL) {
					x++;
					stale_stencil = 2;
					continue;
				}
				
				output[loc] = std::max(
					std::max(max_left, max_middle), 
					max_right
				);

				stale_stencil = 1;
			}
		}
	};

	parallelize_blocks(
		std::function<void(
			const uint64_t,const uint64_t,const uint64_t,
			const uint64_t,const uint64_t,const uint64_t
		)>(process_block), 
		sx, sy, /*sz=*/1, threads, /*offset=*/0
	);
}


template <typename LABEL>
void grey_erode(
	LABEL* labels, LABEL* output,
	const uint64_t sx, const uint64_t sy, const uint64_t sz,
	const uint64_t threads
) {

	// assume a 3x3x3 stencil with all voxels on
	const uint64_t sxy = sx * sy;
	constexpr LABEL MIN_LABEL = std::numeric_limits<LABEL>::min();

	auto get_min = [&](
		const uint64_t xi, const uint64_t yi, const uint64_t zi
	) {
		const uint64_t loc = xi + sx * (yi + sy * zi);

		LABEL minval = std::numeric_limits<LABEL>::max();

		if (xi < 0 || xi >= sx) {
			return minval;
		}

		minval = std::min(minval, labels[loc]);

		if (yi > 0) {
			 minval = std::min(minval, labels[loc-sx]);
		}
		if (yi < sy - 1) {
			minval = std::min(minval, labels[loc+sx]);
		}
		if (zi > 0) {
			minval = std::min(minval, labels[loc-sxy]);
		}
		if (zi < sz - 1) {
			minval = std::min(minval, labels[loc+sxy]);
		}
		if (yi > 0 && zi > 0) {
			minval = std::min(minval, labels[loc-sx-sxy]);
		}
		if (yi < sy -1 && zi > 0) {
			minval = std::min(minval, labels[loc+sx-sxy]);
		}
		if (yi > 0 && zi < sz - 1) {
			minval = std::min(minval, labels[loc-sx+sxy]);
		}
		if (yi < sy - 1 && zi < sz - 1) {
			minval = std::min(minval, labels[loc+sx+sxy]);
		}

		return minval;
	};

	auto process_block = [&](
		const uint64_t xs, const uint64_t xe, 
		const uint64_t ys, const uint64_t ye, 
		const uint64_t zs, const uint64_t ze
	){
		LABEL min_left = MIN_LABEL;
		LABEL min_middle = MIN_LABEL;
		LABEL min_right = MIN_LABEL;

		int stale_stencil = 3;

		for (uint64_t z = zs; z < ze; z++) {
			for (uint64_t y = ys; y < ye; y++) {
				stale_stencil = 3;
				for (uint64_t x = xs; x < xe; x++) {
					uint64_t loc = x + sx * (y + sy * z);

					if (labels[loc] == MIN_LABEL) {
						x++;
						stale_stencil += 2;
						continue;
					}

					if (stale_stencil == 1) {
						min_left = min_middle;
						min_middle = min_right;
						min_right = get_min(x+1,y,z);
					}
					else if (stale_stencil >= 3) {
						min_right = get_min(x+1,y,z);
						if (min_right == MIN_LABEL) {
							x += 2;
							stale_stencil = 3;
							continue;
						}
						min_middle = get_min(x,y,z);
						if (min_middle == MIN_LABEL) {
							x++;
							stale_stencil = 2;
							continue;
						}
						min_left = get_min(x-1,y,z);
					}
					else if (stale_stencil == 2) {
						min_left = min_right;
						min_right = get_min(x+1,y,z);
						if (min_right == MIN_LABEL) {
							x += 2;
							stale_stencil = 3;
							continue;
						}
						min_middle = get_min(x,y,z);
					}					

					stale_stencil = 0;

					if (min_right == MIN_LABEL) {
						x += 2;
						stale_stencil = 3;
						continue;
					}
					else if (min_middle == MIN_LABEL) {
						x++;
						stale_stencil = 2;
						continue;
					}
					
					output[loc] = std::min(
						std::min(min_left, min_middle), 
						min_right
					);

					stale_stencil = 1;
				}
			}
		}
	};

	parallelize_blocks(
		std::function<void(
			const uint64_t,const uint64_t,const uint64_t,
			const uint64_t,const uint64_t,const uint64_t
		)>(process_block), 
		sx, sy, sz, threads, /*offset=*/0
	);
}

template <typename LABEL>
void grey_erode(
	LABEL* labels, LABEL* output,
	const uint64_t sx, const uint64_t sy,
	const uint64_t threads
) {

	// assume a 3x3 stencil with all voxels on
	constexpr LABEL MIN_LABEL = std::numeric_limits<LABEL>::min();

	auto get_min = [&](const uint64_t xi, const uint64_t yi) {
		const uint64_t loc = xi + sx * yi;

		LABEL minval = std::numeric_limits<LABEL>::max();

		if (xi < 0 || xi >= sx) {
			return minval;
		}

		minval = std::min(minval, labels[loc]);

		if (yi > 0) {
			 minval = std::min(minval, labels[loc-sx]);
		}
		if (yi < sy - 1) {
			minval = std::min(minval, labels[loc+sx]);
		}

		return minval;
	};

	auto process_block = [&](
		const uint64_t xs, const uint64_t xe, 
		const uint64_t ys, const uint64_t ye, 
		const uint64_t zs, const uint64_t ze
	){
		LABEL min_left = MIN_LABEL;
		LABEL min_middle = MIN_LABEL;
		LABEL min_right = MIN_LABEL;

		int stale_stencil = 3;

		for (uint64_t y = ys; y < ye; y++) {
			stale_stencil = 3;
			for (uint64_t x = xs; x < xe; x++) {
				uint64_t loc = x + sx * y;

				if (labels[loc] == MIN_LABEL) {
					x++;
					stale_stencil += 2;
					continue;
				}

				if (stale_stencil == 1) {
					min_left = min_middle;
					min_middle = min_right;
					min_right = get_min(x+1,y);
				}
				else if (stale_stencil >= 3) {
					min_right = get_min(x+1,y);
					if (min_right == MIN_LABEL) {
						x += 2;
						stale_stencil = 3;
						continue;
					}
					min_middle = get_min(x,y);
					if (min_middle == MIN_LABEL) {
						x++;
						stale_stencil = 2;
						continue;
					}
					min_left = get_min(x-1,y);
				}
				else if (stale_stencil == 2) {
					min_left = min_right;
					min_right = get_min(x+1,y);
					if (min_right == MIN_LABEL) {
						x += 2;
						stale_stencil = 3;
						continue;
					}
					min_middle = get_min(x,y);
				}					

				stale_stencil = 0;

				if (min_right == MIN_LABEL) {
					x += 2;
					stale_stencil = 3;
					continue;
				}
				else if (min_middle == MIN_LABEL) {
					x++;
					stale_stencil = 2;
					continue;
				}
				
				output[loc] = std::min(
					std::min(min_left, min_middle), 
					min_right
				);

				stale_stencil = 1;
			}
		}
	};

	parallelize_blocks(
		std::function<void(
			const uint64_t,const uint64_t,const uint64_t,
			const uint64_t,const uint64_t,const uint64_t
		)>(process_block), 
		sx, sy, /*sz=*/1, threads, /*offset=*/0
	);
}



};

#endif