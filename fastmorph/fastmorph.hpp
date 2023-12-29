#ifndef __FASTMORPH_HXX__
#define __FASTMORPH_HXX__

#include <vector>
#include <cstdlib>
#include <cmath>
#include "threadpool.h"

namespace fastmorph {


template <typename LABEL>
void dilate(
	LABEL* labels, LABEL* output,
	const uint64_t sx, const uint64_t sy, const uint64_t sz,
	const bool background_only, const uint64_t threads = 1
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

	const uint64_t block_size = 64;

	const uint64_t grid_x = std::max(static_cast<uint64_t>((sx + block_size/2) / block_size), static_cast<uint64_t>(1));
	const uint64_t grid_y = std::max(static_cast<uint64_t>((sy + block_size/2) / block_size), static_cast<uint64_t>(1));
	const uint64_t grid_z = std::max(static_cast<uint64_t>((sz + block_size/2) / block_size), static_cast<uint64_t>(1));

	const int real_threads = std::max(std::min(threads, grid_x * grid_y * grid_z), static_cast<uint64_t>(0));

	ThreadPool pool(real_threads);

	for (uint64_t gz = 0; gz < grid_z; gz++) {
		for (uint64_t gy = 0; gy < grid_y; gy++) {
			for (uint64_t gx = 0; gx < grid_x; gx++) {
				pool.enqueue([=]() {
					process_block(
						gx * block_size, std::min((gx+1) * block_size, sx),
						gy * block_size, std::min((gy+1) * block_size, sy),
						gz * block_size, std::min((gz+1) * block_size, sz)
					);
				});
			}
		}
	}

	pool.join();
}

template <typename LABEL>
void erode(
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
			&& (yi < sy -1 && zi > 0 && labels[loc+sx-sxy] == labels[loc])
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

	const uint64_t block_size = 64;

	const uint64_t grid_x = std::max(static_cast<uint64_t>((sx + block_size/2) / block_size), static_cast<uint64_t>(1));
	const uint64_t grid_y = std::max(static_cast<uint64_t>((sy + block_size/2) / block_size), static_cast<uint64_t>(1));
	const uint64_t grid_z = std::max(static_cast<uint64_t>((sz + block_size/2) / block_size), static_cast<uint64_t>(1));

	const int real_threads = std::max(std::min(threads, grid_x * grid_y * grid_z), static_cast<uint64_t>(0));

	ThreadPool pool(real_threads);

	for (uint64_t gz = 0; gz < grid_z; gz++) {
		for (uint64_t gy = 0; gy < grid_y; gy++) {
			for (uint64_t gx = 0; gx < grid_x; gx++) {
				pool.enqueue([=]() {
					const uint64_t one = 1;
					process_block(
						std::max(one, gx * block_size), std::min((gx+1) * block_size, sx - 1),
						std::max(one, gy * block_size), std::min((gy+1) * block_size, sy - 1),
						std::max(one, gz * block_size), std::min((gz+1) * block_size, sz - 1)
					);
				});
			}
		}
	}

	pool.join();
}

};

#endif