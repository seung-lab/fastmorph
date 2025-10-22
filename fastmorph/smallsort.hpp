#ifndef __FASTMORPH_SORT_HXX__
#define __FASTMORPH_SORT_HXX__

#include <vector>

namespace fastmorph {

#define CMP_SWAP(x,y) \
	if (labels[x] > labels[y]) {\
		std::swap(labels[x], labels[y]);\
	}


template <typename LABEL>
void sorting_network_2(std::vector<LABEL>& labels) {
	CMP_SWAP(0,1)
}

template <typename LABEL>
void sorting_network_3(std::vector<LABEL>& labels) {
	CMP_SWAP(0,2)
	CMP_SWAP(0,1)
	CMP_SWAP(1,2)
}

/*
https://bertdobbelaere.github.io/sorting_networks.html
Optimal sorting network:
[(0,2),(1,3)]
[(0,1),(2,3)]
[(1,2)]
*/
template <typename LABEL>
void sorting_network_4(std::vector<LABEL>& labels) {
	CMP_SWAP(0,2)
	CMP_SWAP(1,3)
	CMP_SWAP(0,1)
	CMP_SWAP(2,3)
	CMP_SWAP(1,2)
}

/*
https://bertdobbelaere.github.io/sorting_networks.html
Optimal sorting network:
[(0,3),(1,4)]
[(0,2),(1,3)]
[(0,1),(2,4)]
[(1,2),(3,4)]
[(2,3)] 
*/
template <typename LABEL>
void sorting_network_5(std::vector<LABEL>& labels) {
	CMP_SWAP(0,3)
	CMP_SWAP(1,4)
	CMP_SWAP(0,2)
	CMP_SWAP(1,3)
	CMP_SWAP(0,1)
	CMP_SWAP(2,4)
	CMP_SWAP(1,2)
	CMP_SWAP(3,4)
	CMP_SWAP(2,3)
}

/*
https://bertdobbelaere.github.io/sorting_networks.html
Optimal sorting network:
[(0,5),(1,3),(2,4)]
[(1,2),(3,4)]
[(0,3),(2,5)]
[(0,1),(2,3),(4,5)]
[(1,2),(3,4)]
*/
template <typename LABEL>
void sorting_network_6(std::vector<LABEL>& labels) {
	CMP_SWAP(0,5)
	CMP_SWAP(1,3)
	CMP_SWAP(2,4)
	CMP_SWAP(1,2)
	CMP_SWAP(3,4)
	CMP_SWAP(0,3)
	CMP_SWAP(2,5)
	CMP_SWAP(0,1)
	CMP_SWAP(2,3)
	CMP_SWAP(4,5)
	CMP_SWAP(1,2)
	CMP_SWAP(3,4)
}

/*
https://bertdobbelaere.github.io/sorting_networks.html
Optimal sorting network:
[(0,6),(2,3),(4,5)]
[(0,2),(1,4),(3,6)]
[(0,1),(2,5),(3,4)]
[(1,2),(4,6)]
[(2,3),(4,5)]
[(1,2),(3,4),(5,6)]
*/
template <typename LABEL>
void sorting_network_7(std::vector<LABEL>& labels) {
	CMP_SWAP(0,6)
	CMP_SWAP(2,3)
	CMP_SWAP(4,5)
	CMP_SWAP(0,2)
	CMP_SWAP(1,4)
	CMP_SWAP(3,6)
	CMP_SWAP(0,1)
	CMP_SWAP(2,5)
	CMP_SWAP(3,4)
	CMP_SWAP(1,2)
	CMP_SWAP(4,6)
	CMP_SWAP(2,3)
	CMP_SWAP(4,5)
	CMP_SWAP(1,2)
	CMP_SWAP(3,4)
	CMP_SWAP(5,6)
}

/*
https://bertdobbelaere.github.io/sorting_networks.html
Optimal sorting network:
[(0,2),(1,3),(4,6),(5,7)]
[(0,4),(1,5),(2,6),(3,7)]
[(0,1),(2,3),(4,5),(6,7)]
[(2,4),(3,5)]
[(1,4),(3,6)]
[(1,2),(3,4),(5,6)]
*/
template <typename LABEL>
void sorting_network_8(std::vector<LABEL>& labels) {
	CMP_SWAP(0,2)
	CMP_SWAP(1,3)
	CMP_SWAP(4,6)
	CMP_SWAP(5,7)
	CMP_SWAP(0,4)
	CMP_SWAP(1,5)
	CMP_SWAP(2,6)
	CMP_SWAP(3,7)
	CMP_SWAP(0,1)
	CMP_SWAP(2,3)
	CMP_SWAP(4,5)
	CMP_SWAP(6,7)
	CMP_SWAP(2,4)
	CMP_SWAP(3,5)
	CMP_SWAP(1,4)
	CMP_SWAP(3,6)
	CMP_SWAP(1,2)
	CMP_SWAP(3,4)
	CMP_SWAP(5,6)
}

/*
https://bertdobbelaere.github.io/sorting_networks.html
Optimal sorting network:
[(0,3),(1,7),(2,5),(4,8)]
[(0,7),(2,4),(3,8),(5,6)]
[(0,2),(1,3),(4,5),(7,8)]
[(1,4),(3,6),(5,7)]
[(0,1),(2,4),(3,5),(6,8)]
[(2,3),(4,5),(6,7)]
[(1,2),(3,4),(5,6)]
*/
template <typename LABEL>
void sorting_network_9(std::vector<LABEL>& labels) {
	CMP_SWAP(0,3)
	CMP_SWAP(1,7)
	CMP_SWAP(2,5)
	CMP_SWAP(4,8)
	CMP_SWAP(0,7)
	CMP_SWAP(2,4)
	CMP_SWAP(3,8)
	CMP_SWAP(5,6)
	CMP_SWAP(0,2)
	CMP_SWAP(1,3)
	CMP_SWAP(4,5)
	CMP_SWAP(7,8)
	CMP_SWAP(1,4)
	CMP_SWAP(3,6)
	CMP_SWAP(5,7)
	CMP_SWAP(0,1)
	CMP_SWAP(2,4)
	CMP_SWAP(3,5)
	CMP_SWAP(6,8)
	CMP_SWAP(2,3)
	CMP_SWAP(4,5)
	CMP_SWAP(6,7)
	CMP_SWAP(1,2)
	CMP_SWAP(3,4)
	CMP_SWAP(5,6)
}

template <typename LABEL>
void sort(std::vector<LABEL>& labels) {

	switch (labels.size()) {
	case 0:
		break;
	case 1:
		break;
	case 2:
		sorting_network_2(labels);
		break;
	case 3:
		sorting_network_3(labels);
		break;
	case 4:
		sorting_network_4(labels);
		break;
	case 5:
		sorting_network_5(labels);
		break;
	case 6:
		sorting_network_6(labels);
		break;
	case 7:
		sorting_network_7(labels);
		break;
	case 8:
		sorting_network_8(labels);
		break;
	case 9:
		sorting_network_9(labels);
		break;
	default:
		std::sort(labels.begin(), labels.end());
	}
}

#undef CMP_SWAP

};

#endif