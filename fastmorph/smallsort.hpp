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

/*
https://bertdobbelaere.github.io/sorting_networks.html
Optimal sorting network:
[(0,8),(1,9),(2,7),(3,5),(4,6)]
[(0,2),(1,4),(5,8),(7,9)]
[(0,3),(2,4),(5,7),(6,9)]
[(0,1),(3,6),(8,9)]
[(1,5),(2,3),(4,8),(6,7)]
[(1,2),(3,5),(4,6),(7,8)]
[(2,3),(4,5),(6,7)]
[(3,4),(5,6)]
*/
template <typename LABEL>
void sorting_network_10(std::vector<LABEL>& labels) {
	CMP_SWAP(0,8)
	CMP_SWAP(1,9)
	CMP_SWAP(2,7)
	CMP_SWAP(3,5)
	CMP_SWAP(4,6)
	CMP_SWAP(0,2)
	CMP_SWAP(1,4)
	CMP_SWAP(5,8)
	CMP_SWAP(7,9)
	CMP_SWAP(0,3)
	CMP_SWAP(2,4)
	CMP_SWAP(5,7)
	CMP_SWAP(6,9)
	CMP_SWAP(0,1)
	CMP_SWAP(3,6)
	CMP_SWAP(8,9)
	CMP_SWAP(1,5)
	CMP_SWAP(2,3)
	CMP_SWAP(4,8)
	CMP_SWAP(6,7)
	CMP_SWAP(1,2)
	CMP_SWAP(3,5)
	CMP_SWAP(4,6)
	CMP_SWAP(7,8)
	CMP_SWAP(2,3)
	CMP_SWAP(4,5)
	CMP_SWAP(6,7)
	CMP_SWAP(3,4)
	CMP_SWAP(5,6)
}


/*
https://bertdobbelaere.github.io/sorting_networks.html
Optimal sorting network:
[(0,9),(1,6),(2,4),(3,7),(5,8)]
[(0,1),(3,5),(4,10),(6,9),(7,8)]
[(1,3),(2,5),(4,7),(8,10)]
[(0,4),(1,2),(3,7),(5,9),(6,8)]
[(0,1),(2,6),(4,5),(7,8),(9,10)]
[(2,4),(3,6),(5,7),(8,9)]
[(1,2),(3,4),(5,6),(7,8)]
[(2,3),(4,5),(6,7)]
*/
template <typename LABEL>
void sorting_network_11(std::vector<LABEL>& labels) {
	CMP_SWAP(0,9)
	CMP_SWAP(1,6)
	CMP_SWAP(2,4)
	CMP_SWAP(3,7)
	CMP_SWAP(5,8)
	CMP_SWAP(0,1)
	CMP_SWAP(3,5)
	CMP_SWAP(4,10)
	CMP_SWAP(6,9)
	CMP_SWAP(7,8)
	CMP_SWAP(1,3)
	CMP_SWAP(2,5)
	CMP_SWAP(4,7)
	CMP_SWAP(8,10)
	CMP_SWAP(0,4)
	CMP_SWAP(1,2)
	CMP_SWAP(3,7)
	CMP_SWAP(5,9)
	CMP_SWAP(6,8)
	CMP_SWAP(0,1)
	CMP_SWAP(2,6)
	CMP_SWAP(4,5)
	CMP_SWAP(7,8)
	CMP_SWAP(9,10)
	CMP_SWAP(2,4)
	CMP_SWAP(3,6)
	CMP_SWAP(5,7)
	CMP_SWAP(8,9)
	CMP_SWAP(1,2)
	CMP_SWAP(3,4)
	CMP_SWAP(5,6)
	CMP_SWAP(7,8)
	CMP_SWAP(2,3)
	CMP_SWAP(4,5)
	CMP_SWAP(6,7)
}

/*
https://bertdobbelaere.github.io/sorting_networks.html
Optimal sorting network:
[(0,8),(1,7),(2,6),(3,11),(4,10),(5,9)]
[(0,1),(2,5),(3,4),(6,9),(7,8),(10,11)]
[(0,2),(1,6),(5,10),(9,11)]
[(0,3),(1,2),(4,6),(5,7),(8,11),(9,10)]
[(1,4),(3,5),(6,8),(7,10)]
[(1,3),(2,5),(6,9),(8,10)]
[(2,3),(4,5),(6,7),(8,9)]
[(4,6),(5,7)]
[(3,4),(5,6),(7,8)]
*/
template <typename LABEL>
void sorting_network_12(std::vector<LABEL>& labels) {
	CMP_SWAP(0,8)
	CMP_SWAP(1,7)
	CMP_SWAP(2,6)
	CMP_SWAP(3,11)
	CMP_SWAP(4,10)
	CMP_SWAP(5,9)
	CMP_SWAP(0,1)
	CMP_SWAP(2,5)
	CMP_SWAP(3,4)
	CMP_SWAP(6,9)
	CMP_SWAP(7,8)
	CMP_SWAP(10,11)
	CMP_SWAP(0,2)
	CMP_SWAP(1,6)
	CMP_SWAP(5,10)
	CMP_SWAP(9,11)
	CMP_SWAP(0,3)
	CMP_SWAP(1,2)
	CMP_SWAP(4,6)
	CMP_SWAP(5,7)
	CMP_SWAP(8,11)
	CMP_SWAP(9,10)
	CMP_SWAP(1,4)
	CMP_SWAP(3,5)
	CMP_SWAP(6,8)
	CMP_SWAP(7,10)
	CMP_SWAP(1,3)
	CMP_SWAP(2,5)
	CMP_SWAP(6,9)
	CMP_SWAP(8,10)
	CMP_SWAP(2,3)
	CMP_SWAP(4,5)
	CMP_SWAP(6,7)
	CMP_SWAP(8,9)
	CMP_SWAP(4,6)
	CMP_SWAP(5,7)
	CMP_SWAP(3,4)
	CMP_SWAP(5,6)
	CMP_SWAP(7,8)
}

/*
https://bertdobbelaere.github.io/sorting_networks.html
Optimal sorting network:
[(0,1),(2,3),(4,5),(6,7),(8,9),(10,11),(12,13),(14,15),(16,17)]
[(0,2),(1,3),(4,12),(5,13),(6,8),(9,11),(14,16),(15,17)]
[(0,14),(1,16),(2,15),(3,17)]
[(0,6),(1,10),(2,9),(7,16),(8,15),(11,17)]
[(1,4),(3,9),(5,7),(8,14),(10,12),(13,16)]
[(0,1),(2,5),(3,13),(4,14),(7,9),(8,10),(12,15),(16,17)]
[(1,2),(3,5),(4,6),(11,13),(12,14),(15,16)]
[(4,8),(5,12),(6,10),(7,11),(9,13)]
[(1,4),(2,8),(3,6),(5,7),(9,15),(10,12),(11,14),(13,16)]
[(2,4),(5,8),(6,10),(7,11),(9,12),(13,15)]
[(3,5),(6,8),(7,10),(9,11),(12,14)]
[(3,4),(5,6),(7,8),(9,10),(11,12),(13,14)]
*/
template <typename LABEL>
void sorting_network_18(std::vector<LABEL>& labels) {
	CMP_SWAP(0,1)
	CMP_SWAP(2,3)
	CMP_SWAP(4,5)
	CMP_SWAP(6,7)
	CMP_SWAP(8,9)
	CMP_SWAP(10,11)
	CMP_SWAP(12,13)
	CMP_SWAP(14,15)
	CMP_SWAP(16,17)
	CMP_SWAP(0,2)
	CMP_SWAP(1,3)
	CMP_SWAP(4,12)
	CMP_SWAP(5,13)
	CMP_SWAP(6,8)
	CMP_SWAP(9,11)
	CMP_SWAP(14,16)
	CMP_SWAP(15,17)
	CMP_SWAP(0,14)
	CMP_SWAP(1,16)
	CMP_SWAP(2,15)
	CMP_SWAP(3,17)
	CMP_SWAP(0,6)
	CMP_SWAP(1,10)
	CMP_SWAP(2,9)
	CMP_SWAP(7,16)
	CMP_SWAP(8,15)
	CMP_SWAP(11,17)
	CMP_SWAP(1,4)
	CMP_SWAP(3,9)
	CMP_SWAP(5,7)
	CMP_SWAP(8,14)
	CMP_SWAP(10,12)
	CMP_SWAP(13,16)
	CMP_SWAP(0,1)
	CMP_SWAP(2,5)
	CMP_SWAP(3,13)
	CMP_SWAP(4,14)
	CMP_SWAP(7,9)
	CMP_SWAP(8,10)
	CMP_SWAP(12,15)
	CMP_SWAP(16,17)
	CMP_SWAP(1,2)
	CMP_SWAP(3,5)
	CMP_SWAP(4,6)
	CMP_SWAP(11,13)
	CMP_SWAP(12,14)
	CMP_SWAP(15,16)
	CMP_SWAP(4,8)
	CMP_SWAP(5,12)
	CMP_SWAP(6,10)
	CMP_SWAP(7,11)
	CMP_SWAP(9,13)
	CMP_SWAP(1,4)
	CMP_SWAP(2,8)
	CMP_SWAP(3,6)
	CMP_SWAP(5,7)
	CMP_SWAP(9,15)
	CMP_SWAP(10,12)
	CMP_SWAP(11,14)
	CMP_SWAP(13,16)
	CMP_SWAP(2,4)
	CMP_SWAP(5,8)
	CMP_SWAP(6,10)
	CMP_SWAP(7,11)
	CMP_SWAP(9,12)
	CMP_SWAP(13,15)
	CMP_SWAP(3,5)
	CMP_SWAP(6,8)
	CMP_SWAP(7,10)
	CMP_SWAP(9,11)
	CMP_SWAP(12,14)
	CMP_SWAP(3,4)
	CMP_SWAP(5,6)
	CMP_SWAP(7,8)
	CMP_SWAP(9,10)
	CMP_SWAP(11,12)
	CMP_SWAP(13,14)
}


/*
https://bertdobbelaere.github.io/sorting_networks.html
Optimal sorting network:
[(0,1),(2,3),(4,5),(6,7),(8,9),(10,11),(12,13),(14,15),(16,17),(18,19),(20,21),(22,23),(24,25)]
[(0,2),(1,3),(4,6),(5,7),(8,10),(9,11),(14,16),(15,17),(18,20),(19,21),(22,24),(23,25)]
[(0,4),(1,6),(2,5),(3,7),(8,14),(9,16),(10,15),(11,17),(18,22),(19,24),(20,23),(21,25)]
[(0,18),(1,19),(2,20),(3,21),(4,22),(5,23),(6,24),(7,25),(9,12),(13,16)]
[(3,11),(8,9),(10,13),(12,15),(14,22),(16,17)]
[(0,8),(1,9),(2,14),(6,12),(7,15),(10,18),(11,23),(13,19),(16,24),(17,25)]
[(1,2),(3,18),(4,8),(7,22),(17,21),(23,24)]
[(3,14),(4,10),(5,18),(7,20),(8,13),(11,22),(12,17),(15,21)]
[(1,4),(5,6),(7,9),(8,10),(15,17),(16,18),(19,20),(21,24)]
[(2,5),(3,10),(6,14),(9,13),(11,19),(12,16),(15,22),(20,23)]
[(2,8),(5,7),(6,9),(11,12),(13,14),(16,19),(17,23),(18,20)]
[(2,4),(3,5),(6,11),(7,10),(9,16),(12,13),(14,19),(15,18),(20,22),(21,23)]
[(3,4),(5,8),(6,7),(9,11),(10,12),(13,15),(14,16),(17,20),(18,19),(21,22)]
[(5,6),(7,8),(9,10),(11,12),(13,14),(15,16),(17,18),(19,20)]
[(4,5),(6,7),(8,9),(10,11),(12,13),(14,15),(16,17),(18,19),(20,21)]
*/
template <typename LABEL>
void sorting_network_26(std::vector<LABEL>& labels) {
	CMP_SWAP(0,1)
	CMP_SWAP(2,3)
	CMP_SWAP(4,5)
	CMP_SWAP(6,7)
	CMP_SWAP(8,9)
	CMP_SWAP(10,11)
	CMP_SWAP(12,13)
	CMP_SWAP(14,15)
	CMP_SWAP(16,17)
	CMP_SWAP(18,19)
	CMP_SWAP(20,21)
	CMP_SWAP(22,23)
	CMP_SWAP(24,25)
	CMP_SWAP(0,2)
	CMP_SWAP(1,3)
	CMP_SWAP(4,6)
	CMP_SWAP(5,7)
	CMP_SWAP(8,10)
	CMP_SWAP(9,11)
	CMP_SWAP(14,16)
	CMP_SWAP(15,17)
	CMP_SWAP(18,20)
	CMP_SWAP(19,21)
	CMP_SWAP(22,24)
	CMP_SWAP(23,25)
	CMP_SWAP(0,4)
	CMP_SWAP(1,6)
	CMP_SWAP(2,5)
	CMP_SWAP(3,7)
	CMP_SWAP(8,14)
	CMP_SWAP(9,16)
	CMP_SWAP(10,15)
	CMP_SWAP(11,17)
	CMP_SWAP(18,22)
	CMP_SWAP(19,24)
	CMP_SWAP(20,23)
	CMP_SWAP(21,25)
	CMP_SWAP(0,18)
	CMP_SWAP(1,19)
	CMP_SWAP(2,20)
	CMP_SWAP(3,21)
	CMP_SWAP(4,22)
	CMP_SWAP(5,23)
	CMP_SWAP(6,24)
	CMP_SWAP(7,25)
	CMP_SWAP(9,12)
	CMP_SWAP(13,16)
	CMP_SWAP(3,11)
	CMP_SWAP(8,9)
	CMP_SWAP(10,13)
	CMP_SWAP(12,15)
	CMP_SWAP(14,22)
	CMP_SWAP(16,17)
	CMP_SWAP(0,8)
	CMP_SWAP(1,9)
	CMP_SWAP(2,14)
	CMP_SWAP(6,12)
	CMP_SWAP(7,15)
	CMP_SWAP(10,18)
	CMP_SWAP(11,23)
	CMP_SWAP(13,19)
	CMP_SWAP(16,24)
	CMP_SWAP(17,25)
	CMP_SWAP(1,2)
	CMP_SWAP(3,18)
	CMP_SWAP(4,8)
	CMP_SWAP(7,22)
	CMP_SWAP(17,21)
	CMP_SWAP(23,24)
	CMP_SWAP(3,14)
	CMP_SWAP(4,10)
	CMP_SWAP(5,18)
	CMP_SWAP(7,20)
	CMP_SWAP(8,13)
	CMP_SWAP(11,22)
	CMP_SWAP(12,17)
	CMP_SWAP(15,21)
	CMP_SWAP(1,4)
	CMP_SWAP(5,6)
	CMP_SWAP(7,9)
	CMP_SWAP(8,10)
	CMP_SWAP(15,17)
	CMP_SWAP(16,18)
	CMP_SWAP(19,20)
	CMP_SWAP(21,24)
	CMP_SWAP(2,5)
	CMP_SWAP(3,10)
	CMP_SWAP(6,14)
	CMP_SWAP(9,13)
	CMP_SWAP(11,19)
	CMP_SWAP(12,16)
	CMP_SWAP(15,22)
	CMP_SWAP(20,23)
	CMP_SWAP(2,8)
	CMP_SWAP(5,7)
	CMP_SWAP(6,9)
	CMP_SWAP(11,12)
	CMP_SWAP(13,14)
	CMP_SWAP(16,19)
	CMP_SWAP(17,23)
	CMP_SWAP(18,20)
	CMP_SWAP(2,4)
	CMP_SWAP(3,5)
	CMP_SWAP(6,11)
	CMP_SWAP(7,10)
	CMP_SWAP(9,16)
	CMP_SWAP(12,13)
	CMP_SWAP(14,19)
	CMP_SWAP(15,18)
	CMP_SWAP(20,22)
	CMP_SWAP(21,23)
	CMP_SWAP(3,4)
	CMP_SWAP(5,8)
	CMP_SWAP(6,7)
	CMP_SWAP(9,11)
	CMP_SWAP(10,12)
	CMP_SWAP(13,15)
	CMP_SWAP(14,16)
	CMP_SWAP(17,20)
	CMP_SWAP(18,19)
	CMP_SWAP(21,22)
	CMP_SWAP(5,6)
	CMP_SWAP(7,8)
	CMP_SWAP(9,10)
	CMP_SWAP(11,12)
	CMP_SWAP(13,14)
	CMP_SWAP(15,16)
	CMP_SWAP(17,18)
	CMP_SWAP(19,20)
	CMP_SWAP(4,5)
	CMP_SWAP(6,7)
	CMP_SWAP(8,9)
	CMP_SWAP(10,11)
	CMP_SWAP(12,13)
	CMP_SWAP(14,15)
	CMP_SWAP(16,17)
	CMP_SWAP(18,19)
	CMP_SWAP(20,21)
}

/*
https://bertdobbelaere.github.io/sorting_networks.html
Optimal sorting network:
[(0,1),(2,3),(4,5),(6,7),(8,9),(10,11),(12,13),(14,15),(16,17),(18,19),(20,21),(22,23),(24,25)]
[(0,2),(1,3),(4,6),(5,7),(8,10),(9,11),(14,16),(15,17),(18,20),(19,21),(22,24),(23,25)]
[(0,4),(1,5),(2,6),(3,7),(8,14),(9,16),(10,13),(11,17),(12,15),(18,22),(19,23),(20,24),(21,25)]
[(0,18),(1,19),(2,20),(3,21),(4,22),(5,23),(6,24),(7,25),(8,12),(10,14),(11,15),(13,17)]
[(1,18),(2,10),(3,20),(4,8),(5,22),(6,14),(7,24),(9,12),(11,19),(13,16),(15,23),(17,21)]
[(0,4),(1,9),(3,13),(5,15),(6,18),(7,19),(8,11),(10,20),(12,22),(14,17),(16,24),(21,25)]
[(2,4),(3,11),(5,9),(10,12),(13,15),(14,22),(16,20),(21,23)]
[(1,4),(3,8),(6,10),(7,13),(9,11),(12,18),(14,16),(15,19),(17,22),(21,24)]
[(1,2),(3,6),(4,5),(7,12),(8,10),(9,14),(11,16),(13,18),(15,17),(19,22),(20,21),(23,24)]
[(2,3),(4,6),(5,10),(7,9),(11,13),(12,14),(15,20),(16,18),(19,21),(22,23)]
[(3,4),(5,6),(7,8),(9,10),(11,12),(13,14),(15,16),(17,18),(19,20),(21,22)]
[(5,7),(6,8),(9,11),(10,12),(13,15),(14,16),(17,19),(18,20)]
[(4,5),(6,7),(8,9),(10,11),(12,13),(14,15),(16,17),(18,19),(20,21)]
*/
template <typename LABEL>
void sorting_network_27(std::vector<LABEL>& labels) {
	CMP_SWAP(0,1)
	CMP_SWAP(2,3)
	CMP_SWAP(4,5)
	CMP_SWAP(6,7)
	CMP_SWAP(8,9)
	CMP_SWAP(10,11)
	CMP_SWAP(12,13)
	CMP_SWAP(14,15)
	CMP_SWAP(16,17)
	CMP_SWAP(18,19)
	CMP_SWAP(20,21)
	CMP_SWAP(22,23)
	CMP_SWAP(24,25)
	CMP_SWAP(0,2)
	CMP_SWAP(1,3)
	CMP_SWAP(4,6)
	CMP_SWAP(5,7)
	CMP_SWAP(8,10)
	CMP_SWAP(9,11)
	CMP_SWAP(14,16)
	CMP_SWAP(15,17)
	CMP_SWAP(18,20)
	CMP_SWAP(19,21)
	CMP_SWAP(22,24)
	CMP_SWAP(23,25)
	CMP_SWAP(0,4)
	CMP_SWAP(1,5)
	CMP_SWAP(2,6)
	CMP_SWAP(3,7)
	CMP_SWAP(8,14)
	CMP_SWAP(9,16)
	CMP_SWAP(10,13)
	CMP_SWAP(11,17)
	CMP_SWAP(12,15)
	CMP_SWAP(18,22)
	CMP_SWAP(19,23)
	CMP_SWAP(20,24)
	CMP_SWAP(21,25)
	CMP_SWAP(0,18)
	CMP_SWAP(1,19)
	CMP_SWAP(2,20)
	CMP_SWAP(3,21)
	CMP_SWAP(4,22)
	CMP_SWAP(5,23)
	CMP_SWAP(6,24)
	CMP_SWAP(7,25)
	CMP_SWAP(8,12)
	CMP_SWAP(10,14)
	CMP_SWAP(11,15)
	CMP_SWAP(13,17)
	CMP_SWAP(1,18)
	CMP_SWAP(2,10)
	CMP_SWAP(3,20)
	CMP_SWAP(4,8)
	CMP_SWAP(5,22)
	CMP_SWAP(6,14)
	CMP_SWAP(7,24)
	CMP_SWAP(9,12)
	CMP_SWAP(11,19)
	CMP_SWAP(13,16)
	CMP_SWAP(15,23)
	CMP_SWAP(17,21)
	CMP_SWAP(0,4)
	CMP_SWAP(1,9)
	CMP_SWAP(3,13)
	CMP_SWAP(5,15)
	CMP_SWAP(6,18)
	CMP_SWAP(7,19)
	CMP_SWAP(8,11)
	CMP_SWAP(10,20)
	CMP_SWAP(12,22)
	CMP_SWAP(14,17)
	CMP_SWAP(16,24)
	CMP_SWAP(21,25)
	CMP_SWAP(2,4)
	CMP_SWAP(3,11)
	CMP_SWAP(5,9)
	CMP_SWAP(10,12)
	CMP_SWAP(13,15)
	CMP_SWAP(14,22)
	CMP_SWAP(16,20)
	CMP_SWAP(21,23)
	CMP_SWAP(1,4)
	CMP_SWAP(3,8)
	CMP_SWAP(6,10)
	CMP_SWAP(7,13)
	CMP_SWAP(9,11)
	CMP_SWAP(12,18)
	CMP_SWAP(14,16)
	CMP_SWAP(15,19)
	CMP_SWAP(17,22)
	CMP_SWAP(21,24)
	CMP_SWAP(1,2)
	CMP_SWAP(3,6)
	CMP_SWAP(4,5)
	CMP_SWAP(7,12)
	CMP_SWAP(8,10)
	CMP_SWAP(9,14)
	CMP_SWAP(11,16)
	CMP_SWAP(13,18)
	CMP_SWAP(15,17)
	CMP_SWAP(19,22)
	CMP_SWAP(20,21)
	CMP_SWAP(23,24)
	CMP_SWAP(2,3)
	CMP_SWAP(4,6)
	CMP_SWAP(5,10)
	CMP_SWAP(7,9)
	CMP_SWAP(11,13)
	CMP_SWAP(12,14)
	CMP_SWAP(15,20)
	CMP_SWAP(16,18)
	CMP_SWAP(19,21)
	CMP_SWAP(22,23)
	CMP_SWAP(3,4)
	CMP_SWAP(5,6)
	CMP_SWAP(7,8)
	CMP_SWAP(9,10)
	CMP_SWAP(11,12)
	CMP_SWAP(13,14)
	CMP_SWAP(15,16)
	CMP_SWAP(17,18)
	CMP_SWAP(19,20)
	CMP_SWAP(21,22)
	CMP_SWAP(5,7)
	CMP_SWAP(6,8)
	CMP_SWAP(9,11)
	CMP_SWAP(10,12)
	CMP_SWAP(13,15)
	CMP_SWAP(14,16)
	CMP_SWAP(17,19)
	CMP_SWAP(18,20)
	CMP_SWAP(4,5)
	CMP_SWAP(6,7)
	CMP_SWAP(8,9)
	CMP_SWAP(10,11)
	CMP_SWAP(12,13)
	CMP_SWAP(14,15)
	CMP_SWAP(16,17)
	CMP_SWAP(18,19)
	CMP_SWAP(20,21)
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
	case 10:
		sorting_network_10(labels);
		break;
	case 11:
		sorting_network_11(labels);
		break;
	case 12:
		sorting_network_12(labels);
		break;
	// ensure this compiles to a jump table
	case 13:
	case 14:
	case 15:
	case 16:
	case 17:
		std::sort(labels.begin(), labels.end());
		break;
	case 18:
		sorting_network_18(labels);
		break;
	// ensure this compiles to a jump table
	case 19:
	case 20:
	case 21:
	case 22:
	case 23:
	case 24:
	case 25:
		std::sort(labels.begin(), labels.end());
		break;
	case 26:
		sorting_network_26(labels);
		break;
	case 27:
		sorting_network_27(labels);
		break;
	default:
		std::sort(labels.begin(), labels.end());
	}
}

#undef CMP_SWAP

};

#endif