#ifndef VEC_H_
#define VEC_H_

#include <bitset>

template <size_t T>
using Vec = std::bitset<T>;

template <size_t T, size_t U>
using Vec2 = Vec<T * U>;

#endif