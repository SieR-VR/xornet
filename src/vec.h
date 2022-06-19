#ifndef VEC_H_
#define VEC_H_

#include <bitset>
#include <vector>
#include <random>
#include <type_traits>

static std::random_device rd;
static std::mt19937 gen(rd());

template <class T, size_t S, size_t... Sizes>
class Tensor {
public:
    Tensor() {}
    Tensor(const std::initializer_list<Tensor<T, Sizes...>> &list) {
        size_t idx = 0;
        for (auto &tensor : list) {
            data[idx] = tensor;
            idx++;
        }
    }

    Tensor<T, Sizes...> &operator[](size_t i) {
        return data[i];
    }

    const Tensor<T, Sizes...> &operator[](size_t i) const {
        return data[i];
    }

private:
    Tensor<T, Sizes...> data[S];
};

template <class T, size_t S>
class Tensor<T, S> {
public:
    Tensor() {}
    Tensor(const std::initializer_list<int> &list) {
        size_t idx = 0;
        for (auto &tensor : list) {
            data[idx] = tensor;
            idx++;
        }
    }

    T &operator[](size_t i) {
        return data[i];
    }

    const T &operator[](size_t i) const {
        return data[i];
    }

    T data[S];
};

template <size_t S>
class Tensor<bool, S> : public std::bitset<S> {
public:
    Tensor() {}

    Tensor(const std::initializer_list<bool> &list) {
        size_t idx = 0;
        for (auto &tensor : list) {
            (*this)[idx] = tensor;
            idx++;
        }
    }

    Tensor(const std::bitset<S> &bits) : std::bitset<S>(bits) {}

    Tensor<bool, S> operator*(const Tensor<bool, S> &other) const {
        return ~(*this ^ other);
    }

    template <size_t SO>
    Tensor<bool, SO> dot(const Tensor<bool, SO, S> &b) const {
        Tensor<bool, SO> result;
        for (size_t i = 0; i < SO; i++)
            result[i] = ((*this) * b[i]).count() * 2 < S;
        return result;
    }

    template <size_t SO>
    Tensor<size_t, SO> dot_raw(const Tensor<bool, SO, S> &b) const {
        Tensor<size_t, SO> result;
        for (size_t i = 0; i < SO; i++)
            result[i] = S - ((*this) * b[i]).count();
        return result;
    }

    template <size_t SO>
    Tensor<bool, 1, SO> transpose() {
        Tensor<bool, 1, SO> result;
        for (size_t i = 0; i < SO; i++)
            result[i] = (*this)[i];
        return result;
    }
};

template <class T, size_t S>
using Vector = Tensor<T, S>;

template <size_t... Sizes>
using Quantized = Tensor<bool, Sizes...>;

template <class T, size_t S1, size_t S2>
Tensor<T, S2, S1> transpose(const Tensor<T, S1, S2> &t) {
    Tensor<T, S2, S1> result;
    for (size_t i = 0; i < S1; i++)
        for (size_t j = 0; j < S2; j++)
            result[j][i] = t[i][j];
    return result;
}

template <class T, size_t S>
size_t argmax(const Tensor<T, S> &t) {
    size_t max_index, max_value = 0;
    for (size_t i = 0; i < S; i++) {
        if (t[i] > max_value) {
            max_index = i;
            max_value = t[i];
        }
    }

    return max_index;
}

template <size_t S>
Quantized<S> one_hot(size_t index) {
    Quantized<S> result;
    result[index] = 1;
    return result;
}

template <size_t S1, size_t S2>
Tensor<int, S1> dot_delta(const Tensor<int, S2> &delta, const Quantized<S1, S2> &weights) {
    Tensor<int, S1> result;
    for (size_t i = 0; i < S1; i++) {
        result[i] = 0;
        for (size_t j = 0; j < S2; j++)
            result[i] += weights[i][j] ? delta[j] : -delta[j];
    }
    return result;
}

template <size_t S1, size_t S2>
Tensor<int, S1> loss(const Tensor<size_t, S1> &output, const Quantized<S1> &target) {
    Tensor<int, S1> result;
    for (size_t i = 0; i < S1; i++)
        result[i] = output[i] * 2 - S2 + (target[i] ? S2 : -S2);
    return result;
}


#endif