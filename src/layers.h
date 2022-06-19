#ifndef LAYERS_H_
#define LAYERS_H_

#include "vec.h"

template <size_t I, size_t O>
class BaseNet
{
public:
    BaseNet() {}

    Quantized<O> forward(const Quantized<I> &input);
    Vector<int, I> backward(const Vector<int, O> &delta);
};

template <size_t I, size_t O>
class Dense : public BaseNet<I, O>
{
public:
    Dense() : weights()
    {
    }

    Quantized<O> forward(const Quantized<I> &input)
    {
        input_temp = input;
        output_temp = input.dot(weights);
        return output_temp;
    }

    Tensor<int, I> backward(const Tensor<int, O> &next_delta, double lr)
    {
        Tensor<int, I> delta = dot_delta(next_delta, weights.transpose());
        weights.update(dot_reverse(next_delta, input_temp), lr);

        return delta;
    }

private:
    Quantized<O, I> weights;

    Quantized<I> input_temp;
    Quantized<O> output_temp;
};

template <size_t I, size_t O>
class Softmax : public BaseNet<I, O>
{
public:
    Softmax() : weights()
    {
    }

    Tensor<size_t, O> forward(const Quantized<I> &input)
    {
        input_temp = input;
        return input.dot_raw(weights);
    }

    Tensor<int, I> backward(const Tensor<int, O> &delta_next, double lr)
    {
        Tensor<int, I> delta = dot_delta(delta_next, transpose(weights));
        weights.update(dot_reverse(delta_next, input_temp), lr);

        return delta;
    }

private:
    Quantized<O, I> weights;

    Quantized<I> input_temp;
    Quantized<O> output_temp;
};

#endif
