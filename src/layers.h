#ifndef LAYERS_H_
#define LAYERS_H_

#include <initializer_list>
#include <vector>
#include <random>
#include <iostream>

#include "vec.h"

template <size_t I, size_t O>
class BaseNet
{
public:
    BaseNet() {}

    Bitvec<O> forward(const Bitvec<I> &input);
    Bitvec<I> backward(const Bitvec<O> &delta);
};

template <size_t I, size_t O>
class Dense : public BaseNet<I, O>
{
public:
    Dense() : weights(O), gen(rd()), dist(0.5)
    {
        for (size_t i = 0; i < O; i++)
            for (size_t j = 0; j < I; j++)
                weights[i][j] = dist(gen);
    }

    Bitvec<O> forward(const Bitvec<I> &input)
    {
        input_temp = input;
        Bitvec<O> output;

        for (size_t i = 0; i < O; i++)
        {
            size_t product = (input ^ weights[i]).count();
            output[i] = (product * 2 >= I);
        }

        output_temp = output;
        return output;
    }

    Bitvec<I> backward(const Bitvec<O> &delta)
    {
        Bitvec<O> diff = delta ^ output_temp;
        Bitvec<I> delta_input = input_temp;

        std::vector<Bitvec<I>> weights_diff = weights;

        for (size_t i = 0; i < O; i++)
            if (diff[i] && dist(gen))
                weights[i] = (delta[i] ? Bitvec<I>().set() : Bitvec<I>().reset()) ^ input_temp;

        for (size_t i = 0; i < O; i++)
            weights_diff[i] ^= weights[i];

        for (size_t i = 0; i < I; i++)
        {
            size_t product = 0;
            for (size_t j = 0; j < O; j++)
                product += weights_diff[j][i];
            if (product * 2 >= O)
                delta_input[i] = ~delta_input[i];
        }

        return delta_input;
    }

private:
    std::vector<Bitvec<I>> weights;

    Bitvec<I> input_temp;
    Bitvec<O> output_temp;

    std::random_device rd;
    std::mt19937 gen;
    std::bernoulli_distribution dist;
};

#endif
