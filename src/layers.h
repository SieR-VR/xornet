#ifndef LAYERS_H_
#define LAYERS_H_

#include <initializer_list>
#include <vector>
#include <random>

#include "vec.h"

template <size_t I, size_t O>
class BaseNet
{
public:
    BaseNet() {}

    Vec<O> forward(const Vec<I> &input);
    Vec<I> backward(const Vec<O> &delta);
};

template <size_t I, size_t O>
class Dense : public BaseNet<I, O>
{
public:
    Dense()
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::bernoulli_distribution dist(0.5);

        for (size_t i = 0; i < I * O; i++)
            weights[i] = dist(gen);
    }

    Vec<O> forward(const Vec<I> &input)
    {
        input_temp = input;
        Vec<O> output;

        for (size_t i = 0; i < O; i++)
        {
            Vec<I *O> weight_shifted = (weights >> (i * I));
            Vec<I> weight_sliced;
            for (size_t j = 0; j < I; j++)
                weight_sliced[j] = weight_shifted[j];

            size_t product = (input ^ weight_sliced).count();

            output[i] = (product >= (I / 2));
        }

        output_temp = output;
        return output;
    }

    Vec<I> backward(const Vec<O> &delta)
    {
        Vec<I> input_delta = input_temp;
        Vec<O> diff = delta ^ output_temp;

        std::vector<size_t> diff_count(I, 0);

        for (size_t i = 0; i < O; i++)
        {
            for (size_t j = 0; j < I; j++)
            {
                if (diff[i])
                {
                    weights[i * I + j] = ~weights[i * I + j];
                    diff_count[j]++;
                }
            }
        }

        for (size_t i = 0; i < I; i++)
        {
            if (diff_count[i] > (O / 2))
                input_delta[i] = ~input_temp[i];
        }

        return input_delta;
    }

private:
    Vec2<I, O> weights;

    Vec<I> input_temp;
    Vec<O> output_temp;
};

#endif