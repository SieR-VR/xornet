#include <iostream>
#include <vector>
#include "src/layers.h"

int main()
{
    Dense<2, 100> dense1 = Dense<2, 100>();
    Dense<100, 100> dense2 = Dense<100, 100>();
    Dense<100, 1> dense3 = Dense<100, 1>();

    std::vector<Vec<2>> input = {
        Vec<2>(0b00),
        Vec<2>(0b01),
        Vec<2>(0b10),
        Vec<2>(0b11)};

    std::vector<Vec<1>> output = {
        Vec<1>(0b0),
        Vec<1>(0b1),
        Vec<1>(0b1),
        Vec<1>(0b0)};

    for (size_t epoch = 1; epoch <= 100; epoch++)
    {
        for (size_t i = 0; i < input.size(); i++)
        {
            auto result = dense3.forward(dense2.forward(dense1.forward(input[i])));
            dense1.backward(dense2.backward(dense3.backward(output[i])));
        }
    }

    for (size_t i = 0; i < input.size(); i++)
    {
        auto result = dense3.forward(dense2.forward(dense1.forward(input[i])));

        std::cout << "input: " << input[i] << std::endl;
        std::cout << "result: " << result << std::endl;
        std::cout << "target: " << output[i] << std::endl
                  << std::endl;
    }
}