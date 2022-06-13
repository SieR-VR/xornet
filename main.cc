#include <iostream>
#include <vector>
#include "src/layers.h"

int main()
{
    Dense<2, 100> dense1 = Dense<2, 100>();
    Dense<100, 100> dense2 = Dense<100, 100>();
    Dense<100, 100> dense3 = Dense<100, 100>();
    Dense<100, 2> dense4 = Dense<100, 2>();

    std::vector<Bitvec<2>> input = {
        Bitvec<2>(0b00),
        Bitvec<2>(0b01),
        Bitvec<2>(0b10),
        Bitvec<2>(0b11)};

    std::vector<Bitvec<2>> output = {
        Bitvec<2>(0b01),
        Bitvec<2>(0b10),
        Bitvec<2>(0b10),
        Bitvec<2>(0b01)};

    for (size_t epoch = 1; epoch <= 1000; epoch++)
    {
        for (size_t i = 0; i < input.size(); i++)
        {
            dense4.forward(dense3.forward(dense2.forward(dense1.forward(input[i]))));
            dense1.backward(dense2.backward(dense3.backward(dense4.backward(output[i]))));
        }
    }

    for (size_t i = 0; i < input.size(); i++)
    {
        auto result = dense4.forward(dense3.forward(dense2.forward(dense1.forward(input[i]))));

        std::cout << "input: " << input[i] << std::endl;
        std::cout << "result: " << result << std::endl;
        std::cout << "target: " << output[i] << std::endl
                  << std::endl;
    }
}