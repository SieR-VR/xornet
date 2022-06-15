#include <iostream>
#include <vector>
#include "src/layers.h"

int main()
{
    Dense<2, 10> dense1 = Dense<2, 10>();
    Dense<10, 2> dense4 = Dense<10, 2>();

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
            dense4.forward(dense1.forward(input[i]));
            dense1.backward(dense4.backward(output[i]));
        }
    }

    for (size_t i = 0; i < input.size(); i++)
    {
        auto result = dense4.forward(dense1.forward(input[i]));

        std::cout << "input: " << input[i] << std::endl;
        std::cout << "result: " << result << std::endl;
        std::cout << "target: " << output[i] << std::endl
                  << std::endl;
    }
}