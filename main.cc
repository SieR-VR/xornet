#include <iostream>
#include <vector>
#include "src/layers.h"

int main()
{
    Dense<2, 10> dense1 = Dense<2, 10>();
    Dense<10, 10> dense2 = Dense<10, 10>();
    Softmax<10, 2> dense4 = Softmax<10, 2>();

    Quantized<4, 2> input = Quantized<4, 2>({
        { 0, 0 },
        { 0, 1 },
        { 1, 0 },
        { 1, 1 }
    });

    Quantized<4, 2> target = Quantized<4, 2>({
        { 0, 1 },
        { 1, 0 },
        { 1, 0 },
        { 0, 1 }
    });

    for (size_t epoch = 1; epoch <= 1000; epoch++)
    {
        for (size_t i = 0; i < 4; i++)
        {
            auto output = dense1.forward(input[i]);
            output = dense2.forward(output);
            auto result = dense4.forward(output);

            auto loss_ = loss<2, 10>(result, target[i]);

            auto delta = dense4.backward(loss_, 0.1);
            delta = dense2.backward(delta, 0.1);
            dense1.backward(delta, 0.2);
        }
    }

    for (size_t i = 0; i < 4; i++)
    {
        auto result = dense2.forward(dense1.forward(input[i]));

        std::cout << "input: " << input[i] << std::endl;
        std::cout << "result: " << result << std::endl;
        std::cout << "target: " << target[i] << std::endl
                  << std::endl;
    }

    Tensor<int, 2, 4> t = Tensor<int, 2, 4>({{1, 2, 3, 4}, {5, 6, 7, 8}});
}