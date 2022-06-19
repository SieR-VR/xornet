#include <iostream>
#include <vector>
#include "src/layers.h"
#include "src/mnist.h"

int main()
{
    // Dense<784, 784> dense1 = Dense<784, 784>();
    Softmax<784, 10> softmax = Softmax<784, 10>();

    int train_size, image_size;
    auto mnist_images = read_mnist_images("./train-images.idx3-ubyte", train_size, image_size);
    auto mnist_labels = read_mnist_labels("./train-labels.idx1-ubyte", train_size);

    std::vector<Vector<784>> train_images(train_size);
    std::vector<Vector<10>> train_labels(train_size, Vector<10>().reset());

    for (int i = 0; i < train_size; i++)
    {
        for (int j = 0; j < 784; j++)
            train_images[i][j] = mnist_images[i][j] > 127;
        train_labels[i][mnist_labels[i]] = 1;
    }

    for (int epochs = 1; epochs <= 10; epochs++)
    {
        double lr = 0.3;

        // dense1.update_lr(lr);
        softmax.update_lr(lr);

        for (int j = 0; j < train_size; j++)
        {
            // auto output = dense1.forward(train_images[j]);
            softmax.forward(train_images[j]);

            auto loss = softmax.backward(train_labels[j]);
            // dense1.backward(loss);
        }

        if (epochs % 1 == 0) {
            std::cout << "Epoch: " << epochs << std::endl;
            lr *= 0.75;
        }
    }

    int test_size;
    mnist_images = read_mnist_images("./t10k-images.idx3-ubyte", test_size, image_size);
    mnist_labels = read_mnist_labels("./t10k-labels.idx1-ubyte", test_size);

    std::vector<Vector<784>> test_images(test_size);
    std::vector<Vector<10>> test_labels(test_size, Vector<10>().reset());

    for (int i = 0; i < test_size; i++)
    {
        for (int j = 0; j < 784; j++)
            test_images[i][j] = mnist_images[i][j] > 127;
        test_labels[i][mnist_labels[i]] = 1;
    }

    int correct = 0;
    for (int i = 0; i < test_size; i++)
    {
        // auto output = dense1.forward(test_images[i]);
        auto result = softmax.forward(test_images[i]);

        correct += (result == test_labels[i]);
    }

    std::cout << std::endl
              << "Accuracy: " << correct / (float)test_size * 100.f << "%" << std::endl;
}