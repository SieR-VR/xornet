#include <iostream>
#include <vector>
#include "src/layers.h"
#include "src/mnist.h"

int main()
{
    Dense<784, 100> dense1 = Dense<784, 100>();
    Dense<100, 100> dense2 = Dense<100, 100>();
    Dense<100, 100> dense3 = Dense<100, 100>();
    Dense<100, 10> dense4 = Dense<100, 10>();

    int train_size, image_size;
    auto mnist_images = read_mnist_images("./train-images-idx3-ubyte", train_size, image_size);
    auto mnist_labels = read_mnist_labels("./train-labels-idx1-ubyte", train_size);

    std::vector<Bitvec<784>> train_images(train_size);
    std::vector<Bitvec<10>> train_labels(train_size, Bitvec<10>().reset());

    for(int i = 0; i < train_size; i++) {
        for(int j = 0; j < 784; j++)
            train_images[i][j] = mnist_images[i][j] > 127;
        train_labels[i][mnist_labels[i]] = 1;
    }

    for (int epochs = 1; epochs <= 1; epochs++) {
        for(int j = 0; j < train_size; j++) {
            dense4.forward(dense3.forward(dense2.forward(dense1.forward(train_images[j]))));
            dense1.backward(dense2.backward(dense3.backward(dense4.backward(train_labels[j]))));
            
            if (j % 1000 == 0)
                std::cout << "Batch: " << j << std::endl;
        }

    }

    int test_size;
    mnist_images = read_mnist_images("./t10k-images-idx3-ubyte", test_size, image_size);
    mnist_labels = read_mnist_labels("./t10k-labels-idx1-ubyte", test_size);

    std::vector<Bitvec<784>> test_images(test_size);
    std::vector<Bitvec<10>> test_labels(test_size, Bitvec<10>().reset());

    for(int i = 0; i < 100; i++) {
        for(int j = 0; j < 784; j++)
            test_images[i][j] = mnist_images[i][j] > 127;
        test_labels[i][mnist_labels[i]] = 1;
    }

    int correct = 0;
    for (int i = 0; i < test_size; i++) {
        auto output = dense4.forward(dense2.forward(dense1.forward(test_images[i])));
        correct += (output == test_labels[i]);
    }

    std::cout << std::endl << "Accuracy: " << correct / (float)test_size * 100.f << "%"; 
}