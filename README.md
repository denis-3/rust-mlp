
# Multi-Layer Perceptron in Rust
A simple AI model written without any AI- or data-science-related dependencies (i.e. from scratch). It's applied on the MNIST digits dataset.

Some features of this model are...
* Fully configurable design: adjust the model's hyperparameters and structure.
* Leaky ReLU activation function: this is generally the best activation function for small MLP models; it overcomes the "neuron death" exhibited in the classic ReLU function by having f(x) = mx for x < 0, where m is a very small slope (in the code, it's a default of `0.01`).
* Dropout: some (`25%` by default) of the neurons get dropped for each backpropagation call, which makes the model overfit less to the training data set.
* Parallelization: a couple of the high-volume functions are parallelized.
* Model persistence: the model's weights can be saved to and loaded from a file on disk.

## High-Level Overview
If the model is training from scratch, it works as follows.
1. Initialize the model weights across the input, hidden, and output layers.
2. Load the training and validation digit sets.
3. Do inference (i.e. a prediction) on each of the digits in the training set, and do backpropagation on the digits it incorrectly predicted.
4. Do inference on the validation set to check the accuracy.
5. Repeat steps 3 and 4 for each remaining epoch.
6. *Optional*. Save the model's weights to `model-weights.bin`.

If, instead, the model is being validated on a test image, it does the following:
1. Initialize model weights from `model-weights.bin`.
2. Load image from disk.
3. Do inference on the image.

All the matrix and AI functions are implemented from scratch to directly show how the MLP works at its core. The two most important function are `model_forward_pass` and `model_backprop_pass`, which are located near the end of `main.rs`. Also, the `matrix_multiply()` and `matrix_subtract()` functions are parallelized, meaning they use multiple threads for quicker execution.

There are also many adjustable global variables that can control the model's structure and behavior. Find them near the top of `main.rs` along with accompanying comments.

For a mathematical/technical description of how an MLP can be implemented in code, there are many great resources online, such as [this](https://github.com/KirillShmilovich/MLP-Neural-Network-From-Scratch/blob/master/MLP.ipynb) or [this one](https://python.plainenglish.io/building-the-foundations-a-step-by-step-guide-to-implementing-multi-layer-perceptrons-in-python-51ebd9d7ecbe).

(There's also an extra `preview_img` function available, in case you want to see how one of the images from the data set looks like.)

## How can I try this out?
0. Download the data set (training images, training labels, test images, test labels). [This link](https://github.com/cvdfoundation/mnist), for example, has all four files. Make sure to uncompress them and move them to a `data` folder in the directory of this repo.
1. Execute the `cargo run` command, which will download the required libraries, compile the code, and run the model. Use the `--release` flag for faster performance.
2. There will be messages in the console about the progress of the model. With the default parameters in `src/main.rs` (`2` hidden layers, `300` neurons in the hidden layer, `10` epochs, `10` batch size, `7e-3` learning rate, and `0.25` drop rate), you can usually get about 93% accuracy. Play around with the parameters to see how they affect the training time and accuracy of the mode.
3. Once you've trained the model, you can set `LOAD_MODEL_FROM_FILE` to `true` to test the model on an image of your choice. Just make sure that the image is placed in `./test-image.png`.

