# Multi-Layer Perceptron in Rust
A simple AI model written without any AI-related dependencies (i.e. from scratch). It's applied on the MNIST digits dataset.

## How does this work?
The model's layers and weights are first initialized in lines `19-26` in `src/main.rs`. After that, the data files are loaded and parsed to create two sets: training and validation. The big `for` loop trains the model on all 60,000 digits images and then validates it on (never-before-seen) data in the validation set (which has 10,000 images). For each batch of images (i.e., set of images on which the model is trained simultaneously), the model does a forward pass to make its prediction, and then it does a backwards pass to update the weights. This is the training phase of each epoch, and it is implemented in lines `54-81`, though the most important parts are just on lines `55-60` and `81`. For validation, the model does a forward pass on all the images in the validation set and outputs how many it got correct.

All the matrix and AI functions are implemented from scratch to directly show how the MLP works at its core. The two most important function are `model_forward_pass` and `model_backprop_pass`, which are located on lines `335-349` and `351-388`, respectively.

For a mathematical/technical description of how an MLP can be implemented in code, there are many great resources online, such as [this](https://github.com/KirillShmilovich/MLP-Neural-Network-From-Scratch/blob/master/MLP.ipynb) or [this one](https://python.plainenglish.io/building-the-foundations-a-step-by-step-guide-to-implementing-multi-layer-perceptrons-in-python-51ebd9d7ecbe).

(There's also an extra `preview_img` function available, in case you want to see how one of the images from the data set looks like).

## How can I try this out?
0. Download the data set (training images, training labels, test images, test labels). [This link](https://github.com/cvdfoundation/mnist) has all four files. Make sure to uncompress them and move them to a `data` folder in the directory of this repo.
1. Execute the `cargo run` command, which will download the image and RNG libraries, compile the code, and run the model. Use the `--release` flag for faster performance.
2. There will be messages in the console about the progress of the model. With the default parameters in `src/main.rs` (`1` hidden layer, `200` neurons in the hidden, `2` epochs, `10` batch size, and `0.01` learning rate), you can usually get 70%-80% accuracy. Play around with the parameters to see how they affect the training time and accuracy of the mode.

