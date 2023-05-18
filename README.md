# NeuraLisp

NeuraLisp is a machine learning framework for Common Lisp, designed for ease of use and extensibility. It provides a modular implementation for building, training, and evaluating neural networks for various machine learning tasks.

## Features

- Core components for tensor operations, autograd, GPU support, and more
- A variety of layers, including linear, convolutional, recurrent, and attention-based layers
- Common activation functions like ReLU, Sigmoid, and Tanh
- Loss functions like mean squared error and cross-entropy
- Optimizers like Stochastic Gradient Descent (SGD) and Adam

## Installation

To install NeuraLisp, clone the repository and load the system definition using Quicklisp or ASDF:

```
git clone https://github.com/yourusername/NeuraLisp.git
```

Then, in your Common Lisp REPL:

```lisp
(ql:quickload :NeuraLisp)
```

## Usage

Here's an example of how to create a simple neural network using NeuraLisp:

```lisp
(use-package :NeuraLisp.core.tensor)
(use-package :NeuraLisp.layers.linear)
(use-package :NeuraLisp.activations.relu)

;; Create a linear layer
(defvar *layer* (make-instance 'linear :input-dim 3 :output-dim 2))

;; Create input tensor
(defvar *input* (make-tensor #(1.0 2.0 3.0) #(1 3)))

;; Apply the linear layer
(defvar *output* (forward *layer* *input*))

;; Apply ReLU activation
(defvar *relu* (make-instance 'relu))
(defvar *activated-output* (activate *relu* *output*))
```

For more examples, please refer to the `examples/` folder.

## Documentation

Detailed documentation for each component can be found in the `docs/` folder.

## Contributing

We welcome contributions to NeuraLisp! If you find a bug, want to improve the code quality, or have ideas for new features, please feel free to create an issue or submit a pull request on GitHub.

## License

NeuraLisp is licensed under the MIT License. Please see the `LICENSE` file for more information.
