# Tensor

The tensor module provides basic tensor operations and data structures for the Neuralisp machine learning framework. This module defines tensor classes and functions to create, manipulate and perform mathematical operations on multi-dimensional tensors, including support for GPU acceleration.

## Usage

To use the tensor module, import the functions and classes provided by the `neuralisp.core.tensor` package:

```common-lisp
(use-package :neuralisp.core.tensor)
```

## Classes

### `tensor`

The `tensor` class represents a multi-dimensional array (tensor) and contains the following slots:

- `data`: A simple-array holding the numerical values of the tensor.
- `shape`: A list of integers representing the dimensions of the tensor.

## Functions

### `make-tensor` (shape &key (initial-element 0) (on-gpu nil))

This function creates a new tensor with the specified shape and initializes to the given initial-element. If `on-gpu` is true, the created tensor will be moved to the GPU.

Arguments:

- `shape`: A list of integers representing the dimensions of the new tensor.
- `initial-element` (optional, default: `0`): The initial value used to fill the tensor.
- `on-gpu` (optional, default: `nil`): If true, the tensor will be moved to the GPU.

Returns:

- A new `tensor` instance.

## Tensor Operations

The following functions perform element-wise operations on tensors. 

- `tensor-add` (tensor-a tensor-b)
- `tensor-subtract` (tensor-a tensor-b)
- `tensor-multiply` (tensor-a tensor-b)
- `tensor-divide` (tensor-a tensor-b)

### Broadcasting

When performing element-wise operations on tensors with different shapes, the tensor module automatically broadcasts the smaller tensor to match the shape of the larger tensor, if the shapes are compatible.

### Matrix Multiplication

- `tensor-matmul` (tensor-a tensor-b &key (transpose-a nil) (transpose-b nil))

This function performs matrix multiplication between two tensors.

Arguments:

- `tensor-a` and `tensor-b`: Tensors to be multiplied
- `transpose-a` (optional, default: `nil`): If true, tensor-a will be transposed before multiplying
- `transpose-b` (optional, default: `nil`): If true, tensor-b will be transposed before multiplying

Returns:

- A new `tensor` instance representing the result of the multiplication.

### Reduction Operations

- `tensor-sum` (tensor &key (axis nil) (keepdims nil))
- `tensor-mean` (tensor &key (axis nil) (keepdims nil))

These functions perform reduction operations on a tensor along the specified axis or axes. If no axis is specified, the reduction is applied across all elements of the tensor.

Arguments:

- `tensor`: A tensor to perform the reduction operation on
- `axis` (optional, default: `nil`): An integer or list of integers representing the axis or axes to reduce
- `keepdims` (optional, default: `nil`): If true, the reduced axes will be kept with size 1

## Examples

Creating a tensor:

```common-lisp
(defparameter *a*
  (make-tensor (list 2 3) :initial-element 0.5d0))
```

Performing element-wise operations on tensors:

```common-lisp
(defparameter *b*
  (tensor-add *a* *a*))
```

Matrix multiplication:

```common-lisp
(defparameter *c*
  (tensor-matmul *a* (tensor-transpose *a*)))
```

Performing reduction operations:

```common-lisp
(defparameter *sum*
  (tensor-sum *a* :axis 0))

(defparameter *mean*
  (tensor-mean *a* :axis 1 :keepdims t))
```