# Autograd

The autograd module provides automatic differentiation and gradient computation for the Neuralisp machine learning framework. This module defines variable classes and functions for creating variables that hold tensor values, gradient tensors, and backward functions to compute changes in variables when updating the network.

## Usage

To use the autograd module, import the functions and classes provided by the `neuralisp.core.autograd` package:

```common-lisp
(use-package :neuralisp.core.autograd)
```

## Classes

### `variable`

The `variable` class represents an autograd variable and contains the following slots:

- `value`: A tensor that represents the value of the variable.
- `gradient`: A tensor (or null) that represents the gradient of the variable.
- `backward`: A function (or null) used for computing the gradients during the backward pass.

## Functions

### `create-variable` (value &key (requires-grad t) (on-gpu nil))

This function creates a new autograd variable with the input value, a gradient tensor (if requires-grad is true), and sets the backward function to `nil`. If `on-gpu` is true, the created variable and its gradients will be moved to the GPU.

Arguments:

- `value`: A tensor representing the value of the new variable.
- `requires-grad` (optional, default: `t`): If true, the variable's gradient tensor will be created.
- `on-gpu` (optional, default: `nil`): If true, the variable's tensor and gradients will be moved to the GPU.

Returns:

- A new autograd `variable` instance.

### `backward` (var &optional (grad-output 1.0))

Computes the gradients of the variable with respect to its values and accumulates gradient output.

Arguments:

- `var`: An autograd `variable` to compute gradients for.
- `grad-output` (optional, default: `1.0`): A scalar value for the gradient output's accumulation.

### `zero-gradient` (var)

Sets the gradient of the variable to zero.

Arguments:

- `var`: An autograd `variable` instance whose gradient will be set to zero.

### `partial-grad` (node-a node-b)

Computes the partial derivatives between two 'variable' nodes.

Arguments:

- `node-a` and `node-b`: `variable` nodes in the computational graph.

Returns:

- A scalar representing the computed partial gradients.

### `apply-partial-grad` (node-a node-b)

Applies the computed partial gradients of node-b with respect to node-a to the gradients of both nodes.

Arguments:

- `node-a` and `node-b`: `variable` nodes in the computational graph.

## Examples

Creating an autograd variable:

```common-lisp
(defparameter *var*
  (create-variable (make-tensor (list 3 3) '(0.5d0 0.5d0))))
```

Performing the backward pass on a variable:

```common-lisp
; Assuming *var* has a backward function assigned
(backward *var*)
```

Setting the gradient of a variable to zero:

```common-lisp
(zero-gradient *var*)
```

Computing and applying partial gradients between variables:

```common-lisp
; Assuming *var-a* and *var-b* are variables in the computational graph
(defparameter *partial*
  (partial-grad *var-a* *var-b*))

(apply-partial-grad *var-a* *var-b*)
```