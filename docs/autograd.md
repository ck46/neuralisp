# Autograd

The autograd module implements reverse-mode automatic differentiation.  A `variable` wraps a tensor and remembers the
variables it was computed from; `backward` walks that graph in reverse topological order, so a gradient that reaches a
node along several paths is summed and delivered exactly once.  Every function described here is covered by
[`tests/core/test_autograd.lisp`](../tests/core/test_autograd.lisp).

## Usage

The package exports a symbol named `variable`, which collides with `cl:variable`.  `use-package` will therefore signal
a name conflict — refer to the package qualified, or import deliberately:

```common-lisp
(defpackage :my-model
  (:use :cl :neuralisp.core.tensor)
  (:shadowing-import-from :neuralisp.core.autograd #:variable)
  (:import-from :neuralisp.core.autograd
                #:create-variable #:backward #:zero-gradient
                #:variable-value #:variable-gradient
                #:variable-add #:variable-multiply))
```

## Classes

### `variable`

A node in the autograd graph.

- `value` — the tensor this variable stands for.  Read with `variable-value`.
- `gradient` — the accumulated gradient tensor, or `nil` when gradients are not tracked.  Accessed with
  `variable-gradient`.
- `parents` — the variables this one was computed from; empty for a leaf.  Read with `variable-parents`.
- `backward-fn` — called with this node's gradient to pass it on to `parents`.  Accessed with `variable-backward`.
- `requires-grad` — whether gradients accumulate here.  Read with `variable-requires-grad-p`.

## Functions

### `create-variable` (value &key (requires-grad t))

Wraps the tensor `value` in a leaf variable.  With `requires-grad` true the variable starts with a zero gradient ready
to accumulate into; otherwise its gradient stays `nil` and gradients flowing to it are discarded.

There is no `:on-gpu` option: no GPU backend exists yet.

### `backward` (var &optional gradient)

Propagates `gradient` back from `var` through the graph that produced it, returning `var`.

- `gradient` defaults to a tensor of ones shaped like `var`'s value.  It is a **tensor**, not a scalar.
- Gradients accumulate into each variable's `gradient` slot across calls, mirroring the usual training-loop
  convention.  Call `zero-gradient` between independent backward passes.
- In-flight gradients are held per-pass rather than in the `gradient` slots, so a second pass starts from a clean seed
  instead of re-propagating what the first one accumulated.

### `zero-gradient` (var)

Resets `var`'s accumulated gradient to zero, returning `var`.  A no-op on a variable that does not require gradients.

### `propagate-gradient` (var gradient)

Queues `gradient` to reach `var` later in the current backward pass.  This is what a backward function calls to hand a
gradient to an operand; the traversal delivers it once every contribution has arrived.  Only useful when defining a new
differentiable operation.

## Differentiable operations

- `variable-add` (var-a var-b) — elementwise sum.  `d(a+b)/da = d(a+b)/db = 1`.
- `variable-multiply` (var-a var-b) — elementwise product.  `d(a*b)/da = b`, `d(a*b)/db = a`.

These two are what exists today.  Matrix multiply, activations, and losses are not yet differentiable; see
[ROADMAP.md](../ROADMAP.md).

### Defining a new operation

Compute the forward value, record the operands as parents, and give the node a backward function that calls
`propagate-gradient` on each operand with that operand's local derivative times the incoming gradient.

## Examples

```common-lisp
;; y = x * x, so dy/dx = 2x.  Both operands are the same node; the traversal
;; visits it once with the summed gradient.
(let* ((x (create-variable (make-tensor '(2) :data '(3 4))))
       (y (variable-multiply x x)))
  (backward y)
  (tensor-data (variable-gradient x)))       ; => #(6.0d0 8.0d0)

;; z = (a + b) * c  =>  dz/da = dz/db = c, dz/dc = a + b
(let* ((a (create-variable (make-tensor '(2) :data '(1 2))))
       (b (create-variable (make-tensor '(2) :data '(3 4))))
       (c (create-variable (make-tensor '(2) :data '(10 100))))
       (z (variable-multiply (variable-add a b) c)))
  (backward z)
  (values (tensor-data (variable-gradient a))    ; => #(10.0d0 100.0d0)
          (tensor-data (variable-gradient c))))  ; => #(4.0d0 6.0d0)

;; Seed the pass with an explicit gradient instead of ones.
(backward z (make-tensor '(2) :data '(2 5)))

;; Clear before an independent pass.
(zero-gradient a)
```

## Not implemented

`partial-grad` and `apply-partial-grad`, described in earlier revisions of this document, have been removed.  They
walked the `backward` slot as though it were a parent pointer and combined gradients with `cl:*` as though tensors were
numbers; neither could run.  `backward` replaces both.
