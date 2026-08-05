# Tensor

The tensor module provides the dense numeric container the rest of NeuraLisp is built on.  It is implemented in ANSI
Common Lisp with no external dependencies: a tensor is a flat row-major vector of `double-float`s plus an explicit
shape.  Every function described here is covered by [`tests/core/test_tensor.lisp`](../tests/core/test_tensor.lisp).

## Usage

```common-lisp
(use-package :neuralisp.core.tensor)
```

## Classes

### `tensor`

- `data` — a `(simple-array double-float (*))` holding the elements in row-major order.
- `shape` — a list of positive integers giving the extent of each dimension.
- `device` — `:cpu` or `:gpu`.  Always `:cpu` unless a GPU backend is installed; see [gpu.md](#gpu-placement).
- `gpu-pointer` — a backend-specific device handle, or `nil` on the CPU.

Elements are always stored as `double-float`, whatever the input type.  `(tensor-ref t 0 0)` on a tensor built from
integers returns `1.0d0`, not `1`.

## Functions

### `make-tensor` (shape &key (initial-element 0) data)

Creates a tensor of `shape`.

- `shape` — a non-empty list of positive integers.  Anything else signals an error.
- `initial-element` (default `0`) — the fill value, coerced to `double-float`.
- `data` — a sequence of numbers to adopt as the row-major contents.  Its length must equal the product of `shape`,
  otherwise `shape-mismatch` is signalled.  A vector that is already `(simple-array double-float (*))` is adopted
  without copying, so do not mutate it afterwards.

`initial-element` is ignored when `data` is supplied.

### `tensor-ref` (tensor &rest subscripts)

Reads one element; `setf`-able.  Signals an error when the number of subscripts does not match the rank, or when any
subscript is out of bounds.

### `tensor-copy` (tensor)

Returns a tensor sharing no storage with the original.

### `tensor-equal-p` (tensor-a tensor-b &key (tolerance 1d-9))

True when both tensors have the same shape and agree elementwise within `tolerance`.

### `tensor-rank` / `tensor-size`

Number of dimensions, and total number of elements.

## Tensor operations

Element-wise, on two tensors of identical shape:

- `tensor-add` (tensor-a tensor-b)
- `tensor-subtract` (tensor-a tensor-b)
- `tensor-multiply` (tensor-a tensor-b) — the Hadamard product, **not** a matrix product
- `tensor-divide` (tensor-a tensor-b)

### Broadcasting

**Not implemented.**  Operands must have identical shapes; anything else signals `shape-mismatch`.  Broadcasting is on
the roadmap, and until it lands these operations refuse rather than guess.

### Transpose

- `tensor-transpose` (tensor)

Rank-2 only.  Higher ranks signal an error.

### Matrix multiplication

- `tensor-matmul` (tensor-a tensor-b &key transpose-a transpose-b)

Rank-2 only.  Transposes the named operand first, then requires the inner dimensions to agree; a mismatch signals
`shape-mismatch`.  Returns a new tensor of shape `(rows-of-a cols-of-b)`.

### Reduction operations

- `tensor-sum` (tensor &key axis keepdims)
- `tensor-mean` (tensor &key axis keepdims)

With `axis` `nil`, the whole tensor is reduced and a `double-float` **scalar** is returned (or, with `keepdims`, a
tensor of all-1 dimensions).  With an integer `axis`, a rank-2 tensor is reduced along that axis and a tensor is
returned; `keepdims` decides whether the reduced axis survives with extent 1.

Reducing a single axis of a rank > 2 tensor is not implemented and signals an error rather than returning something
plausible.  `axis` does not accept a list of axes.

## Conditions

### `shape-mismatch`

Signalled when operands disagree on shape.  Readers: `shape-mismatch-operation`, `shape-mismatch-expected`,
`shape-mismatch-actual`.

## Examples

```common-lisp
(defparameter *a* (make-tensor '(2 3) :initial-element 0.5))

;; From explicit contents, in row-major order.
(defparameter *b* (make-tensor '(2 2) :data '(1 2 3 4)))

;; Element-wise.
(tensor-add *a* *a*)

;; [[1 2] [3 4]] @ [[5 6] [7 8]] = [[19 22] [43 50]]
(tensor-matmul *b* (make-tensor '(2 2) :data '(5 6 7 8)))

;; (2x3) @ (3x2) via transpose.
(tensor-matmul *a* *a* :transpose-b t)

;; Reductions.
(tensor-sum *b*)                        ; => 10.0d0, a scalar
(tensor-sum *b* :axis 0)                ; => a (2) tensor: #(4.0d0 6.0d0)
(tensor-mean *b* :axis 1 :keepdims t)   ; => a (2 1) tensor
```

## GPU placement

Tensors carry `device` and `gpu-pointer` slots, but there is no working device backend; see
[`src/core/gpu.lisp`](../src/core/gpu.lisp).  `make-tensor` has no `:on-gpu` option — call
`neuralisp.core.gpu:move-to-gpu` explicitly, which signals `gpu-backend-unavailable` until a backend is installed.
