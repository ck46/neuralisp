# NeuraLisp

NeuraLisp is an experimental neural computing environment for Common Lisp.  It is at a very early stage: there is a
working tensor core and a working reverse-mode autograd, and essentially nothing above them.  The layers, optimisers,
losses, transformers, and cognitive agents described in the roadmap are empty files today.

## What actually exists

- **A tensor core** in [`src/core/tensor.lisp`](src/core/tensor.lisp): dense row-major `double-float` tensors,
  elementwise arithmetic, transpose, matrix multiply, and full/axis reductions.  Covered by unit tests.
- **Reverse-mode autograd** in [`src/core/autograd.lisp`](src/core/autograd.lisp): differentiable variables, a graph
  walked in reverse topological order, and gradient accumulation.  Covered by unit tests, including the diamond case
  where one variable feeds an operation twice.
- **A GPU placement interface** in [`src/core/gpu.lisp`](src/core/gpu.lisp).  There is **no working device backend**;
  requesting one signals `gpu-backend-unavailable`.  The module defines the interface a backend must implement.
- **Examples** under [`examples/`](examples).  Only [`simple-mlp.lisp`](examples/simple-mlp.lisp) uses the library;
  the other two are standalone illustrative sketches that deliberately depend on nothing.
- **A manifesto and roadmap** describing where this is meant to go.  Read them as intent, not as description.

Everything is written in ANSI Common Lisp with no external dependencies, so it loads and tests on a bare SBCL.

## Quickstart

### 1. Install dependencies

| Dependency | Purpose | Notes |
|------------|---------|-------|
| [SBCL](https://www.sbcl.org/) (or another ANSI Common Lisp) | Runs the NeuraLisp source, tests, and examples | Tested with SBCL 2.2.9; ASDF ships with it |

There are no third-party library dependencies, so Quicklisp is not required.  A future accelerated backend
([`magicl`](https://github.com/quil-lang/magicl), BLAS, or CUDA) would introduce one, behind the existing entry points.

## Quickstart

### 1. Install dependencies

| Dependency | Purpose | Notes |
|------------|---------|-------|
| [SBCL](https://www.sbcl.org/) (or another ANSI Common Lisp) | Runs the NeuraLisp source and examples | Tested with SBCL ≥ 2.3 |
| [Quicklisp](https://www.quicklisp.org/beta/) | Manages third-party libraries | Required to pull `magicl` and other math deps |
| [`magicl`](https://github.com/quil-lang/magicl) | Dense linear algebra backend | Load through Quicklisp (`(ql:quickload :magicl)`) |
| [`cl-cuda`](https://github.com/takagi/cl-cuda) *(optional)* | CUDA bindings for GPU experiments | Only needed if you intend to evaluate `neuralisp.core.gpu` |

```bash
git clone https://github.com/yourusername/neuralisp.git
cd neuralisp
```

### 2. Load the system

```lisp
(require :asdf)
(asdf:load-asd (merge-pathnames "neuralisp.asd" (uiop:getcwd)))
(asdf:load-system "neuralisp")
```

Then try it:

```lisp
(use-package :neuralisp.core.tensor)

;; [[1 2] [3 4]] @ [[5 6] [7 8]] = [[19 22] [43 50]]
(let ((a (make-tensor '(2 2) :data '(1 2 3 4)))
      (b (make-tensor '(2 2) :data '(5 6 7 8))))
  (tensor-ref (tensor-matmul a b) 0 0))          ; => 19.0d0

;; y = x * x, so dy/dx = 2x
(let* ((x (neuralisp.core.autograd:create-variable (make-tensor '(2) :data '(3 4))))
       (y (neuralisp.core.autograd:variable-multiply x x)))
  (neuralisp.core.autograd:backward y)
  (tensor-data (neuralisp.core.autograd:variable-gradient x)))   ; => #(6.0d0 8.0d0)
```

`neuralisp.core.autograd` exports a symbol named `variable`, which collides with `cl:variable`.  Refer to it
package-qualified as above, or import it deliberately with
`(:shadowing-import-from :neuralisp.core.autograd #:variable)`.

### 3. Run the examples

```bash
sbcl --script examples/simple-mlp.lisp       # uses the tensor core
sbcl --script examples/sequence-model.lisp   # standalone sketch
sbcl --script examples/cognitive-loop.lisp   # standalone sketch
```

Each script's header comment states the output it should produce, and the smoke suite checks that it still does.

### 4. Run the tests

```bash
sbcl --script tests/run-tests.lisp   # unit tests only
./tests/run-smoke.sh                 # unit tests, plus each example's output
```

Both exit non-zero on failure.  The CI workflow in [`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs
`run-smoke.sh` on GitHub Actions.

## Documentation

Describing what is implemented — every code snippet in these has been run against the current code:

- [`docs/tensor.md`](docs/tensor.md) — the tensor API, including what is *not* supported (broadcasting, rank > 2 axis
  reductions).
- [`docs/autograd.md`](docs/autograd.md) — the autograd API and how to define a new differentiable operation.
- [`docs/internals/tensor-autograd.md`](docs/internals/tensor-autograd.md) — the storage model, the backward
  traversal, and why it is ordered the way it is.

Describing intent, not current behaviour:

- [`docs/primitives/neural-primitives.md`](docs/primitives/neural-primitives.md) — planned differentiable building
  blocks.
- [`docs/cognition/cognitive-modules.md`](docs/cognition/cognitive-modules.md) — how cognitive agents would be composed
  once the primitives exist.
- [`docs/manifesto.md`](docs/manifesto.md) — the long-term research manifesto behind the roadmap.

`docs/getting_started.md`, `docs/layers.md`, `docs/losses.md`, `docs/optimizers.md`, `docs/activations.md`, and
`docs/transformers.md` are empty placeholder files, as are most modules under `src/` and `tests/`.

## Contributing

Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) for coding standards, documentation expectations, and workflow
requirements.  The high-level roadmap in [`ROADMAP.md`](ROADMAP.md) and the annotated release history in
[`CHANGELOG.md`](CHANGELOG.md) show how ongoing work maps onto the manifesto phases.  Every pull request should update the
relevant entries when behaviour or developer-facing guarantees change.

## License

NeuraLisp is released under the MIT License.  See [`LICENSE`](LICENSE) for the full text.
