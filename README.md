# NeuraLisp

NeuraLisp is an experimental neural computing environment for Common Lisp.  The current codebase focuses on
foundational tensor structures, automatic differentiation scaffolding, and the research manifesto that guides the
future cognitive roadmap.  Many higher-level layers, optimisers, and cognitive agents are still stubs, but the
supporting infrastructure—documentation, examples, and contributor workflow—is now in place so that the community can
iterate safely.

## Project highlights

- **Tensor core prototypes** implemented in [`src/core/tensor.lisp`](src/core/tensor.lisp) for constructing tensors,
  moving data between CPU/GPU backends, and performing elementary arithmetic.
- **Autograd scaffolding** in [`src/core/autograd.lisp`](src/core/autograd.lisp) outlining differentiable variables and
  gradient accumulation primitives for future optimisation work.
- **GPU hooks** via [`src/core/gpu.lisp`](src/core/gpu.lisp) demonstrating how CUDA bindings will be integrated (the
  module currently targets `cl-cuda` and is optional during development).
- **Living manifesto and roadmap** that document the long-term vision and the current development phase.
- **Runnable example scripts** under [`examples/`](examples) that illustrate a minimal MLP forward pass, a symbolic
  sequence model sketch, and a cognitive control loop narrative, all instrumented with comments and expected output.
- **Automated smoke tests** and contribution guidelines that keep documentation, examples, and roadmap updates aligned.

## Quickstart

### 1. Install dependencies

| Dependency | Purpose | Notes |
|------------|---------|-------|
| [SBCL](https://www.sbcl.org/) (or another ANSI Common Lisp) | Runs the NeuraLisp source and examples | Tested with SBCL ≥ 2.3 |
| [Quicklisp](https://www.quicklisp.org/beta/) | Manages third-party libraries | Required to pull `magicl` and other math deps |
| [`magicl`](https://github.com/quil-lang/magicl) | Dense linear algebra backend | Load through Quicklisp (`(ql:quickload :magicl)`) |
| [`cl-cuda`](https://github.com/takagi/cl-cuda) *(optional)* | CUDA bindings for GPU experiments | Only needed if you intend to evaluate `neuralisp.core.gpu` |

Clone the repository and register the project directory with ASDF (Quicklisp does this automatically when the repo lives
under `~/quicklisp/local-projects/`):

```bash
git clone https://github.com/yourusername/neuralisp.git
cd neuralisp
```

### 2. Load the core packages

From an SBCL/Quicklisp REPL:

```lisp
(ql:quickload :magicl)         ; core tensor backend
(load "src/core/tensor.lisp")
(load "src/core/autograd.lisp")
#+cl-cuda (load "src/core/gpu.lisp")
```

If CUDA is unavailable you can skip the GPU module—the tensor and autograd packages do not require it yet.

### 3. Run the examples

Each example is a standalone script that prints its own expected output for quick verification:

```bash
sbcl --script examples/simple-mlp.lisp
sbcl --script examples/sequence-model.lisp
sbcl --script examples/cognitive-loop.lisp
```

Refer to the inline comments in each script for an explanation of the computation that is being demonstrated.

### 4. Execute the smoke tests (optional)

The automated smoke suite ensures that documentation and examples stay synchronised.  Run it locally before opening a
pull request:

```bash
./tests/run-smoke.sh
```

The CI workflow in [`.github/workflows/ci.yml`](.github/workflows/ci.yml) executes the same command on GitHub Actions.

## Documentation

The `docs/` directory is organised by topic:

- [`docs/internals/tensor-autograd.md`](docs/internals/tensor-autograd.md) dives into the tensor storage model and the
  current automatic differentiation pipeline with architecture diagrams.
- [`docs/primitives/neural-primitives.md`](docs/primitives/neural-primitives.md) catalogues the differentiable building
  blocks that exist today and those planned for the next phase.
- [`docs/cognition/cognitive-modules.md`](docs/cognition/cognitive-modules.md) describes how higher-level cognitive
  agents will be composed once the primitives mature, complete with flow diagrams and reference code snippets.
- [`docs/manifesto.md`](docs/manifesto.md) articulates the long-term research manifesto that informs the changelog and
  roadmap.

Start with [`docs/getting_started.md`](docs/getting_started.md) for a lighter introduction, then follow the cross-links
into the detailed internals.

## Contributing

Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) for coding standards, documentation expectations, and workflow
requirements.  The high-level roadmap in [`ROADMAP.md`](ROADMAP.md) and the annotated release history in
[`CHANGELOG.md`](CHANGELOG.md) show how ongoing work maps onto the manifesto phases.  Every pull request should update the
relevant entries when behaviour or developer-facing guarantees change.

## License

NeuraLisp is released under the MIT License.  See [`LICENSE`](LICENSE) for the full text.
