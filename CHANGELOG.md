# Changelog

All notable changes to this project will be documented in this file.  The log is organised by manifesto phase so that
stakeholders can correlate technical progress with the long-term research plan.  Dates use the ISO format `YYYY-MM-DD`.

## [Unreleased]

### Phase 0 – Foundational Infrastructure (working implementation)

The core had never been executed: `src/core/tensor.lisp` could not be read without CUDA installed, every tensor
operation passed a `:data` keyword that `make-tensor` did not accept, and `src/core/autograd.lisp` called `make-tensor`
with the wrong arity.  `neuralisp.asd` was an empty file, so the system could not be loaded at all.  The smoke suite
passed because it checked that markdown files existed and ran three examples that imported nothing from `src/`.

- **Added a real `neuralisp.asd`** defining the `neuralisp` and `neuralisp/tests` systems.  Both are dependency-free and
  load on a bare SBCL; ASDF ships with SBCL, so Quicklisp is no longer required.
- **Rewrote the tensor core** on plain ANSI Common Lisp arrays instead of `magicl`.  A tensor is now a flat row-major
  `double-float` vector plus a shape.  `make-tensor` accepts `:data`; added `tensor-ref` (`setf`-able), `tensor-copy`,
  `tensor-equal-p`, `tensor-transpose`, and a `shape-mismatch` condition.  Elementwise operations, matrix multiply, and
  full/axis reductions all work and are tested.
- **Replaced the autograd stub with working reverse-mode differentiation.**  Variables record their parents; `backward`
  walks the graph in reverse topological order, holding in-flight gradients per pass so repeated passes accumulate
  correctly.  Added `variable-add`, `variable-multiply`, and `propagate-gradient` for defining new operations.  Removed
  `partial-grad` and `apply-partial-grad`, which combined tensors with `cl:*` and could not run.
- **Made the GPU module loadable without CUDA.**  It had two contradictory representations of device residency and
  called a `cublas:` package that does not exist.  It now defines one placement interface with a single representation;
  with no backend installed, `move-to-gpu` signals `gpu-backend-unavailable` instead of returning a tensor that merely
  claims to be on a device.  There is still no working device backend.
- **Replaced the lisp-unit test file with a dependency-free harness** (`tests/harness.lisp`) and 41 unit tests across
  tensor, autograd, and GPU placement.  `tests/run-tests.lisp` loads the system through ASDF and exits non-zero on
  failure.
- **Rewrote the smoke suite to verify something.**  It now runs the unit tests and checks that each example prints the
  output its header comment promises, instead of asserting that documentation files exist.  Verified that it fails on
  both a broken tensor operation and drifted example output.
- **Corrected documented output** in `examples/sequence-model.lisp` and `examples/cognitive-loop.lisp`, which claimed
  values the scripts do not produce, and rewrote `examples/simple-mlp.lisp` to use the tensor core rather than
  reimplementing it.
- **Brought the docs in line with the code.**  `docs/tensor.md`, `docs/autograd.md`, and
  `docs/internals/tensor-autograd.md` described the old non-working API; every code snippet in them is now executed and
  matches its stated output.

### Phase 0 – Foundational Infrastructure (documentation)
- Set up documentation architecture covering tensor/autograd internals, neural primitives, and cognitive modules.
- Replaced placeholder README with an accurate quickstart, dependency overview, and documentation index.
- Added runnable example scripts (`examples/simple-mlp.lisp`, `examples/sequence-model.lisp`, `examples/cognitive-loop.lisp`).
- Introduced contribution guidelines, roadmap alignment rules, and a smoke-test-based CI workflow.
- Published the NeuraLisp manifesto to anchor roadmap discussions.

Future releases will promote entries out of the *Unreleased* section as milestones (0.1.0, 0.2.0, etc.) are tagged.
