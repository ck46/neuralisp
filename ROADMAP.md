# Roadmap

This roadmap maps upcoming work to the manifesto phases defined in [`docs/manifesto.md`](docs/manifesto.md).  Each table
tracks the status of the major deliverables and links to the documentation or examples that showcase current progress.

## Phase 0 – Foundational Infrastructure (In progress)

| Deliverable | Status | Notes |
|-------------|--------|-------|
| Loadable system | ✅ Complete | [`neuralisp.asd`](neuralisp.asd) defines `neuralisp` and `neuralisp/tests`; no external dependencies |
| Tensor core | ✅ Complete | [`src/core/tensor.lisp`](src/core/tensor.lisp) — arithmetic, matmul, reductions.  No broadcasting; axis reductions are rank-2 only |
| Reverse-mode autograd | ✅ Complete | [`src/core/autograd.lisp`](src/core/autograd.lisp) — only `variable-add` and `variable-multiply` are differentiable so far |
| Tensor/autograd documentation | ✅ Complete | See [`docs/internals/tensor-autograd.md`](docs/internals/tensor-autograd.md) |
| Example gallery | ✅ Complete | [`examples/`](examples) ships three runnable scenarios; one uses the library |
| Contributor workflow | ✅ Complete | [`CONTRIBUTING.md`](CONTRIBUTING.md) + CI running unit tests and example output |
| GPU integration | ⏳ Planned | Placement interface exists in [`src/core/gpu.lisp`](src/core/gpu.lisp); **no device backend** — `move-to-gpu` signals `gpu-backend-unavailable` |

## Phase 1 – Differentiable Primitives (Planned)

| Deliverable | Status | Notes |
|-------------|--------|-------|
| Layer constructors | ⏳ Planned | Define `linear`, `convolutional`, and `recurrent` forwards/backwards |
| Activation implementations | ⏳ Planned | Fill in `src/activations/*.lisp` with numerically stable ops |
| Loss and optimiser suites | ⏳ Planned | Implement prototypes documented in [`docs/primitives/neural-primitives.md`](docs/primitives/neural-primitives.md) |
| Broadcasting | ⏳ Planned | Elementwise ops currently require identical shapes; needs a matching gradient reduction |
| Differentiable matmul | ⏳ Planned | `tensor-matmul` has no autograd wrapper yet |
| Unit tests | 🔄 Ongoing | 41 tests cover tensor, autograd, and GPU placement; extend alongside each new module |

## Phase 2 – Cognitive Routines (Planned)

| Deliverable | Status | Notes |
|-------------|--------|-------|
| Working memory module | ⏳ Planned | Follow the flow in [`docs/cognition/cognitive-modules.md`](docs/cognition/cognitive-modules.md) |
| Policy/value learners | ⏳ Planned | Requires primitives from Phase 1 |
| Sensor/effector bridges | ⏳ Planned | Documented in manifesto and illustrated in the cognitive-loop example |
| Diagnostics | ⏳ Planned | Build instrumentation around the cognitive loop trace outputs |

The roadmap is reviewed whenever a pull request lands.  Contributors should update the relevant status markers and add
links to new documentation, examples, or tests so stakeholders can track progress across phases.
