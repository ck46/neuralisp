# Roadmap

This roadmap maps upcoming work to the manifesto phases defined in [`docs/manifesto.md`](docs/manifesto.md).  Each table
tracks the status of the major deliverables and links to the documentation or examples that showcase current progress.

## Phase 0 – Foundational Infrastructure (In progress)

| Deliverable | Status | Notes |
|-------------|--------|-------|
| Tensor/autograd documentation | ✅ Complete | See [`docs/internals/tensor-autograd.md`](docs/internals/tensor-autograd.md) |
| Example gallery | ✅ Complete | [`examples/`](examples) now ships three runnable scenarios |
| Contributor workflow | ✅ Complete | [`CONTRIBUTING.md`](CONTRIBUTING.md) + CI smoke tests |
| GPU integration plan | ⏳ Planned | Requires optional dependency management and runtime guards |

## Phase 1 – Differentiable Primitives (Planned)

| Deliverable | Status | Notes |
|-------------|--------|-------|
| Layer constructors | ⏳ Planned | Define `linear`, `convolutional`, and `recurrent` forwards/backwards |
| Activation implementations | ⏳ Planned | Fill in `src/activations/*.lisp` with numerically stable ops |
| Loss and optimiser suites | ⏳ Planned | Implement prototypes documented in [`docs/primitives/neural-primitives.md`](docs/primitives/neural-primitives.md) |
| Unit tests | ⏳ Planned | Extend `tests/` with CPU/GPU parity checks |

## Phase 2 – Cognitive Routines (Planned)

| Deliverable | Status | Notes |
|-------------|--------|-------|
| Working memory module | ⏳ Planned | Follow the flow in [`docs/cognition/cognitive-modules.md`](docs/cognition/cognitive-modules.md) |
| Policy/value learners | ⏳ Planned | Requires primitives from Phase 1 |
| Sensor/effector bridges | ⏳ Planned | Documented in manifesto and illustrated in the cognitive-loop example |
| Diagnostics | ⏳ Planned | Build instrumentation around the cognitive loop trace outputs |

The roadmap is reviewed whenever a pull request lands.  Contributors should update the relevant status markers and add
links to new documentation, examples, or tests so stakeholders can track progress across phases.
