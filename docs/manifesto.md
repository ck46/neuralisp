# NeuraLisp Manifesto

NeuraLisp envisions a Common Lisp environment where differentiable programming and symbolic reasoning co-exist.  The
manifesto divides the journey into three phases so that contributors can orient their work and track progress through the
roadmap and changelog.

## Phase 0 – Foundational Infrastructure

*Goal:* establish reliable tensor storage, GPU hooks, and documentation tooling.

*Status:* **In progress.**  The tensor/autograd prototypes, documentation overhaul, runnable examples, and CI pipeline
added in this release all belong to this phase.  Remaining tasks include robust tensor constructors, feature-complete
GPU transfer helpers, and a comprehensive smoke test suite.

## Phase 1 – Differentiable Primitives

*Goal:* implement the reusable building blocks required for deep learning workloads (layers, activations, losses,
optimisers).

*Key deliverables:*

- Layer constructors that manage weight tensors and register backward functions.
- Numerically stable activation and loss operators.
- Optimiser modules (SGD, Adam) that iterate over trainable parameters.
- Extensive unit tests validating CPU and GPU parity.

## Phase 2 – Cognitive Routines

*Goal:* compose differentiable primitives with symbolic loops to create autonomous cognitive agents.

*Key deliverables:*

- Memory and attention modules for representing internal state.
- Policy/value learners that drive decision-making.
- Sensor and effector APIs for integrating with external environments.
- Diagnostic tooling for visualising cognition loops in real time.

Progress across these phases is summarised in [`ROADMAP.md`](../ROADMAP.md) and annotated release-by-release in
[`CHANGELOG.md`](../CHANGELOG.md).  Each pull request should map its scope to at least one phase to keep the community
aligned on long-term goals.
