# Contributing to NeuraLisp

Thank you for considering a contribution!  NeuraLisp is still in its foundational phase, so every change should reinforce
the core tensor/autograd modules and keep the roadmap realistic.  This guide summarises the expectations for code style,
documentation, and workflow.

## Getting started

1. Fork the repository and create a feature branch.
2. Install the dependencies listed in [`README.md`](README.md) (SBCL, Quicklisp, `magicl`, and optionally `cl-cuda`).
3. Run the smoke suite before making changes to ensure your environment is wired correctly:
   ```bash
   ./tests/run-smoke.sh
   ```

## Coding standards

- **Common Lisp style.**  Follow the community conventions promoted by the SBCL project: two-space indentation, hyphenated
  symbol names, and docstrings for all public functions.
- **Packages.**  Export only the symbols that must be consumed by other modules.  Use `in-package` at the top of each file
  and keep `defpackage` forms in `*.lisp` files rather than ASDF metadata.
- **Error handling.**  Prefer signalling meaningful conditions over returning `nil`.  Wrap foreign-library calls with
  guards so that missing optional dependencies (e.g. CUDA) degrade gracefully.

## Documentation requirements

Every pull request should include the relevant documentation updates:

- Update the appropriate page in `docs/` if you change behaviour or add a new module.  The `docs/internals/` and
  `docs/primitives/` sections describe the canonical structure expected for architecture diagrams and code snippets.
- Expand the examples if you introduce new workflows.  Each example script must run via `sbcl --script` and print its
  own expected output for quick verification.
- Amend [`CHANGELOG.md`](CHANGELOG.md) and [`ROADMAP.md`](ROADMAP.md) when your change advances a manifesto phase.

## Testing

The repository currently ships with a documentation-focused smoke test:

```bash
./tests/run-smoke.sh
```

This command checks that the example scripts execute and that critical documentation files exist.  As the tensor and
autograd libraries stabilise we will extend the suite with unit tests that validate numerical correctness across CPU and
GPU backends.

The CI workflow in [`.github/workflows/ci.yml`](.github/workflows/ci.yml) must stay green.  Please run the smoke suite
locally before submitting a pull request.

## Pull request checklist

- [ ] Tests (`./tests/run-smoke.sh`) pass locally.
- [ ] Documentation in `docs/` or `examples/` reflects your change.
- [ ] `CHANGELOG.md` contains an entry for your work under the correct manifesto phase.
- [ ] `ROADMAP.md` status tables are updated if scope has shifted.
- [ ] The pull request description references the manifesto phase(s) touched.

Keeping these checkpoints in sync ensures that contributors, reviewers, and stakeholders share the same context as the
project progresses through the manifesto phases.
