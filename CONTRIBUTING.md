# Contributing to NeuraLisp

Thank you for considering a contribution!  NeuraLisp is still in its foundational phase, so every change should reinforce
the core tensor/autograd modules and keep the roadmap realistic.  This guide summarises the expectations for code style,
documentation, and workflow.

## Getting started

1. Fork the repository and create a feature branch.
2. Install SBCL.  There are no other dependencies — ASDF ships with SBCL, and the project uses no third-party libraries.
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
- Expand the examples if you introduce new workflows.  Each example script must run via `sbcl --script` and carry an
  `Expected output:` block in its header comment; the smoke suite compares the script's actual output against it, so a
  stale block fails CI.
- Code snippets in `docs/` are expected to run as written.  Paste them into a REPL and check the output before
  committing.
- Amend [`CHANGELOG.md`](CHANGELOG.md) and [`ROADMAP.md`](ROADMAP.md) when your change advances a manifesto phase.

## Testing

```bash
sbcl --script tests/run-tests.lisp   # unit tests only
./tests/run-smoke.sh                 # unit tests, plus each example's output
```

The unit tests live under `tests/` and use the small harness in [`tests/harness.lisp`](tests/harness.lisp) — no
third-party test library, so `sbcl --script` is enough.  Add tests with `deftest` and the `check`, `check-equal`,
`check-near`, and `check-signals` macros, then register the file as a component of the `neuralisp/tests` system in
[`neuralisp.asd`](neuralisp.asd).  Both commands exit non-zero on failure.

**A test must be able to fail.**  Assert on computed values, not on the existence of files or on a script merely not
crashing.  If you add a check, break the thing it covers once and confirm the suite goes red before you submit.

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
