#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

check_file() {
  local path="$1"
  if [[ ! -f "$ROOT/$path" ]]; then
    echo "[ERROR] Missing required file: $path" >&2
    exit 1
  fi
}

check_file "docs/internals/tensor-autograd.md"
check_file "docs/primitives/neural-primitives.md"
check_file "docs/cognition/cognitive-modules.md"
check_file "docs/manifesto.md"
check_file "CHANGELOG.md"
check_file "ROADMAP.md"
check_file "CONTRIBUTING.md"

# Execute example scripts to ensure they run without errors.
for script in "examples/simple-mlp.lisp" \
              "examples/sequence-model.lisp" \
              "examples/cognitive-loop.lisp"; do
  echo "[INFO] Running $script"
  sbcl --script "$ROOT/$script" >/dev/null
done

echo "[INFO] Smoke suite completed successfully."
