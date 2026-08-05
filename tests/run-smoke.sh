#!/usr/bin/env bash
#
# NeuraLisp smoke suite.
#
# This checks two things that can actually fail:
#   1. The ASDF system loads and its unit tests pass.
#   2. Each example runs AND prints the output its header comment promises.
#
# It deliberately does not assert that documentation files exist -- a file
# existence check cannot fail for any reason worth a CI run, and a suite that
# only does that reports green while the library underneath is broken.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "[INFO] Loading the system and running unit tests"
sbcl --script "$ROOT/tests/run-tests.lisp"

# Pull the documented output out of an example's header comment: the ";;; "
# lines following "Expected output:", up to the first line that is not a
# comment.
extract_expected_output() {
  awk '/Expected output:/ { capturing = 1; next }
       capturing && /^;;;/  { sub(/^;;;   /, ""); print; next }
       capturing            { exit }' "$1"
}

for script in "examples/simple-mlp.lisp" \
              "examples/sequence-model.lisp" \
              "examples/cognitive-loop.lisp"; do
  echo "[INFO] Running $script"

  expected="$(extract_expected_output "$ROOT/$script")"
  if [[ -z "$expected" ]]; then
    echo "[ERROR] $script has no 'Expected output:' block to verify against" >&2
    exit 1
  fi

  # Compiler notes go to stderr; only stdout is the example's output.
  actual="$(sbcl --script "$ROOT/$script" 2>/dev/null)"

  if [[ "$actual" != "$expected" ]]; then
    echo "[ERROR] $script printed output that does not match its header comment:" >&2
    diff <(printf '%s\n' "$expected") <(printf '%s\n' "$actual") >&2 || true
    exit 1
  fi
done

echo "[INFO] Smoke suite completed successfully."
