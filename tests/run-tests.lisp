;;;; Load the NeuraLisp system through ASDF and run its test suite.
;;;;
;;;; Run with: sbcl --script tests/run-tests.lisp
;;;; Exits non-zero when any test fails, so CI notices.

(require :asdf)

(let* ((here (uiop:pathname-directory-pathname *load-truename*))
       (root (uiop:pathname-parent-directory-pathname here)))
  (asdf:load-asd (merge-pathnames "neuralisp.asd" root))
  (asdf:load-system "neuralisp/tests"))

(uiop:quit (if (uiop:symbol-call :neuralisp.tests.harness '#:run-all-tests) 0 1))
