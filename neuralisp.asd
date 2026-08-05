;;;; ASDF system definition for NeuraLisp.
;;;;
;;;; Both systems are dependency-free, so they load and test against a bare
;;;; ANSI Common Lisp -- no Quicklisp required.  Only the modules that contain
;;;; working code are listed; the empty placeholder files under src/ and tests/
;;;; are deliberately left out until they have something in them.

(defsystem "neuralisp"
  :description "An experimental neural computing environment for Common Lisp."
  :author "ck46"
  :license "MIT"
  :version "0.0.1"
  :pathname "src/"
  :components
  ((:module "core"
    :components ((:file "tensor")
                 (:file "gpu" :depends-on ("tensor"))
                 (:file "autograd" :depends-on ("tensor")))))
  :in-order-to ((test-op (test-op "neuralisp/tests"))))

(defsystem "neuralisp/tests"
  :description "Test suite for NeuraLisp."
  :depends-on ("neuralisp")
  :pathname "tests/"
  :components
  ((:file "harness")
   (:module "core"
    :depends-on ("harness")
    :components ((:file "test_tensor")
                 (:file "test_gpu")
                 (:file "test_autograd"))))
  :perform (test-op (operation component)
             (declare (ignore operation component))
             (unless (uiop:symbol-call :neuralisp.tests.harness '#:run-all-tests)
               (error "NeuraLisp test suite failed."))))
