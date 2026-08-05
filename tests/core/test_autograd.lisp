;;;; Tests for reverse-mode automatic differentiation.

(defpackage :neuralisp.tests.core.autograd
  (:use :cl :neuralisp.tests.harness)
  (:import-from :neuralisp.core.tensor
                #:make-tensor #:tensor-equal-p)
  (:import-from :neuralisp.core.autograd
                #:create-variable #:variable-value #:variable-gradient
                #:variable-parents #:variable-requires-grad-p
                #:backward #:zero-gradient
                #:variable-add #:variable-multiply))
(in-package :neuralisp.tests.core.autograd)

(defun leaf (&rest elements)
  (create-variable (make-tensor (list (length elements)) :data elements)))

(deftest create-variable-starts-with-a-zero-gradient
  (let ((var (leaf 1 2 3)))
    (check (tensor-equal-p (make-tensor '(3) :initial-element 0)
                           (variable-gradient var)))
    (check-equal '() (variable-parents var))))

(deftest create-variable-without-grad-tracks-no-gradient
  (let ((var (create-variable (make-tensor '(2)) :requires-grad nil)))
    (check (null (variable-gradient var)))
    (check (not (variable-requires-grad-p var)))))

(deftest variable-add-computes-the-forward-value
  (let ((sum (variable-add (leaf 1 2) (leaf 10 20))))
    (check (tensor-equal-p (make-tensor '(2) :data '(11 22))
                           (variable-value sum)))))

(deftest variable-multiply-computes-the-forward-value
  (let ((product (variable-multiply (leaf 2 3) (leaf 4 5))))
    (check (tensor-equal-p (make-tensor '(2) :data '(8 15))
                           (variable-value product)))))

(deftest backward-through-add-routes-the-gradient-to-both-operands
  (let* ((a (leaf 1 2))
         (b (leaf 3 4))
         (sum (variable-add a b)))
    (backward sum)
    ;; d(a+b)/da = d(a+b)/db = 1
    (check (tensor-equal-p (make-tensor '(2) :initial-element 1) (variable-gradient a)))
    (check (tensor-equal-p (make-tensor '(2) :initial-element 1) (variable-gradient b)))))

(deftest backward-through-multiply-applies-the-product-rule
  (let* ((a (leaf 2 3))
         (b (leaf 5 7))
         (product (variable-multiply a b)))
    (backward product)
    ;; d(a*b)/da = b, d(a*b)/db = a
    (check (tensor-equal-p (make-tensor '(2) :data '(5 7)) (variable-gradient a)))
    (check (tensor-equal-p (make-tensor '(2) :data '(2 3)) (variable-gradient b)))))

(deftest backward-sums-gradients-arriving-by-several-paths
  ;; y = x*x, so dy/dx = 2x.  Both operands of the product are the same node, so
  ;; a correct traversal visits it once with the total incoming gradient.
  (let* ((x (leaf 3 4))
         (y (variable-multiply x x)))
    (backward y)
    (check (tensor-equal-p (make-tensor '(2) :data '(6 8)) (variable-gradient x)))))

(deftest backward-chains-through-composed-operations
  ;; z = (a + b) * c  =>  dz/da = dz/db = c, dz/dc = a + b
  (let* ((a (leaf 1 2))
         (b (leaf 3 4))
         (c (leaf 10 100))
         (z (variable-multiply (variable-add a b) c)))
    (backward z)
    (check (tensor-equal-p (make-tensor '(2) :data '(10 100)) (variable-gradient a)))
    (check (tensor-equal-p (make-tensor '(2) :data '(10 100)) (variable-gradient b)))
    (check (tensor-equal-p (make-tensor '(2) :data '(4 6)) (variable-gradient c)))))

(deftest backward-accepts-an-explicit-seed-gradient
  (let* ((a (leaf 1 2))
         (b (leaf 3 4))
         (sum (variable-add a b)))
    (backward sum (make-tensor '(2) :data '(2 5)))
    (check (tensor-equal-p (make-tensor '(2) :data '(2 5)) (variable-gradient a)))))

(deftest backward-accumulates-across-passes-until-zeroed
  (let* ((a (leaf 1 2))
         (b (leaf 3 4))
         (sum (variable-add a b)))
    (backward sum)
    (backward sum)
    (check (tensor-equal-p (make-tensor '(2) :initial-element 2) (variable-gradient a)))
    (zero-gradient a)
    (check (tensor-equal-p (make-tensor '(2) :initial-element 0) (variable-gradient a)))))

(deftest backward-skips-operands-that-do-not-require-gradients
  (let* ((a (leaf 2 3))
         (constant (create-variable (make-tensor '(2) :data '(5 7)) :requires-grad nil))
         (product (variable-multiply a constant)))
    (backward product)
    (check (tensor-equal-p (make-tensor '(2) :data '(5 7)) (variable-gradient a)))
    (check (null (variable-gradient constant)))))
