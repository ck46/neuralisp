(defpackage :neuralisp.tests.core.tensor
  (:use :cl
        :lisp-unit)
  (:import-from :neuralisp.core.tensor
                :make-tensor :tensor-add :tensor-subtract :tensor-multiply
                :tensor-divide :tensor-matmul :tensor-sum :tensor-mean
                :tensor-data :tensor-shape))
(in-package :neuralisp.tests.core.tensor)

(defun same-shape-and-equal-data (tensor-a tensor-b)
  (and (equal (tensor-shape tensor-a) (tensor-shape tensor-b))
       (magicl:matrix-equalp (tensor-data tensor-a) (tensor-data tensor-b))))

(define-test test-tensor-add
  (let ((a (make-tensor '(2 2) :initial-element 1))
        (b (make-tensor '(2 2) :initial-element 2))
        (expected (make-tensor '(2 2) :initial-element 3)))
    (assert-true (same-shape-and-equal-data (tensor-add a b) expected))))

(define-test test-tensor-subtract
  (let ((a (make-tensor '(2 2) :initial-element 4))
        (b (make-tensor '(2 2) :initial-element 1))
        (expected (make-tensor '(2 2) :initial-element 3)))
    (assert-true (same-shape-and-equal-data (tensor-subtract a b) expected))))

(define-test test-tensor-multiply
  (let ((a (make-tensor '(2 2) :initial-element 2))
        (b (make-tensor '(2 2) :initial-element 3))
        (expected (make-tensor '(2 2) :initial-element 6)))
    (assert-true (same-shape-and-equal-data (tensor-multiply a b) expected))))

(define-test test-tensor-divide
  (let ((a (make-tensor '(2 2) :initial-element 6))
        (b (make-tensor '(2 2) :initial-element 2))
        (expected (make-tensor '(2 2) :initial-element 3)))
    (assert-true (same-shape-and-equal-data (tensor-divide a b) expected))))

(define-test test-tensor-matmul
  (let ((a (make-tensor '(2 2) :initial-element 1))
        (b (make-tensor '(2 2) :initial-element 2))
        (expected (make-tensor '(2 2) :initial-element 4)))
    (assert-true (same-shape-and-equal-data (tensor-matmul a b) expected))))

(define-test test-tensor-sum
  (let ((a (make-tensor '(2 2) :initial-element 2)))
    (assert-equal (tensor-sum a) 8)))

(define-test test-tensor-mean
  (let ((a (make-tensor '(2 2) :initial-element 4)))
    (assert-equal (tensor-mean a) 4)))

;;; Load and run the tests
(lisp-unit:run-tests :suite :neuralisp.tests.core.tensor)