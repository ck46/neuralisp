;;;; Tests for the tensor core.

(defpackage :neuralisp.tests.core.tensor
  (:use :cl :neuralisp.tests.harness)
  (:import-from :neuralisp.core.tensor
                #:make-tensor #:tensor-ref #:tensor-copy #:tensor-equal-p
                #:tensor-shape #:tensor-data #:tensor-rank #:tensor-size
                #:tensor-transpose
                #:tensor-add #:tensor-subtract #:tensor-multiply #:tensor-divide
                #:tensor-matmul #:tensor-sum #:tensor-mean
                #:shape-mismatch))
(in-package :neuralisp.tests.core.tensor)

(deftest tensor-construction-from-initial-element
  (let ((tensor (make-tensor '(2 3) :initial-element 7)))
    (check-equal '(2 3) (tensor-shape tensor))
    (check-equal 2 (tensor-rank tensor))
    (check-equal 6 (tensor-size tensor))
    (check-near 7d0 (tensor-ref tensor 1 2))))

(deftest tensor-construction-from-data
  (let ((tensor (make-tensor '(2 2) :data '(1 2 3 4))))
    (check-near 1d0 (tensor-ref tensor 0 0))
    (check-near 2d0 (tensor-ref tensor 0 1))
    (check-near 3d0 (tensor-ref tensor 1 0))
    (check-near 4d0 (tensor-ref tensor 1 1))))

(deftest tensor-construction-rejects-mismatched-data
  (check-signals shape-mismatch (make-tensor '(2 2) :data '(1 2 3))))

(deftest tensor-construction-rejects-invalid-shape
  (check-signals error (make-tensor '()))
  (check-signals error (make-tensor '(2 0)))
  (check-signals error (make-tensor '(2 -1))))

(deftest tensor-ref-is-setf-able
  (let ((tensor (make-tensor '(2 2))))
    (setf (tensor-ref tensor 1 0) 5)
    (check-near 5d0 (tensor-ref tensor 1 0))
    (check-near 0d0 (tensor-ref tensor 0 1))))

(deftest tensor-ref-bounds-are-checked
  (let ((tensor (make-tensor '(2 2))))
    (check-signals error (tensor-ref tensor 2 0))
    (check-signals error (tensor-ref tensor 0))))

(deftest tensor-copy-does-not-share-storage
  (let* ((original (make-tensor '(2 2) :initial-element 1))
         (duplicate (tensor-copy original)))
    (setf (tensor-ref duplicate 0 0) 9)
    (check-near 1d0 (tensor-ref original 0 0))
    (check-near 9d0 (tensor-ref duplicate 0 0))))

(deftest tensor-add-sums-elementwise
  (let ((a (make-tensor '(2 2) :initial-element 1))
        (b (make-tensor '(2 2) :initial-element 2)))
    (check (tensor-equal-p (make-tensor '(2 2) :initial-element 3) (tensor-add a b)))))

(deftest tensor-subtract-differences-elementwise
  (let ((a (make-tensor '(2 2) :initial-element 4))
        (b (make-tensor '(2 2) :initial-element 1)))
    (check (tensor-equal-p (make-tensor '(2 2) :initial-element 3) (tensor-subtract a b)))))

(deftest tensor-multiply-is-elementwise-not-matrix-product
  (let ((a (make-tensor '(2 2) :data '(1 2 3 4)))
        (b (make-tensor '(2 2) :data '(5 6 7 8))))
    (check (tensor-equal-p (make-tensor '(2 2) :data '(5 12 21 32))
                           (tensor-multiply a b)))))

(deftest tensor-divide-quotients-elementwise
  (let ((a (make-tensor '(2 2) :initial-element 6))
        (b (make-tensor '(2 2) :initial-element 2)))
    (check (tensor-equal-p (make-tensor '(2 2) :initial-element 3) (tensor-divide a b)))))

(deftest elementwise-operations-reject-shape-mismatch
  (let ((a (make-tensor '(2 2)))
        (b (make-tensor '(2 3))))
    (check-signals shape-mismatch (tensor-add a b))
    (check-signals shape-mismatch (tensor-multiply a b))))

(deftest tensor-transpose-swaps-axes
  (let ((tensor (make-tensor '(2 3) :data '(1 2 3 4 5 6))))
    (check (tensor-equal-p (make-tensor '(3 2) :data '(1 4 2 5 3 6))
                           (tensor-transpose tensor)))))

(deftest tensor-matmul-computes-the-matrix-product
  ;; [[1 2] [3 4]] @ [[5 6] [7 8]] = [[19 22] [43 50]]
  (let ((a (make-tensor '(2 2) :data '(1 2 3 4)))
        (b (make-tensor '(2 2) :data '(5 6 7 8))))
    (check (tensor-equal-p (make-tensor '(2 2) :data '(19 22 43 50))
                           (tensor-matmul a b)))))

(deftest tensor-matmul-handles-non-square-operands
  ;; (2x3) @ (3x2) = (2x2)
  (let ((a (make-tensor '(2 3) :data '(1 2 3 4 5 6)))
        (b (make-tensor '(3 2) :data '(7 8 9 10 11 12))))
    (check (tensor-equal-p (make-tensor '(2 2) :data '(58 64 139 154))
                           (tensor-matmul a b)))))

(deftest tensor-matmul-transposes-operands-on-request
  (let ((a (make-tensor '(3 2) :data '(1 4 2 5 3 6)))   ; transpose of (2x3) 1..6
        (b (make-tensor '(3 2) :data '(7 8 9 10 11 12))))
    (check (tensor-equal-p (make-tensor '(2 2) :data '(58 64 139 154))
                           (tensor-matmul a b :transpose-a t)))))

(deftest tensor-matmul-rejects-incompatible-inner-dimensions
  (let ((a (make-tensor '(2 3)))
        (b (make-tensor '(2 3))))
    (check-signals shape-mismatch (tensor-matmul a b))))

(deftest tensor-sum-reduces-the-whole-tensor
  (check-near 8d0 (tensor-sum (make-tensor '(2 2) :initial-element 2))))

(deftest tensor-mean-reduces-the-whole-tensor
  (check-near 4d0 (tensor-mean (make-tensor '(2 2) :initial-element 4)))
  (check-near 2.5d0 (tensor-mean (make-tensor '(2 2) :data '(1 2 3 4)))))

(deftest tensor-sum-reduces-along-an-axis
  (let ((tensor (make-tensor '(2 3) :data '(1 2 3 4 5 6))))
    (check (tensor-equal-p (make-tensor '(3) :data '(5 7 9))
                           (tensor-sum tensor :axis 0)))
    (check (tensor-equal-p (make-tensor '(2) :data '(6 15))
                           (tensor-sum tensor :axis 1)))))

(deftest tensor-sum-keepdims-preserves-rank
  (let ((tensor (make-tensor '(2 3) :data '(1 2 3 4 5 6))))
    (check-equal '(1 3) (tensor-shape (tensor-sum tensor :axis 0 :keepdims t)))
    (check-equal '(2 1) (tensor-shape (tensor-sum tensor :axis 1 :keepdims t)))))

(deftest tensor-mean-reduces-along-an-axis
  (let ((tensor (make-tensor '(2 3) :data '(1 2 3 4 5 6))))
    (check (tensor-equal-p (make-tensor '(3) :data '(2.5 3.5 4.5))
                           (tensor-mean tensor :axis 0)))))

(deftest axis-reductions-reject-unsupported-input
  (check-signals error (tensor-sum (make-tensor '(2 2 2)) :axis 0))
  (check-signals error (tensor-sum (make-tensor '(2 2)) :axis 2)))
