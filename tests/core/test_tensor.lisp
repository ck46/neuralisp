(defpackage :neuralisp.tests.core.tensor
  (:use :cl :fiveam)
  (:import-from :neuralisp.core.tensor
                :make-tensor
                :tensor-data
                :tensor-shape
                :tensor-size
                :tensor-dtype
                :tensor-add
                :tensor-subtract
                :tensor-multiply
                :tensor-divide
                :tensor-matmul
                :tensor-sum
                :tensor-mean))
(in-package :neuralisp.tests.core.tensor)

(def-suite tensor-suite :description "Tensor creation and arithmetic tests")
(in-suite tensor-suite)

(defun tensor->list (tensor)
  (loop for i below (tensor-size tensor)
        collect (aref (tensor-data tensor) i)))

(test make-tensor-from-sequence
  (let ((tensor (make-tensor :data '((1 2 3) (4 5 6)))))
    (is (equal (tensor-shape tensor) '(2 3)))
    (is (equal (tensor->list tensor) '(1 2 3 4 5 6)))
    (is (eql (tensor-dtype tensor) :int))))

(test make-tensor-with-initializer
  (let ((tensor (make-tensor :shape '(2 2)
                             :initializer (lambda (indices)
                                            (+ (first indices) (second indices))))))
    (is (equal (tensor->list tensor) '(0 1 1 2))))

(test broadcast-addition
  (let* ((a (make-tensor :data '((1 2 3) (4 5 6))))
         (b (make-tensor :data '(10 20 30)))
         (result (tensor-add a b)))
    (is (equal (tensor-shape result) '(2 3)))
    (is (equal (tensor->list result) '(11 22 33 14 25 36)))))

(test type-promotion
  (let* ((a (make-tensor :data '(1 2 3)))
         (b (make-tensor :data '(1.5 2.5 3.5)))
         (result (tensor-add a b)))
    (is (eql (tensor-dtype result) :float64))
    (is (every (lambda (expected actual)
                 (< (abs (- expected actual)) 1e-6))
               '(2.5 4.5 6.5)
               (tensor->list result)))))

(test tensor-matmul-basic
  (let* ((a (make-tensor :data '((1 2) (3 4))))
         (b (make-tensor :data '((5 6) (7 8))))
         (result (tensor-matmul a b)))
    (is (equal (tensor-shape result) '(2 2)))
    (is (equal (tensor->list result) '(19 22 43 50)))))

(test tensor-sum-axis
  (let* ((a (make-tensor :data '((1 2 3) (4 5 6))))
         (result (tensor-sum a :axis 0 :keepdims t)))
    (is (equal (tensor-shape result) '(1 3)))
    (is (equal (tensor->list result) '(5 7 9)))))

(test tensor-mean-reduction
  (let* ((a (make-tensor :data '((1 2) (3 4))))
         (mean-all (tensor-mean a))
         (mean-axis (tensor-mean a :axis 1 :keepdims t)))
    (is (< (abs (- mean-all 2.5)) 1e-6))
    (is (equal (tensor-shape mean-axis) '(2 1)))
    (is (every (lambda (expected actual)
                 (< (abs (- expected actual)) 1e-6))
               '(1.5 3.5)
               (tensor->list mean-axis)))))
