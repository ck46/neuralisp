(defpackage :neuralisp.tests.core.autograd
  (:use :cl :fiveam)
  (:import-from :neuralisp.core.tensor
                :make-tensor
                :tensor-data
                :tensor-shape
                :tensor-size)
  (:import-from :neuralisp.core.autograd
                :create-variable
                :variable-add
                :variable-subtract
                :variable-multiply
                :variable-divide
                :variable-matmul
                :variable-sum
                :variable-mean
                :variable-gradient
                :variable-value
                :backward
                :zero-gradient))
(in-package :neuralisp.tests.core.autograd)

(def-suite autograd-suite :description "Autograd differentiation tests")
(in-suite autograd-suite)

(defun tensor->list (tensor)
  (loop for i below (tensor-size tensor)
        collect (aref (tensor-data tensor) i)))

(test add-backward
  (let* ((a (create-variable (make-tensor :data '((1 2) (3 4)))))
         (b (create-variable (make-tensor :data '((5 6) (7 8)))))
         (c (variable-add a b)))
    (backward c)
    (is (equal (tensor->list (variable-gradient a)) '(1 1 1 1)))
    (is (equal (tensor->list (variable-gradient b)) '(1 1 1 1)))))

(test broadcast-add-backward
  (let* ((a (create-variable (make-tensor :data '((1 2 3) (4 5 6)))))
         (b (create-variable (make-tensor :data '(1 1 1))))
         (c (variable-add a b)))
    (backward c)
    (is (equal (tensor->list (variable-gradient a)) '(1 1 1 1 1 1)))
    (is (equal (tensor->list (variable-gradient b)) '(2 2 2)))))

(test multiply-backward
  (let* ((a (create-variable (make-tensor :data '(1 2 3))))
         (b (create-variable (make-tensor :data '(4 5 6))))
         (c (variable-multiply a b)))
    (backward c)
    (is (equal (tensor->list (variable-gradient a)) '(4 5 6)))
    (is (equal (tensor->list (variable-gradient b)) '(1 2 3)))))

(test divide-backward
  (let* ((a (create-variable (make-tensor :data '(2 4 6))))
         (b (create-variable (make-tensor :data '(2 2 2))))
         (c (variable-divide a b)))
    (backward c)
    (is (every (lambda (expected actual)
                 (< (abs (- expected actual)) 1e-6))
               '(0.5 0.5 0.5)
               (tensor->list (variable-gradient a))))
    (is (every (lambda (expected actual)
                 (< (abs (- expected actual)) 1e-6))
               '(-0.5 -1.0 -1.5)
               (tensor->list (variable-gradient b))))))

(test matmul-backward
  (let* ((a (create-variable (make-tensor :data '((1 2)))))
         (b (create-variable (make-tensor :data '((3) (4)))))
         (c (variable-matmul a b)))
    (backward c)
    (is (equal (tensor->list (variable-gradient a)) '(3 4)))
    (is (equal (tensor->list (variable-gradient b)) '(1 2)))))

(test sum-backward-axis
  (let* ((a (create-variable (make-tensor :data '((1 2 3) (4 5 6)))))
         (s (variable-sum a :axis 1))
         (grad (make-tensor :data '(1 1) :shape '(2))))
    (backward s grad)
    (is (equal (tensor->list (variable-gradient a)) '(1 1 1 1 1 1)))))

(test mean-backward
  (let* ((a (create-variable (make-tensor :data '((1 2) (3 4)))))
         (m (variable-mean a)))
    (backward m)
    (is (every (lambda (actual)
                 (< (abs (- actual 0.25)) 1e-6))
               (tensor->list (variable-gradient a))))
    (zero-gradient a)
    (is (every #'zerop (tensor->list (variable-gradient a))))))
