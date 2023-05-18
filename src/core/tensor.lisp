(defpackage :neuralisp.core.tensor
  (:use :cl)
  (:import-from :magicl :matrix :eql :addf :subf :mult :divf
                            :transpose :dot :sum :mean)
  (:export :tensor :tensor-data :tensor-shape :make-tensor
           :tensor-add :tensor-subtract :tensor-multiply :tensor-divide
           :tensor-matmul :tensor-sum :tensor-mean))
(in-package :neuralisp.core.tensor)

(defclass tensor ()
  ((data :initarg :data
         :accessor tensor-data
         :type magicl:matrix
         :documentation "N-dimensional array holding the tensor's data.")
   (shape :initarg :shape
          :accessor tensor-shape
          :type list
          :documentation "List of integers representing the tensor's shape.")
   (gpu-pointer :initform nil
                :accessor tensor-gpu-pointer
                :type (or null cl-cuda.buffer:cublas-device-pointer)
                :documentation "Pointer to tensor's data on GPU memory.")))

(defun make-tensor (shape &key (initial-element 0) (on-gpu nil))
  "Create a new tensor from input 'shape' and initialize it with 'initial-element'.
   Optionally move the tensor to GPU memory if 'on-gpu' is true."
  (let ((tensor (make-instance 'tensor
                               :data (magicl:const initial-element shape :layout :row-major)
                               :shape shape)))
    (when on-gpu
      (move-to-gpu tensor))
    tensor))

; Basic tensor operations

(defun tensor-add (tensor-a tensor-b)
  "Compute the element-wise addition of two tensors."
  (let ((res (magicl:copy-matrix (tensor-data tensor-a))))
    (magicl:addf res (tensor-data tensor-b))
    (make-tensor (tensor-shape tensor-a) :data res)))

(defun tensor-subtract (tensor-a tensor-b)
  "Compute the element-wise subtraction of two tensors."
  (let ((res (magicl:copy-matrix (tensor-data tensor-a))))
    (magicl:subf res (tensor-data tensor-b))
    (make-tensor (tensor-shape tensor-a) :data res)))

(defun tensor-multiply (tensor-a tensor-b)
  "Compute the element-wise multiplication of two tensors."
  (let ((res (magicl:copy-matrix (tensor-data tensor-a))))
    (magicl:mult ".*" res (tensor-data tensor-b))
    (make-tensor (tensor-shape tensor-a) :data res)))

(defun tensor-divide (tensor-a tensor-b)
  "Compute the element-wise division of two tensors."
  (let ((res (magicl:copy-matrix (tensor-data tensor-a))))
    (magicl:divf ".*" res (tensor-data tensor-b))
    (make-tensor (tensor-shape tensor-a) :data res)))

; Matrix multiplication and reduction functions

(defun tensor-matmul (tensor-a tensor-b &key (transpose-a nil) (transpose-b nil))
  "Compute the matrix multiplication of two tensors."
  (let* ((a (if transpose-a (magicl:transpose (tensor-data tensor-a)) (tensor-data tensor-a)))
         (b (if transpose-b (magicl:transpose (tensor-data tensor-b)) (tensor-data tensor-b)))
         (res (magicl:mult a b)))
    (make-tensor (list (magicl:matrix-rows res) (magicl:matrix-cols res)) :data res)))

(defun tensor-sum (tensor &key (axis nil) (keepdims nil))
  "Compute the sum of tensor elements along the specified axis or axes."
  (let ((sum (magicl:sum (tensor-data tensor) :axis axis :keepdims keepdims)))
    (if keepdims
        (make-tensor (magicl:matrix-shape sum) :data sum)
        sum))) ; return scalar

(defun tensor-mean (tensor &key (axis nil) (keepdims nil))
  "Compute the mean of tensor elements along the specified axis or axes."
  (let ((mean (magicl:mean (tensor-data tensor) :axis axis :keepdims keepdims)))
    (if keepdims
        (make-tensor (magicl:matrix-shape mean) :data mean)
        mean))) ; return scalar