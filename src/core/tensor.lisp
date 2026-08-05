;;;; Core tensor type for NeuraLisp.
;;;;
;;;; This module depends on nothing outside ANSI Common Lisp so that the system
;;;; loads -- and its tests run -- against a bare SBCL with no Quicklisp.  Data
;;;; is held as a flat row-major vector of DOUBLE-FLOATs next to an explicit
;;;; shape, which keeps the storage model obvious and makes every operation here
;;;; directly testable.  An accelerated backend (magicl, BLAS, CUDA) can be
;;;; introduced later behind these same entry points.

(defpackage :neuralisp.core.tensor
  (:use :cl)
  (:export #:tensor #:tensor-data #:tensor-shape #:tensor-device
           #:tensor-gpu-pointer #:tensor-rank #:tensor-size
           #:make-tensor #:tensor-ref #:tensor-copy #:tensor-equal-p
           #:tensor-transpose
           #:tensor-add #:tensor-subtract #:tensor-multiply #:tensor-divide
           #:tensor-matmul #:tensor-sum #:tensor-mean
           #:shape-mismatch #:shape-mismatch-operation
           #:shape-mismatch-expected #:shape-mismatch-actual))
(in-package :neuralisp.core.tensor)

(deftype tensor-storage ()
  "Flat row-major backing store for a tensor."
  '(simple-array double-float (*)))

(define-condition shape-mismatch (error)
  ((operation :initarg :operation :reader shape-mismatch-operation)
   (expected :initarg :expected :reader shape-mismatch-expected)
   (actual :initarg :actual :reader shape-mismatch-actual))
  (:report (lambda (condition stream)
             (format stream "~a: expected shape ~a but got ~a."
                     (shape-mismatch-operation condition)
                     (shape-mismatch-expected condition)
                     (shape-mismatch-actual condition))))
  (:documentation "Signalled when operands do not agree on shape."))

(defclass tensor ()
  ((data :initarg :data
         :accessor tensor-data
         :type tensor-storage
         :documentation "Flat row-major vector holding the tensor's elements.")
   (shape :initarg :shape
          :accessor tensor-shape
          :type list
          :documentation "List of positive integers describing the tensor's extent.")
   (device :initarg :device
           :initform :cpu
           :accessor tensor-device
           :documentation "Where the data currently lives; :CPU or :GPU.")
   (gpu-pointer :initarg :gpu-pointer
                :initform nil
                :accessor tensor-gpu-pointer
                :documentation "Backend-specific device handle, or NIL on the CPU.
Deliberately untyped: naming a cl-cuda type here would make this file unreadable
on a machine without CUDA installed."))
  (:documentation "A dense, row-major, double-float tensor."))

(defun shape-size (shape)
  "Total number of elements described by SHAPE."
  (reduce #'* shape :initial-value 1))

(defun valid-shape-p (shape)
  (and (consp shape)
       (every (lambda (dimension) (and (integerp dimension) (plusp dimension)))
              shape)))

(defun coerce-storage (sequence)
  "Return SEQUENCE as tensor storage, without copying when it already is one."
  (if (typep sequence 'tensor-storage)
      sequence
      (let ((storage (make-array (length sequence) :element-type 'double-float)))
        (map-into storage (lambda (x) (coerce x 'double-float)) sequence)
        storage)))

(defun make-tensor (shape &key (initial-element 0) data)
  "Create a tensor of SHAPE.

With DATA, adopt that sequence as the tensor's row-major contents; its length
must equal the number of elements implied by SHAPE.  Without DATA, fill the
tensor with INITIAL-ELEMENT.  A DATA vector that is already tensor storage is
adopted rather than copied, so callers must not mutate it afterwards."
  (unless (valid-shape-p shape)
    (error "Invalid tensor shape ~s: expected a non-empty list of positive integers."
           shape))
  (let ((size (shape-size shape)))
    (make-instance 'tensor
                   :shape (copy-list shape)
                   :data (if data
                             (let ((storage (coerce-storage data)))
                               (unless (= (length storage) size)
                                 (error 'shape-mismatch
                                        :operation "MAKE-TENSOR"
                                        :expected (list size)
                                        :actual (list (length storage))))
                               storage)
                             (make-array size
                                         :element-type 'double-float
                                         :initial-element (coerce initial-element
                                                                  'double-float))))))

(defun tensor-rank (tensor)
  "Number of dimensions in TENSOR."
  (length (tensor-shape tensor)))

(defun tensor-size (tensor)
  "Total number of elements in TENSOR."
  (length (tensor-data tensor)))

(defun tensor-index (tensor subscripts)
  "Row-major offset of SUBSCRIPTS within TENSOR."
  (let ((shape (tensor-shape tensor)))
    (unless (= (length subscripts) (length shape))
      (error 'shape-mismatch
             :operation "TENSOR-REF"
             :expected shape
             :actual subscripts))
    (loop with offset = 0
          for subscript in subscripts
          for dimension in shape
          do (unless (and (integerp subscript) (< -1 subscript dimension))
               (error "Subscript ~s is out of bounds for dimension ~a." subscript dimension))
             (setf offset (+ (* offset dimension) subscript))
          finally (return offset))))

(defun tensor-ref (tensor &rest subscripts)
  "Element of TENSOR at SUBSCRIPTS."
  (aref (tensor-data tensor) (tensor-index tensor subscripts)))

(defun (setf tensor-ref) (value tensor &rest subscripts)
  (setf (aref (tensor-data tensor) (tensor-index tensor subscripts))
        (coerce value 'double-float)))

(defun tensor-copy (tensor)
  "A fresh tensor sharing no storage with TENSOR."
  (make-tensor (tensor-shape tensor) :data (copy-seq (tensor-data tensor))))

(defun tensor-equal-p (tensor-a tensor-b &key (tolerance 1d-9))
  "True when both tensors have the same shape and agree elementwise to TOLERANCE."
  (and (equal (tensor-shape tensor-a) (tensor-shape tensor-b))
       (let ((x (tensor-data tensor-a))
             (y (tensor-data tensor-b)))
         (loop for i below (length x)
               always (<= (abs (- (aref x i) (aref y i))) tolerance)))))

;;; Elementwise arithmetic

(defun elementwise (operation function tensor-a tensor-b)
  "Apply FUNCTION across two shape-compatible tensors, naming OPERATION on error."
  (unless (equal (tensor-shape tensor-a) (tensor-shape tensor-b))
    (error 'shape-mismatch
           :operation operation
           :expected (tensor-shape tensor-a)
           :actual (tensor-shape tensor-b)))
  (let* ((x (tensor-data tensor-a))
         (y (tensor-data tensor-b))
         (result (make-array (length x) :element-type 'double-float)))
    (dotimes (i (length x))
      (setf (aref result i) (funcall function (aref x i) (aref y i))))
    (make-tensor (tensor-shape tensor-a) :data result)))

(defun tensor-add (tensor-a tensor-b)
  "Elementwise sum of two tensors."
  (elementwise "TENSOR-ADD" #'+ tensor-a tensor-b))

(defun tensor-subtract (tensor-a tensor-b)
  "Elementwise difference of two tensors."
  (elementwise "TENSOR-SUBTRACT" #'- tensor-a tensor-b))

(defun tensor-multiply (tensor-a tensor-b)
  "Elementwise (Hadamard) product of two tensors."
  (elementwise "TENSOR-MULTIPLY" #'* tensor-a tensor-b))

(defun tensor-divide (tensor-a tensor-b)
  "Elementwise quotient of two tensors."
  (elementwise "TENSOR-DIVIDE" #'/ tensor-a tensor-b))

;;; Matrix operations

(defun tensor-transpose (tensor)
  "Transpose a rank-2 tensor."
  (unless (= 2 (tensor-rank tensor))
    (error "TENSOR-TRANSPOSE requires a rank-2 tensor, got rank ~a." (tensor-rank tensor)))
  (destructuring-bind (rows cols) (tensor-shape tensor)
    (let ((source (tensor-data tensor))
          (result (make-array (* rows cols) :element-type 'double-float)))
      (dotimes (r rows)
        (dotimes (c cols)
          (setf (aref result (+ (* c rows) r))
                (aref source (+ (* r cols) c)))))
      (make-tensor (list cols rows) :data result))))

(defun tensor-matmul (tensor-a tensor-b &key transpose-a transpose-b)
  "Matrix product of two rank-2 tensors, optionally transposing either operand."
  (let ((a (if transpose-a (tensor-transpose tensor-a) tensor-a))
        (b (if transpose-b (tensor-transpose tensor-b) tensor-b)))
    (unless (and (= 2 (tensor-rank a)) (= 2 (tensor-rank b)))
      (error "TENSOR-MATMUL requires rank-2 tensors, got ranks ~a and ~a."
             (tensor-rank a) (tensor-rank b)))
    (destructuring-bind (rows inner) (tensor-shape a)
      (destructuring-bind (inner-b cols) (tensor-shape b)
        (unless (= inner inner-b)
          (error 'shape-mismatch
                 :operation "TENSOR-MATMUL"
                 :expected (list inner cols)
                 :actual (tensor-shape b)))
        (let ((x (tensor-data a))
              (y (tensor-data b))
              (result (make-array (* rows cols)
                                  :element-type 'double-float
                                  :initial-element 0d0)))
          (dotimes (r rows)
            (dotimes (k inner)
              (let ((scale (aref x (+ (* r inner) k))))
                (unless (zerop scale)
                  (dotimes (c cols)
                    (incf (aref result (+ (* r cols) c))
                          (* scale (aref y (+ (* k cols) c)))))))))
          (make-tensor (list rows cols) :data result))))))

;;; Reductions
;;;
;;; AXIS NIL reduces the whole tensor and returns a DOUBLE-FLOAT scalar.  An
;;; integer AXIS reduces a rank-2 tensor along that axis and returns a tensor;
;;; KEEPDIMS decides whether the reduced axis survives with extent 1.  Reducing
;;; a single axis of a rank > 2 tensor is not implemented yet, and says so
;;; rather than returning something plausible.

(defun reduce-axis (tensor axis keepdims scale-by-count)
  (unless (= 2 (tensor-rank tensor))
    (error "Axis reductions are implemented for rank-2 tensors only, got rank ~a."
           (tensor-rank tensor)))
  (unless (member axis '(0 1))
    (error "Invalid axis ~s for a rank-2 tensor." axis))
  (destructuring-bind (rows cols) (tensor-shape tensor)
    (let* ((source (tensor-data tensor))
           (length (if (= axis 0) cols rows))
           (count (if (= axis 0) rows cols))
           (result (make-array length :element-type 'double-float :initial-element 0d0)))
      (dotimes (r rows)
        (dotimes (c cols)
          (incf (aref result (if (= axis 0) c r))
                (aref source (+ (* r cols) c)))))
      (when scale-by-count
        (dotimes (i length)
          (setf (aref result i) (/ (aref result i) count))))
      (make-tensor (cond ((not keepdims) (list length))
                         ((= axis 0) (list 1 cols))
                         (t (list rows 1)))
                   :data result))))

(defun tensor-sum (tensor &key axis keepdims)
  "Sum TENSOR entirely (returning a scalar) or along AXIS (returning a tensor)."
  (if (null axis)
      (let ((total 0d0))
        (map nil (lambda (x) (incf total x)) (tensor-data tensor))
        (if keepdims
            (make-tensor (make-list (tensor-rank tensor) :initial-element 1)
                         :data (list total))
            total))
      (reduce-axis tensor axis keepdims nil)))

(defun tensor-mean (tensor &key axis keepdims)
  "Average TENSOR entirely (returning a scalar) or along AXIS (returning a tensor)."
  (if (null axis)
      (let ((mean (/ (tensor-sum tensor) (tensor-size tensor))))
        (if keepdims
            (make-tensor (make-list (tensor-rank tensor) :initial-element 1)
                         :data (list mean))
            mean))
      (reduce-axis tensor axis keepdims t)))
