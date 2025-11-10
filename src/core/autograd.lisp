(defpackage :neuralisp.core.autograd
  (:use :cl)
  (:import-from :neuralisp.core.tensor
                :tensor
                :tensor-data
                :tensor-shape
                :tensor-size
                :tensor-dtype
                :make-tensor
                :tensor-add
                :tensor-subtract
                :tensor-multiply
                :tensor-divide
                :tensor-matmul
                :tensor-sum
                :tensor-mean
                :tensor-zeros-like
                :tensor-ones-like
                :tensor-broadcast-to
                :tensor-reshape)
  (:export :variable
           :variable-value
           :variable-gradient
           :variable-requires-grad-p
           :create-variable
           :variable-add
           :variable-subtract
           :variable-multiply
           :variable-divide
           :variable-matmul
           :variable-sum
           :variable-mean
           :backward
           :accumulate-gradient
           :zero-gradient))
(in-package :neuralisp.core.autograd)

;;;; Variable definition

(defclass variable ()
  ((value :initarg :value
          :accessor variable-value
          :type tensor
          :documentation "Tensor value held by the variable.")
   (gradient :initarg :gradient
             :initform nil
             :accessor variable-gradient
             :type (or null tensor)
             :documentation "Accumulated gradient tensor if gradients are required.")
   (requires-grad :initarg :requires-grad
                  :initform t
                  :accessor variable-requires-grad-p
                  :type boolean
                  :documentation "Whether gradients should be accumulated for this variable.")
   (inputs :initarg :inputs
           :initform '()
           :accessor variable-inputs
           :documentation "Input variables that produced this node.")
   (backward :initarg :backward
             :initform nil
             :accessor variable-backward
             :documentation "Closure that backpropagates gradients to inputs.")))

(defun %ensure-tensor (value &key dtype)
  (cond
    ((typep value 'tensor) value)
    ((or (numberp value) (complexp value))
     (make-tensor :data (list value)
                  :shape '()
                  :dtype dtype))
    (t (error "Cannot convert ~A to tensor." value))))

(defun %gradient-dtype (tensor)
  (case (tensor-dtype tensor)
    (:float64 :float64)
    (:float32 :float32)
    (otherwise :float64)))

(defun create-variable (value &key (requires-grad t) (on-gpu nil))
  "Create a variable from VALUE. VALUE may be a tensor or raw data."
  (let ((tensor-value (if (typep value 'tensor)
                          value
                          (make-tensor :data value)))
        (var (make-instance 'variable :requires-grad requires-grad)))
    (setf (variable-value var)
          (if on-gpu
              (make-tensor :data (tensor-data tensor-value)
                           :shape (tensor-shape tensor-value)
                           :dtype (tensor-dtype tensor-value)
                           :on-gpu t)
              tensor-value))
    var))

(defun %ensure-variable (value)
  (cond
    ((typep value 'variable) value)
    ((typep value 'tensor) (create-variable value :requires-grad nil))
    ((or (numberp value) (complexp value))
     (create-variable (%ensure-tensor value) :requires-grad nil))
    (t (error "Unsupported operand ~A for autograd operation." value))))

(defun %accumulate-or-set (var grad)
  (if (variable-gradient var)
      (setf (variable-gradient var)
            (tensor-add (variable-gradient var) grad))
      (setf (variable-gradient var) grad)))

(defun accumulate-gradient (var grad)
  "Accumulate GRAD into VAR's gradient when gradients are required."
  (when (variable-requires-grad-p var)
    (%accumulate-or-set var grad))
  grad)

(defun zero-gradient (var)
  "Reset VAR's gradient to zeros if gradients are tracked."
  (when (variable-gradient var)
    (setf (variable-gradient var)
          (tensor-zeros-like (variable-value var)
                              :dtype (%gradient-dtype (variable-value var)))))
  var)

(defun %normalize-axis (axis rank)
  (when (null axis)
    (return-from %normalize-axis nil))
  (let ((axes (if (listp axis) axis (list axis)))
        (result '()))
    (dolist (ax axes)
      (let ((normalized (if (< ax 0) (+ rank ax) ax)))
        (when (or (< normalized 0) (>= normalized rank))
          (error "Axis ~A out of bounds for rank ~A" ax rank))
        (pushnew normalized result :test #'=)))
    (sort result #'<)))

(defun %broadcast-reduction-axes (input-shape result-shape)
  (let* ((len-result (length result-shape))
         (len-input (length input-shape))
         (offset (- len-result len-input))
         (axes '()))
    (dotimes (axis len-result)
      (let* ((dim-result (nth axis result-shape))
             (dim-input (if (< axis offset)
                            1
                            (nth (- axis offset) input-shape))))
        (cond
          ((< axis offset) (push axis axes))
          ((= dim-input dim-result))
          ((= dim-input 1) (push axis axes))
          (t (error "Cannot reduce gradient of shape ~A to ~A"
                    result-shape input-shape)))))
    (sort axes #'<)))

(defun %reduce-like (grad target-shape)
  (if (equal (tensor-shape grad) target-shape)
      grad
      (let* ((axes (%broadcast-reduction-axes target-shape (tensor-shape grad))))
        (if axes
            (let ((summed (tensor-sum grad :axis axes :keepdims t)))
              (tensor-reshape summed target-shape))
            grad))))

(defun %expand-sum-gradient (grad input-shape axes keepdims)
  (let ((expanded (if (and axes (not keepdims))
                      (let ((grad-dims (copy-list (tensor-shape grad)))
                            (constructed '()))
                        (dotimes (axis (length input-shape))
                          (if (member axis axes)
                              (push 1 constructed)
                              (push (pop grad-dims) constructed)))
                        (let ((target (nreverse constructed)))
                          (if (equal target (tensor-shape grad))
                              grad
                              (tensor-reshape grad target))))
                      grad)))
    (if (equal (tensor-shape expanded) input-shape)
        expanded
        (tensor-broadcast-to expanded input-shape))))

(defun %scale-tensor (tensor scalar)
  (tensor-multiply tensor (%ensure-tensor scalar :dtype (tensor-dtype tensor))))

(defun %prepare-initial-gradient (var grad-output)
  (let* ((value (variable-value var))
         (shape (tensor-shape value))
         (dtype (%gradient-dtype value)))
    (cond
      ((null grad-output)
       (tensor-ones-like value :dtype dtype))
      ((typep grad-output 'tensor)
       (if (equal (tensor-shape grad-output) shape)
           grad-output
           (tensor-broadcast-to grad-output shape)))
      ((or (numberp grad-output) (complexp grad-output))
       (let ((scalar (%ensure-tensor grad-output :dtype dtype)))
         (if shape
             (tensor-broadcast-to scalar shape)
             scalar)))
      (t (error "Unsupported gradient output ~A" grad-output)))))

(defun %topological-order (root)
  (let ((visited (make-hash-table :test 'eq))
        (order '()))
    (labels ((visit (node)
               (unless (gethash node visited)
                 (setf (gethash node visited) t)
                 (dolist (input (variable-inputs node))
                   (when input (visit input)))
                 (push node order))))
      (visit root))
    order))

(defun backward (var &optional grad-output)
  "Run reverse-mode automatic differentiation from VAR."
  (let* ((order (%topological-order var))
         (propagation-order (reverse order)))
    (dolist (node order)
      (setf (variable-gradient node) nil))
    (setf (variable-gradient var) (%prepare-initial-gradient var grad-output))
    (dolist (node propagation-order)
      (let ((grad (variable-gradient node)))
        (when (and grad (variable-backward node))
          (funcall (variable-backward node) grad))))
    var))

;;;; Primitive operations

(defun %make-variable (value inputs requires-grad backward)
  (let ((var (make-instance 'variable :value value
                                      :inputs inputs
                                      :requires-grad requires-grad)))
    (setf (variable-backward var) backward)
    var))

(defun variable-add (a b)
  (let* ((var-a (%ensure-variable a))
         (var-b (%ensure-variable b))
         (value (tensor-add (variable-value var-a) (variable-value var-b)))
         (requires-grad (or (variable-requires-grad-p var-a)
                            (variable-requires-grad-p var-b)))
         (backward (when requires-grad
                     (lambda (grad)
                       (when (variable-requires-grad-p var-a)
                         (accumulate-gradient var-a
                                              (%reduce-like grad (tensor-shape (variable-value var-a)))))
                       (when (variable-requires-grad-p var-b)
                         (accumulate-gradient var-b
                                              (%reduce-like grad (tensor-shape (variable-value var-b)))))))))
    (%make-variable value (list var-a var-b) requires-grad backward)))

(defun variable-subtract (a b)
  (let* ((var-a (%ensure-variable a))
         (var-b (%ensure-variable b))
         (value (tensor-subtract (variable-value var-a) (variable-value var-b)))
         (requires-grad (or (variable-requires-grad-p var-a)
                            (variable-requires-grad-p var-b)))
         (backward (when requires-grad
                     (lambda (grad)
                       (when (variable-requires-grad-p var-a)
                         (accumulate-gradient var-a
                                              (%reduce-like grad (tensor-shape (variable-value var-a)))))
                       (when (variable-requires-grad-p var-b)
                         (let* ((neg (%scale-tensor grad -1))
                                (reduced (%reduce-like neg (tensor-shape (variable-value var-b)))))
                           (accumulate-gradient var-b reduced)))))))
    (%make-variable value (list var-a var-b) requires-grad backward)))

(defun variable-multiply (a b)
  (let* ((var-a (%ensure-variable a))
         (var-b (%ensure-variable b))
         (value (tensor-multiply (variable-value var-a) (variable-value var-b)))
         (requires-grad (or (variable-requires-grad-p var-a)
                            (variable-requires-grad-p var-b)))
         (backward (when requires-grad
                     (lambda (grad)
                       (let* ((grad-shape (tensor-shape grad))
                              (value-a (variable-value var-a))
                              (value-b (variable-value var-b))
                              (broadcast-a (tensor-broadcast-to value-b grad-shape))
                              (broadcast-b (tensor-broadcast-to value-a grad-shape)))
                         (when (variable-requires-grad-p var-a)
                           (accumulate-gradient var-a
                                                (%reduce-like (tensor-multiply grad broadcast-a)
                                                              (tensor-shape value-a))))
                         (when (variable-requires-grad-p var-b)
                           (accumulate-gradient var-b
                                                (%reduce-like (tensor-multiply grad broadcast-b)
                                                              (tensor-shape value-b)))))))))
    (%make-variable value (list var-a var-b) requires-grad backward)))

(defun variable-divide (a b)
  (let* ((var-a (%ensure-variable a))
         (var-b (%ensure-variable b))
         (value (tensor-divide (variable-value var-a) (variable-value var-b)))
         (requires-grad (or (variable-requires-grad-p var-a)
                            (variable-requires-grad-p var-b)))
         (backward (when requires-grad
                     (lambda (grad)
                       (let* ((grad-shape (tensor-shape grad))
                              (value-a (variable-value var-a))
                              (value-b (variable-value var-b))
                              (broadcast-a (tensor-broadcast-to value-a grad-shape))
                              (broadcast-b (tensor-broadcast-to value-b grad-shape)))
                         (when (variable-requires-grad-p var-a)
                           (accumulate-gradient var-a
                                                (%reduce-like (tensor-divide grad broadcast-b)
                                                              (tensor-shape value-a))))
                         (when (variable-requires-grad-p var-b)
                           (let* ((numerator (tensor-multiply grad broadcast-a))
                                  (denominator (tensor-multiply broadcast-b broadcast-b))
                                  (quotient (tensor-divide numerator denominator))
                                  (negated (%scale-tensor quotient -1)))
                             (accumulate-gradient var-b
                                                  (%reduce-like negated
                                                                (tensor-shape value-b)))))))))
    (%make-variable value (list var-a var-b) requires-grad backward)))

(defun variable-matmul (a b &key (transpose-a nil) (transpose-b nil))
  (when (or transpose-a transpose-b)
    (error "Autograd matmul does not support transposed inputs."))
  (let* ((var-a (%ensure-variable a))
         (var-b (%ensure-variable b))
         (value (tensor-matmul (variable-value var-a) (variable-value var-b)))
         (requires-grad (or (variable-requires-grad-p var-a)
                            (variable-requires-grad-p var-b)))
         (backward (when requires-grad
                     (lambda (grad)
                       (when (variable-requires-grad-p var-a)
                         (let ((grad-a (tensor-matmul grad (variable-value var-b) :transpose-b t)))
                           (accumulate-gradient var-a grad-a)))
                       (when (variable-requires-grad-p var-b)
                         (let ((grad-b (tensor-matmul (variable-value var-a) grad :transpose-a t)))
                           (accumulate-gradient var-b grad-b)))))))
    (%make-variable value (list var-a var-b) requires-grad backward)))

(defun variable-sum (a &key axis keepdims)
  (let* ((var-a (%ensure-variable a))
         (sum (tensor-sum (variable-value var-a) :axis axis :keepdims keepdims))
         (value (if (typep sum 'tensor)
                    sum
                    (%ensure-tensor sum :dtype (tensor-dtype (variable-value var-a)))))
         (requires-grad (variable-requires-grad-p var-a))
         (rank (length (tensor-shape (variable-value var-a))))
         (axes (%normalize-axis axis rank))
         (backward (when requires-grad
                     (lambda (grad)
                       (let* ((grad-tensor (if (typep grad 'tensor)
                                               grad
                                               (%ensure-tensor grad :dtype (%gradient-dtype (variable-value var-a)))))
                              (expanded (%expand-sum-gradient grad-tensor
                                                             (tensor-shape (variable-value var-a))
                                                             axes keepdims)))
                         (accumulate-gradient var-a expanded))))))
    (%make-variable value (list var-a) requires-grad backward)))

(defun variable-mean (a &key axis keepdims)
  (let* ((var-a (%ensure-variable a))
         (mean (tensor-mean (variable-value var-a) :axis axis :keepdims keepdims))
         (value (if (typep mean 'tensor)
                    mean
                    (%ensure-tensor mean :dtype (%gradient-dtype (variable-value var-a)))))
         (requires-grad (variable-requires-grad-p var-a))
         (shape (tensor-shape (variable-value var-a)))
         (rank (length shape))
         (axes (%normalize-axis axis rank))
         (count (if axes
                    (reduce #'* (mapcar (lambda (ax) (nth ax shape)) axes))
                    (tensor-size (variable-value var-a))))
         (backward (when requires-grad
                     (lambda (grad)
                       (let* ((grad-tensor (if (typep grad 'tensor)
                                               grad
                                               (%ensure-tensor grad :dtype (%gradient-dtype (variable-value var-a)))))
                              (expanded (%expand-sum-gradient grad-tensor shape axes keepdims))
                              (scaled (%scale-tensor expanded (/ 1.0 count))))
                         (accumulate-gradient var-a scaled))))))
    (%make-variable value (list var-a) requires-grad backward)))
