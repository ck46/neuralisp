(defpackage :neuralisp.core.autograd
  (:use :common-lisp)
  (:import-from :neuralisp.core.tensor :tensor :tensor-data :tensor-shape :make-tensor)
  (:import-from :neuralisp.core.gpu :move-to-gpu :move-to-cpu)
  (:export :variable :value :gradient :create-variable :backward :zero-gradient
           :partial-grad :apply-partial-grad))
(in-package :neuralisp.core.autograd)

(defclass variable ()
  ((value :initarg :value
          :reader variable-value
          :type tensor
          :documentation "The tensor object representing the value of the variable.")
   (gradient :initarg :gradient  
             :accessor variable-gradient
             :type (or null tensor)
             :documentation "The tensor object representing the gradient of the variable.")
   (backward :initarg :backward
             :accessor variable-backward
             :type (or null function)
             :documentation "The backward function for computing gradients.")))

(defun create-variable (value &key (requires-grad t) (on-gpu nil))
  "Create a new autograd variable with the input value and a gradient,
    and set backward to nil."
  (let ((tensor-value (if on-gpu (move-to-gpu value) value))
        (tensor-grad (when requires-grad
                       (let ((grad-tensor (make-tensor (make-array (reduce #'* (tensor-shape value))) (tensor-shape value))))
                         (if on-gpu (move-to-gpu grad-tensor) grad-tensor)))))
    (make-instance 'variable :value tensor-value
                           :gradient tensor-grad)))

(defun backward (var &optional (grad-output 1.0))
  "Computes the gradients of the variable with respect to its values,
    and accumulated gradient output."
  (when (functionp (variable-backward var))
    (funcall (variable-backward var) grad-output)))

(defun zero-gradient (var)
  "Set the gradient of the variable to zero."
  (setf (variable-gradient var) 
        (let ((zero-tensor (make-instance 'tensor :data (make-array (reduce #'* (tensor-shape (variable-value var)))) :shape (tensor-shape (variable-value var)))))
          (if (equal (tensor-data (variable-value var)) :gpu) (move-to-gpu zero-tensor) zero-tensor))))

(defun partial-grad (node-a node-b)
  "Compute the partial derivatives between two 'variable' nodes."
  (let ((prev-node node-b) (grad 1))
    (loop
      (when (eq prev-node node-a)
          (return grad))
      (setq grad (* grad (variable-gradient prev-node)))
      (setq prev-node (variable-backward prev-node)))))

(defun apply-partial-grad (node-a node-b)
  "Apply the computed partial gradients of node-b with respect to node-a to the gradients of both nodes."
  (let ((partial (partial-grad node-a node-b)))
    (setf (variable-gradient node-a) (* (variable-gradient node-a) partial))
    (setf (variable-gradient node-b) (* (variable-gradient node-b) partial))))