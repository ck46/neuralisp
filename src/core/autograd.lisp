;;;; Reverse-mode automatic differentiation for NeuraLisp.
;;;;
;;;; A VARIABLE wraps a tensor and remembers the variables it was computed from.
;;;; BACKWARD walks that graph in reverse topological order, so a gradient
;;;; reaching a node through several paths is summed exactly once.
;;;;
;;;; VARIABLE is shadowed: CL:VARIABLE is an external symbol of COMMON-LISP
;;;; (it is a documentation type), so defining a class by that name in a package
;;;; that uses :CL is a package-lock violation on SBCL.

(defpackage :neuralisp.core.autograd
  (:use :cl)
  (:shadow #:variable)
  (:import-from :neuralisp.core.tensor
                #:tensor #:tensor-shape #:make-tensor
                #:tensor-add #:tensor-multiply)
  (:export #:variable #:variable-value #:variable-gradient #:variable-parents
           #:variable-backward #:variable-requires-grad-p
           #:create-variable #:backward #:zero-gradient #:propagate-gradient
           #:variable-add #:variable-multiply))
(in-package :neuralisp.core.autograd)

(defclass variable ()
  ((value :initarg :value
          :reader variable-value
          :type tensor
          :documentation "The tensor this variable stands for.")
   (gradient :initarg :gradient
             :initform nil
             :accessor variable-gradient
             :type (or null tensor)
             :documentation "Accumulated gradient, or NIL when gradients are not tracked.")
   (parents :initarg :parents
            :initform '()
            :reader variable-parents
            :type list
            :documentation "Variables this one was computed from; empty for a leaf.")
   (backward-fn :initarg :backward-fn
                :initform nil
                :accessor variable-backward
                :type (or null function)
                :documentation "Called with this node's gradient to push it to PARENTS.")
   (requires-grad :initarg :requires-grad
                  :initform t
                  :reader variable-requires-grad-p
                  :documentation "Whether gradients accumulate into this variable."))
  (:documentation "A node in the autograd graph."))

(defun zeros-like (tensor)
  (make-tensor (tensor-shape tensor) :initial-element 0))

(defun create-variable (value &key (requires-grad t))
  "Wrap the tensor VALUE in a leaf variable.

When REQUIRES-GRAD, the variable starts with a zero gradient ready to
accumulate into; otherwise its gradient stays NIL."
  (make-instance 'variable
                 :value value
                 :requires-grad requires-grad
                 :gradient (when requires-grad (zeros-like value))))

(defvar *pending-gradients* nil
  "Gradients queued for the backward pass in progress, keyed by variable.

Bound only inside BACKWARD.  Keeping the in-flight gradients here rather than in
each variable's GRADIENT slot is what lets a second backward pass start from a
clean seed instead of re-propagating whatever the first one accumulated.")

(defun accumulate-gradient (var gradient)
  "Add GRADIENT into VAR's persistent, user-visible gradient."
  (when (variable-requires-grad-p var)
    (setf (variable-gradient var)
          (if (variable-gradient var)
              (tensor-add (variable-gradient var) gradient)
              gradient)))
  var)

(defun propagate-gradient (var gradient)
  "Queue GRADIENT to reach VAR later in the current backward pass.

This is what a backward function calls to hand a gradient to an operand; the
traversal in BACKWARD delivers it once every contribution has arrived."
  (when (and *pending-gradients* (variable-requires-grad-p var))
    (let ((pending (gethash var *pending-gradients*)))
      (setf (gethash var *pending-gradients*)
            (if pending (tensor-add pending gradient) gradient))))
  var)

(defun topological-order (var)
  "VAR and its ancestors, each listed before every variable it was computed from."
  (let ((visited (make-hash-table :test #'eq))
        (order '()))
    (labels ((visit (node)
               (unless (gethash node visited)
                 (setf (gethash node visited) t)
                 (mapc #'visit (variable-parents node))
                 (push node order))))
      (visit var))
    order))

(defun backward (var &optional gradient)
  "Propagate GRADIENT back from VAR through the graph that produced it.

GRADIENT defaults to a tensor of ones shaped like VAR's value.  Gradients
accumulate, so call ZERO-GRADIENT between independent backward passes."
  (let ((*pending-gradients* (make-hash-table :test #'eq)))
    (propagate-gradient var
                        (or gradient
                            (make-tensor (tensor-shape (variable-value var))
                                         :initial-element 1)))
    (dolist (node (topological-order var) var)
      (let ((node-gradient (gethash node *pending-gradients*)))
        (when node-gradient
          (accumulate-gradient node node-gradient)
          (let ((propagate (variable-backward node)))
            (when propagate
              (funcall propagate node-gradient))))))))

(defun zero-gradient (var)
  "Reset VAR's accumulated gradient to zero, returning VAR."
  (when (variable-requires-grad-p var)
    (setf (variable-gradient var) (zeros-like (variable-value var))))
  var)

(defun make-op-variable (value parents backward-fn)
  (let ((requires-grad (some #'variable-requires-grad-p parents)))
    (make-instance 'variable
                   :value value
                   :parents parents
                   :requires-grad requires-grad
                   :gradient (when requires-grad (zeros-like value))
                   :backward-fn (when requires-grad backward-fn))))

(defun variable-add (var-a var-b)
  "Elementwise sum of two variables, tracked for differentiation."
  (make-op-variable (tensor-add (variable-value var-a) (variable-value var-b))
                    (list var-a var-b)
                    (lambda (gradient)
                      (propagate-gradient var-a gradient)
                      (propagate-gradient var-b gradient))))

(defun variable-multiply (var-a var-b)
  "Elementwise product of two variables, tracked for differentiation."
  (make-op-variable (tensor-multiply (variable-value var-a) (variable-value var-b))
                    (list var-a var-b)
                    (lambda (gradient)
                      (propagate-gradient var-a
                                          (tensor-multiply gradient (variable-value var-b)))
                      (propagate-gradient var-b
                                          (tensor-multiply gradient (variable-value var-a))))))
