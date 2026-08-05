#!/usr/bin/env sbcl --script

;;; Two-layer perceptron forward pass, built on the NeuraLisp tensor core.
;;;
;;; Unlike the other two examples, this one loads the actual system, so running
;;; it exercises MAKE-TENSOR, TENSOR-MATMUL and TENSOR-ADD rather than
;;; re-implementing them.
;;;
;;; Run with: sbcl --script examples/simple-mlp.lisp
;;; Expected output:
;;;   Input (1x3): 0.100 0.500 0.900
;;;   Hidden activations (1x2), post-ReLU: 0.470 0.470
;;;   Output logits (1x2): 0.2880 0.2880

(require :asdf)

(let* ((here (uiop:pathname-directory-pathname *load-truename*))
       (root (uiop:pathname-parent-directory-pathname here)))
  (asdf:load-asd (merge-pathnames "neuralisp.asd" root))
  (handler-bind ((warning #'muffle-warning))
    (asdf:load-system "neuralisp" :verbose nil)))

(use-package :neuralisp.core.tensor)

(defun relu (tensor)
  "Elementwise max(0, x). Activations live in src/activations/, which is still
empty, so the example defines what it needs and says so."
  (let ((result (tensor-copy tensor)))
    (map-into (tensor-data result) (lambda (x) (max 0d0 x)) (tensor-data result))
    result))

(defun dense-layer (input weights bias &key (activation #'identity))
  "INPUT (1 x in) @ WEIGHTS (in x out) + BIAS (1 x out), then ACTIVATION."
  (funcall activation (tensor-add (tensor-matmul input weights) bias)))

(let* ((input (make-tensor '(1 3) :data '(0.1 0.5 0.9)))
       ;; Hidden layer: 3 inputs -> 2 units. Columns are units.
       (hidden-weights (make-tensor '(3 2) :data '(0.4 0.4
                                                   0.3 0.3
                                                   0.2 0.2)))
       (hidden-bias (make-tensor '(1 2) :initial-element 0.1))
       (hidden (dense-layer input hidden-weights hidden-bias :activation #'relu))
       ;; Output layer: 2 inputs -> 2 units, identity activation.
       (output-weights (make-tensor '(2 2) :initial-element 0.2))
       (output-bias (make-tensor '(1 2) :initial-element 0.1))
       (output (dense-layer hidden output-weights output-bias)))
  (format t "Input (1x3): ~,3f ~,3f ~,3f~%"
          (tensor-ref input 0 0) (tensor-ref input 0 1) (tensor-ref input 0 2))
  (format t "Hidden activations (1x2), post-ReLU: ~,3f ~,3f~%"
          (tensor-ref hidden 0 0) (tensor-ref hidden 0 1))
  (format t "Output logits (1x2): ~,4f ~,4f~%"
          (tensor-ref output 0 0) (tensor-ref output 0 1)))
