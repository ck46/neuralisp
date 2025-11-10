#!/usr/bin/env sbcl --script

;;; Simple two-layer perceptron forward pass.
;;; Run with: sbcl --script examples/simple-mlp.lisp
;;; Expected output:
;;;   Input vector: (0.1 0.5 0.9)
;;;   Hidden activations: 0.470 0.470
;;;   Output logits: 0.2880 0.2880

(defun dot-product (vector weights)
  (loop for w in weights
        for x in vector
        sum (* w x)))

(defun add-bias (values biases)
  (mapcar #'+ values biases))

(defun relu (values)
  (mapcar (lambda (x) (max 0 x)) values))

(defun dense-layer (inputs weight-matrix bias-vector &key (activation #'identity))
  (let* ((raw (mapcar (lambda (row) (dot-product inputs row)) weight-matrix))
         (biased (add-bias raw bias-vector)))
    (funcall activation biased)))

(let* ((input '(0.1 0.5 0.9))
       ;; 2x3 weight matrix and bias vector for the hidden layer.
       (hidden-weights '((0.4 0.3 0.2)
                         (0.4 0.3 0.2)))
       (hidden-bias '(0.1 0.1))
       (hidden (dense-layer input hidden-weights hidden-bias :activation #'relu))
       ;; Output layer (2 units, identity activation).
       (output-weights '((0.2 0.2)
                         (0.2 0.2)))
       (output-bias '(0.1 0.1))
       (output (dense-layer hidden output-weights output-bias)))
  (format t "Input vector: ~a~%" input)
  (format t "Hidden activations: ~,3f ~,3f~%" (first hidden) (second hidden))
  (format t "Output logits: ~,4f ~,4f~%" (first output) (second output)))
