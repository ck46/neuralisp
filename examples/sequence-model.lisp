#!/usr/bin/env sbcl --script

;;; Minimal recurrent sequence model demonstration.
;;; Run with: sbcl --script examples/sequence-model.lisp
;;; Expected output:
;;;   Time step 0 -> state 0.050, output 0.050
;;;   Time step 1 -> state 0.129, output 0.129
;;;   Time step 2 -> state 0.224, output 0.224
;;;   Time step 3 -> state 0.322, output 0.322

(defun step-rnn (state input weight recurrent-weight bias)
  "Single RNN step using tanh activation."
  (let* ((pre-activation (+ (* weight input)
                            (* recurrent-weight state)
                            bias))
         (new-state (tanh pre-activation)))
    new-state))

(let* ((inputs '(0.1 0.2 0.3 0.4))
       (state 0.0)
       (weight 0.5)
       (recurrent-weight 0.6)
       (bias 0.0))
  (loop for value in inputs
        for time from 0
        do (setf state (step-rnn state value weight recurrent-weight bias))
           (format t "Time step ~d -> state ~,3f, output ~,3f~%"
                   time state state)))
