#!/usr/bin/env sbcl --script

;;; Cognitive loop scenario sketch.
;;; Run with: sbcl --script examples/cognitive-loop.lisp
;;; Expected output:
;;;   Normalised sensors: (0.2 0.4 0.8)
;;;   Working memory after update: 0.14 0.28 0.50
;;;   Selected action: TRACK-TARGET
;;;   Confidence score: 0.376

(defun normalise-sensors (raw-values)
  "Scale raw readings into the [0,1] band given a 0-10 calibration."
  (mapcar (lambda (x) (/ x 10.0)) raw-values))

(defun update-working-memory (memory sensors)
  "Blend the previous memory with the new sensor reading."
  (mapcar (lambda (old new)
            (+ (* 0.6 old) (* 0.4 new)))
          memory sensors))

(defun score-actions (memory)
  "Score symbolic actions using a handcrafted heuristic."
  (let* ((align-score (+ (* 0.5 (first memory))
                         (* 0.3 (second memory))
                         (* 0.2 (third memory))))
         (evade-score (+ (* 0.2 (first memory))
                         (* 0.4 (second memory))
                         (* 0.4 (third memory))))
         (track-score (+ (* 0.1 (first memory))
                         (* 0.4 (second memory))
                         (* 0.5 (third memory)))))
    `((hold-position . ,align-score)
      (evade . ,evade-score)
      (track-target . ,track-score))))

(let* ((raw-sensors '(2 4 8))
       (normalised (normalise-sensors raw-sensors))
       (previous-memory '(0.1 0.2 0.3))
       (memory (update-working-memory previous-memory normalised))
       (action-scores (score-actions memory))
       (decision (car (sort action-scores #'> :key #'cdr))))
  (format t "Normalised sensors: ~a~%" normalised)
  (format t "Working memory after update: ~,2f ~,2f ~,2f~%"
          (first memory) (second memory) (third memory))
  (format t "Selected action: ~a~%" (car decision))
  (format t "Confidence score: ~,3f~%" (cdr decision)))
