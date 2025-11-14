(defpackage :neuralisp.cognition.self-reflection
  (:use :cl)
  (:import-from :neuralisp.cognition.state
                :state-cycle-trace
                :state-cycle-count
                :record-trace
                :register-state-value
                :lookup-state-value
                :update-state-value)
  (:export :self-reflection-module-step))

(in-package :neuralisp.cognition.self-reflection)

(defun %strategy-from-trace (trace)
  (if (> (length trace) 3)
      :deliberate
      :reactive))

(defun self-reflection-module-step (graph)
  "Inspect the current cycle trace and adapt the control strategy accordingly.
The module stores the selected strategy and exposes the inspection summary via trace logs."
  (let* ((cycle (state-cycle-count graph))
         (slice (state-cycle-trace graph cycle))
         (strategy (%strategy-from-trace slice))
         (confidence (lookup-state-value graph :strategy-confidence 0.0)))
    (register-state-value graph :strategy strategy)
    (update-state-value graph :strategy-confidence (lambda (value)
                                                     (+ (or value 0.0) 0.2))
                        confidence)
    (record-trace graph :self-reflection
                  (list :cycle cycle
                        :inspected (mapcar (lambda (entry)
                                             (list :module (getf entry :module)
                                                   :info (getf entry :info)))
                                           slice)
                        :strategy strategy)))
  strategy)
