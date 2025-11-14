(defpackage :neuralisp.cognition.actuation
  (:use :cl)
  (:import-from :neuralisp.cognition.state
                :lookup-state-value
                :record-trace
                :state-cycle-count
                :ensure-node)
  (:export :actuation-module-step))

(in-package :neuralisp.cognition.actuation)

(defun %select-action (graph)
  (let ((strategy (lookup-state-value graph :strategy :reactive)))
    (case strategy
      (:deliberate :respond-with-plan)
      (:reactive :respond-immediately)
      (t :observe))))

(defun actuation-module-step (graph environment)
  "Generate an action suggestion informed by the current cognitive strategy.
Actions are emitted to ENVIRONMENT (if provided) and captured in the shared trace."
  (let ((action (%select-action graph)))
    (when environment
      (funcall environment :act action))
    (ensure-node graph :last-action :type :action :data action)
    (record-trace graph :actuation
                  (list :cycle (state-cycle-count graph)
                        :action action)))
  graph)
