(defpackage :neuralisp.cognition.perception
  (:use :cl)
  (:import-from :neuralisp.cognition.state
                :ensure-node
                :record-trace
                :state-cycle-count)
  (:export :perception-module-step))

(in-package :neuralisp.cognition.perception)

(defun perception-module-step (graph environment)
  "Request a new percept from ENVIRONMENT and store it on the shared graph.
Percepts are queued so that the knowledge representation module can process them in the subsequent cycle."
  (let* ((percept (and environment (funcall environment :perceive)))
         (node (ensure-node graph :percept :type :percept :data '())))
    (when percept
      (setf (node-data node) (append (node-data node) (list percept))))
    (record-trace graph :perception
                  (list :cycle (state-cycle-count graph)
                        :percept percept)))
  graph)
