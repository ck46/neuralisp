(defpackage :neuralisp.cognition.memory
  (:use :cl)
  (:import-from :neuralisp.cognition.state
                :ensure-node
                :update-node-data
                :record-trace
                :state-cycle-count)
  (:import-from :neuralisp.cognition.knowledge
                :knowledge-active-facts)
  (:export :memory-store
           :memory-module-step
           :memory-contents))

(in-package :neuralisp.cognition.memory)

(defclass memory-store ()
  ((buffer :initform '()
           :accessor memory-contents
           :documentation "Ordered list representing episodic memory of facts (newest first).")
   (capacity :initarg :capacity
             :initform 50
             :accessor memory-capacity)))

(defun %append-memory (facts memory capacity)
  (let ((new-buffer (append facts memory)))
    (if (> (length new-buffer) capacity)
        (subseq new-buffer 0 capacity)
        new-buffer)))

(defun memory-module-step (graph knowledge memory)
  "Store the facts produced during the knowledge representation stage into episodic memory."
  (let ((facts (knowledge-active-facts knowledge)))
    (when facts
      (setf (memory-contents memory)
            (%append-memory facts (memory-contents memory) (memory-capacity memory)))))
  (record-trace graph :memory
                (list :cycle (state-cycle-count graph)
                      :stored (memory-contents memory)))
  memory)
