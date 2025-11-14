(defpackage :neuralisp.cognition.knowledge
  (:use :cl)
  (:import-from :neuralisp.cognition.state
                :state-graph
                :ensure-node
                :record-trace
                :tensor->symbolic
                :state-cycle-count
                :register-state-value)
  (:export :knowledge-representation
           :knowledge-module-step
           :extract-knowledge-facts))

(in-package :neuralisp.cognition.knowledge)

(defclass knowledge-representation ()
  ((name :initform :knowledge-representation
         :accessor knowledge-name)
   (active-facts :initform '()
                 :accessor knowledge-active-facts
                 :documentation "List of fact identifiers captured during the most recent cognitive cycle.")))

(defun extract-knowledge-facts (graph)
  "Collect perceptual inputs that have been staged for assimilation.
The perception module stores raw percepts under the :percept node. We normalise them into fact descriptors."
  (let ((node (ensure-node graph :percept :type :percept :data '())))
    (loop for percept in (node-data node)
          for fact-id = (gensym "FACT-")
          collect (list :id fact-id
                        :source percept
                        :embedding (getf percept :embedding)
                        :symbolic (tensor->symbolic (getf percept :embedding))))))

(defun %register-fact (graph fact)
  (let* ((fact-node-id (getf fact :id))
         (node (ensure-node graph fact-node-id :type :fact)))
    (setf (node-data node) fact)
    node))

(defun knowledge-module-step (graph module)
  "Assimilate perceptual information into persistent symbolic facts.
The resulting fact identifiers are recorded so that downstream modules (memory, reasoning, learning) can reference them."
  (let* ((percept-node (ensure-node graph :percept :type :percept :data '()))
         (facts (extract-knowledge-facts graph))
         (cycle (state-cycle-count graph)))
    (setf (knowledge-active-facts module) (mapcar (lambda (fact)
                                                   (node-id (%register-fact graph fact)))
                                                 facts))
    (register-state-value graph :active-facts (knowledge-active-facts module))
    (setf (node-data percept-node) '())
    (record-trace graph :knowledge
                  (list :cycle cycle
                        :facts (knowledge-active-facts module))))
  module)
