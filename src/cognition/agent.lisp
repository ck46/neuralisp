(defpackage :neuralisp.cognition.agent
  (:use :cl)
  (:import-from :neuralisp.cognition.loop
                :make-cognitive-loop
                :run-cognitive-loop
                :register-hook
                :cognitive-loop-state
                :cognitive-loop-logic-graph)
  (:import-from :neuralisp.cognition.state
                :make-state-graph
                :collect-node-data
                :lookup-state-value
                :update-state-value
                :register-state-value
                :record-trace)
  (:import-from :neuralisp.cognition.reasoning
                :logic-graph
                :make-symbolic-rule
                :ensure-embedding-node
                :adjust-embedding-confidence)
  (:export :make-text-environment
           :environment-history
           :make-default-logic-graph
           :reflective-agent
           :agent-loop
           :agent-environment
           :make-reflective-agent
           :run-agent))

(in-package :neuralisp.cognition.agent)

(defclass reflective-agent ()
  ((loop :initarg :loop
         :reader agent-loop)
   (environment :initarg :environment
                :reader agent-environment)))

(defun make-text-environment (&key prompts embeddings)
  "Create a simple textual environment delivering PROMPTS with optional EMBEDDINGS.
The environment function responds to :perceive (returning a percept plist) and :act (capturing agent actions)."
  (let ((input-queue (copy-list prompts))
        (embedding-queue (copy-list embeddings))
        (percept-history '())
        (action-history '()))
    (labels ((dispatch (command &optional payload)
               (case command
                 (:perceive (let ((content (pop input-queue)))
                              (when content
                                (let* ((embedding (or (pop embedding-queue)
                                                     (list (/ (length content) 10.0))))
                                       (percept (list :content content :embedding embedding)))
                                  (push percept percept-history)
                                  percept))))
                 (:act (push (list :action payload) action-history)
                       payload)
                 (:percepts (nreverse percept-history))
                 (:actions (nreverse action-history))
                 (otherwise (error "Unknown environment command ~A" command)))))
      #'dispatch)))

(defun environment-history (environment)
  "Retrieve both percept and action traces from ENVIRONMENT."
  (values (funcall environment :percepts)
          (funcall environment :actions)))

(defun make-default-logic-graph ()
  "Construct a minimal logic graph that elevates salient percepts based on their symbolic summaries."
  (let* ((salience-rule
           (make-symbolic-rule
            :name :promote-salient-percept
            :condition (lambda (graph)
                         (let* ((facts (lookup-state-value graph :active-facts '()))
                                (fact-data (collect-node-data graph facts)))
                           (find-if (lambda (fact)
                                      (> (getf (getf fact :symbolic) :value 0.5))
                                    fact-data)))
            :action (lambda (graph fact rule)
                      (declare (ignore rule))
                      (ensure-embedding-node graph)
                      (adjust-embedding-confidence graph 0.2)
                      (update-state-value graph :salient-facts
                                          (lambda (existing)
                                            (cons (getf fact :id) (or existing '())))
                                          '())
                      (list :selected (getf fact :id))))))
    (make-instance 'logic-graph :rules (list salience-rule))))

(defun make-reflective-agent (&key prompts embeddings (logic-graph (make-default-logic-graph)))
  "Create a REFLECTIVE-AGENT ready to operate inside the standard cognitive loop."
  (let* ((environment (make-text-environment :prompts prompts :embeddings embeddings))
         (state (make-state-graph))
         (loop (make-cognitive-loop :state state :logic-graph logic-graph)))
    (register-state-value state :total-cycles 0)
    (register-hook loop :after-cycle
                   (lambda (loop state)
                     (declare (ignore loop))
                     (record-trace state :agent-summary
                                  (list :cycle (lookup-state-value state :total-cycles 0)
                                        :strategy (lookup-state-value state :strategy :reactive)))
                     (update-state-value state :total-cycles #'1+ 0)))
    (make-instance 'reflective-agent :loop loop :environment environment)))

(defun run-agent (agent &key (cycles 1))
  "Execute CYCLES of the cognitive loop for AGENT."
  (run-cognitive-loop (agent-loop agent) :cycles cycles :environment (agent-environment agent))
  agent)
