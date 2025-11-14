(defpackage :neuralisp.cognition.reasoning
  (:use :cl)
  (:import-from :neuralisp.cognition.state
                :get-node
                :node-data
                :record-trace
                :state-cycle-count
                :tensor->symbolic
                :symbolic->tensor-compatible
                :update-node-data
                :lookup-state-value
                :register-state-value)
  (:export :symbolic-rule
           :logic-graph
           :make-symbolic-rule
           :activate-logic-graph
           :reasoning-module-step
           :rule-fired-p
           :ensure-embedding-node
           :adjust-embedding-confidence))

(in-package :neuralisp.cognition.reasoning)

(defclass symbolic-rule ()
  ((name :initarg :name
         :reader rule-name)
   (condition :initarg :condition
              :reader rule-condition)
   (action :initarg :action
           :reader rule-action)
   (weight :initarg :weight
           :initform 1.0
           :accessor rule-weight)
   (last-fired :initform nil
               :accessor rule-last-fired)))

(defclass logic-graph ()
  ((rules :initarg :rules
          :accessor logic-graph-rules
          :documentation "Collection of SYMBOLIC-RULE instances applied during reasoning.")))

(defun make-symbolic-rule (&key name condition action (weight 1.0))
  (make-instance 'symbolic-rule :name name :condition condition :action action :weight weight))

(defun rule-fired-p (rule)
  (not (null (rule-last-fired rule))))

(defun %evaluate-rule (rule graph)
  (let* ((condition (rule-condition rule))
         (result (handler-case (funcall condition graph)
                   (error (e)
                     (declare (ignore e))
                     nil))))
    (when result
      (let ((action (rule-action rule)))
        (setf (rule-last-fired rule)
              (handler-case (funcall action graph result rule)
                (error (e)
                  (declare (ignore e))
                  nil))))))
  rule)

(defun activate-logic-graph (logic-graph graph)
  "Evaluate all rules in LOGIC-GRAPH against GRAPH, returning rules that fired during this reasoning pass."
  (remove-if-not #'rule-fired-p
                 (mapcar (lambda (rule)
                           (prog1 (%evaluate-rule rule graph)
                             (unless (rule-fired-p rule)
                               (setf (rule-last-fired rule) nil))))
                         (logic-graph-rules logic-graph))))

(defun reasoning-module-step (graph logic-graph)
  "Run symbolic reasoning on GRAPH using LOGIC-GRAPH.
A trace entry is emitted capturing the list of fired rules and their actions."
  (let* ((fired (activate-logic-graph logic-graph graph))
         (cycle (state-cycle-count graph))
         (summary (mapcar #'rule-name fired)))
    (record-trace graph :reasoning
                  (list :cycle cycle
                        :rules summary)))
  logic-graph)

;;; Example bridging utilities -------------------------------------------------

(defun ensure-embedding-node (graph)
  (or (get-node graph :embedding)
      (progn
        (register-state-value graph :embedding-confidence 0.0)
        (update-node-data graph :embedding (lambda (old)
                                             (or old '()))))))

(defun adjust-embedding-confidence (graph delta)
  (let ((current (lookup-state-value graph :embedding-confidence 0.0)))
    (register-state-value graph :embedding-confidence (+ current delta))))
