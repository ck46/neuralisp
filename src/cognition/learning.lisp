(defpackage :neuralisp.cognition.learning
  (:use :cl)
  (:import-from :neuralisp.cognition.state
                :record-trace
                :state-cycle-count
                :update-state-value)
  (:import-from :neuralisp.cognition.reasoning
                :logic-graph-rules
                :rule-fired-p
                :rule-weight)
  (:export :learning-module-step))

(in-package :neuralisp.cognition.learning)

(defun learning-module-step (graph logic-graph)
  "Update the confidence weights for rules that fired during the latest reasoning pass.
A simple delta rule is used to increment weights for successful rules and decay otherwise."
  (let* ((cycle (state-cycle-count graph))
         (updates
          (loop for rule in (logic-graph-rules logic-graph)
                collect (let ((delta (if (rule-fired-p rule) 0.1 -0.05)))
                          (setf (rule-weight rule) (max 0.0 (+ (rule-weight rule) delta)))
                          (list :rule (slot-value rule 'name)
                                :delta delta
                                :weight (rule-weight rule))))))
    (update-state-value graph :reasoning-confidence
                        (lambda (value) (+ (or value 0.0)
                                           (reduce #'+ (mapcar (lambda (entry)
                                                                 (getf entry :delta))
                                                               updates))))
                        0.0)
    (record-trace graph :learning
                  (list :cycle cycle
                        :updates updates)))
  logic-graph)
