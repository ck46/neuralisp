(defpackage :neuralisp.tests.cognition.cognitive-loop
  (:use :cl)
  (:import-from :neuralisp.cognition.agent
                :make-reflective-agent
                :run-agent
                :environment-history
                :agent-loop
                :agent-environment)
  (:import-from :neuralisp.cognition.loop
                :cognitive-loop-state
                :cognitive-loop-memory)
  (:import-from :neuralisp.cognition.memory
                :memory-contents)
  (:import-from :neuralisp.cognition.state
                :lookup-state-value
                :state-trace)
  (:export :run-cognition-loop-tests))

(in-package :neuralisp.tests.cognition.cognitive-loop)

(defun %assert (condition message &rest args)
  (unless condition
    (error (apply #'format nil message args))))

(defun run-cognition-loop-tests ()
  "Integration test that validates a full pass through the cognitive loop."
  (let* ((prompts '("High signal" "Low follow-up"))
         (embeddings '((0.9 0.8) (0.2 0.3)))
         (agent (make-reflective-agent :prompts prompts :embeddings embeddings))
         (loop (agent-loop agent)))
    (run-agent agent :cycles 3)
    (let* ((state (cognitive-loop-state loop))
           (memory (cognitive-loop-memory loop))
           (strategy (lookup-state-value state :strategy nil))
           (salient (lookup-state-value state :salient-facts '()))
           (trace (state-trace state)))
      (%assert (eql strategy :deliberate) "Expected reflective strategy to become DELIBERATE, observed ~A" strategy)
      (%assert (> (length (memory-contents memory)) 0) "Memory buffer should contain facts after loop execution")
      (%assert salient "Reasoning should mark at least one salient fact")
      (multiple-value-bind (percepts actions) (environment-history (agent-environment agent))
        (%assert (= (length percepts) 2) "Environment should have delivered two percepts")
        (%assert (> (length actions) 0) "Agent should have produced at least one action"))
      (let ((self-reflection-entry (find :self-reflection trace :key (lambda (entry) (getf entry :module)))))
        (%assert self-reflection-entry "Self-reflection trace entry missing")
        (%assert (getf (getf self-reflection-entry :info) :inspected)
                 "Self-reflection should record inspected module traces")))
    t))

(defun run-cognition-loop-tests-and-report ()
  (format t "Running cognition loop integration tests...~%")
  (handler-case
      (progn
        (run-cognition-loop-tests)
        (format t "ok~%"))
    (error (e)
      (format t "FAILED: ~A~%" e)))
  (values))

(setf (symbol-function 'run-cognition-loop-tests) #'run-cognition-loop-tests)
