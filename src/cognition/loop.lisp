(defpackage :neuralisp.cognition.loop
  (:use :cl)
  (:import-from :neuralisp.cognition.state
                :make-state-graph
                :state-graph
                :advance-cycle
                :state-cycle-count
                :record-trace
                :register-state-value)
  (:import-from :neuralisp.cognition.knowledge
                :knowledge-representation
                :knowledge-module-step)
  (:import-from :neuralisp.cognition.memory
                :memory-store
                :memory-module-step)
  (:import-from :neuralisp.cognition.reasoning
                :logic-graph
                :reasoning-module-step)
  (:import-from :neuralisp.cognition.learning
                :learning-module-step)
  (:import-from :neuralisp.cognition.self-reflection
                :self-reflection-module-step)
  (:import-from :neuralisp.cognition.perception
                :perception-module-step)
  (:import-from :neuralisp.cognition.actuation
                :actuation-module-step)
  (:export :cognitive-loop
           :make-cognitive-loop
           :register-hook
           :run-cognitive-loop
           :cognitive-loop-state
           :cognitive-loop-knowledge
           :cognitive-loop-memory
           :cognitive-loop-logic-graph
           :cognitive-loop-hooks))

(in-package :neuralisp.cognition.loop)

(defclass cognitive-loop ()
  ((state :initarg :state
          :reader cognitive-loop-state)
   (knowledge :initarg :knowledge
              :reader cognitive-loop-knowledge)
   (memory :initarg :memory
           :reader cognitive-loop-memory)
   (logic-graph :initarg :logic-graph
                :reader cognitive-loop-logic-graph)
   (hooks :initform (make-hash-table :test 'eq)
          :reader cognitive-loop-hooks)))

(defun make-cognitive-loop (&key (state (make-state-graph))
                                 (logic-graph (make-instance 'logic-graph :rules '()))
                                 (memory-capacity 50))
  "Instantiate a COGNITIVE-LOOP with default module implementations.
Callers can inject custom LOGIC-GRAPHs or reuse STATE for multiple experiments."
  (let ((loop (make-instance 'cognitive-loop
                             :state state
                             :knowledge (make-instance 'knowledge-representation)
                             :memory (make-instance 'memory-store :capacity memory-capacity)
                             :logic-graph logic-graph)))
    (register-state-value state :strategy :reactive)
    loop))

(defun register-hook (loop hook-name function)
  "Register FUNCTION as the handler for HOOK-NAME.
Hooks currently supported: :before-cycle, :after-cycle, :before-module, :after-module."
  (setf (gethash hook-name (cognitive-loop-hooks loop)) function)
  loop)

(defun %call-hook (loop hook-name &rest args)
  (let ((fn (gethash hook-name (cognitive-loop-hooks loop))))
    (when fn
      (apply fn loop args))))

(defun %execute-module (loop module-name environment)
  (let ((state (cognitive-loop-state loop)))
    (%call-hook loop :before-module module-name state)
    (ecase module-name
      (:knowledge (knowledge-module-step state (cognitive-loop-knowledge loop)))
      (:memory (memory-module-step state (cognitive-loop-knowledge loop) (cognitive-loop-memory loop)))
      (:reasoning (reasoning-module-step state (cognitive-loop-logic-graph loop)))
      (:learning (learning-module-step state (cognitive-loop-logic-graph loop)))
      (:self-reflection (self-reflection-module-step state))
      (:perception (perception-module-step state environment))
      (:actuation (actuation-module-step state environment)))
    (%call-hook loop :after-module module-name state)))

(defun run-cognitive-loop (loop &key (cycles 1) environment)
  "Execute CYCLES iterations of the canonical cognitive processing order.
ENVIRONMENT is a function that responds to :perceive and :act messages."
  (dotimes (iteration cycles)
    (declare (ignore iteration))
    (let ((state (cognitive-loop-state loop)))
      (advance-cycle state)
      (%call-hook loop :before-cycle state))
    (dolist (module '(:knowledge :memory :reasoning :learning :self-reflection :perception :actuation))
      (%execute-module loop module environment))
    (let ((state (cognitive-loop-state loop)))
      (%call-hook loop :after-cycle state)))
  loop)
