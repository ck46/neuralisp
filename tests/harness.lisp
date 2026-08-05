;;;; A small dependency-free test harness.
;;;;
;;;; NeuraLisp's tests previously used lisp-unit, which meant they could not run
;;;; without Quicklisp -- and so, in practice, never ran at all.  This harness is
;;;; a few dozen lines of ANSI Common Lisp instead, so `sbcl --script` is enough.

(defpackage :neuralisp.tests.harness
  (:use :cl)
  (:export #:deftest #:run-all-tests #:registered-tests
           #:check #:check-equal #:check-near #:check-signals))
(in-package :neuralisp.tests.harness)

(defvar *tests* '()
  "Registered tests as (NAME . THUNK), in definition order.")

(defvar *failures* '()
  "Failure descriptions collected during the current run.")

(defvar *current-test* nil
  "Name of the test being executed, for failure reporting.")

(defun register-test (name thunk)
  (setf *tests* (append (remove name *tests* :key #'car) (list (cons name thunk))))
  name)

(defun registered-tests ()
  "Names of all registered tests, in definition order."
  (mapcar #'car *tests*))

(defmacro deftest (name &body body)
  "Define a test named NAME. Re-evaluating replaces the previous definition."
  `(register-test ',name (lambda () ,@body)))

(defun record-failure (control &rest arguments)
  (push (cons *current-test* (apply #'format nil control arguments)) *failures*)
  nil)

(defmacro check (form &optional description)
  "Fail unless FORM evaluates to a true value."
  (let ((error-var (gensym "ERROR")))
    `(handler-case (or ,form
                       (record-failure "~s was false~@[: ~a~]" ',form ,description))
       (error (,error-var)
         (record-failure "~s signalled ~a" ',form ,error-var)))))

(defmacro check-equal (expected form)
  "Fail unless FORM is EQUALP to EXPECTED."
  (let ((expected-var (gensym "EXPECTED"))
        (actual-var (gensym "ACTUAL"))
        (error-var (gensym "ERROR")))
    `(handler-case
         (let ((,expected-var ,expected)
               (,actual-var ,form))
           (or (equalp ,expected-var ,actual-var)
               (record-failure "~s returned ~s, expected ~s"
                               ',form ,actual-var ,expected-var)))
       (error (,error-var)
         (record-failure "~s signalled ~a" ',form ,error-var)))))

(defmacro check-near (expected form &optional (tolerance 1d-9))
  "Fail unless FORM is numerically within TOLERANCE of EXPECTED."
  (let ((expected-var (gensym "EXPECTED"))
        (actual-var (gensym "ACTUAL"))
        (error-var (gensym "ERROR")))
    `(handler-case
         (let ((,expected-var ,expected)
               (,actual-var ,form))
           (or (<= (abs (- ,expected-var ,actual-var)) ,tolerance)
               (record-failure "~s returned ~s, expected ~s (tolerance ~s)"
                               ',form ,actual-var ,expected-var ,tolerance)))
       (error (,error-var)
         (record-failure "~s signalled ~a" ',form ,error-var)))))

(defmacro check-signals (condition-type form)
  "Fail unless FORM signals a condition of CONDITION-TYPE."
  (let ((block-name (gensym "CHECK-SIGNALS")))
    `(block ,block-name
       (handler-case ,form
         (,condition-type () (return-from ,block-name t))
         (error (unexpected)
           (return-from ,block-name
             (record-failure "~s signalled ~a, expected ~s"
                             ',form unexpected ',condition-type))))
       (record-failure "~s returned normally, expected ~s to be signalled"
                       ',form ',condition-type))))

(defun run-all-tests (&key (stream *standard-output*))
  "Run every registered test. Print a report and return T when all pass."
  (let ((*failures* '())
        (passed 0)
        (failed 0))
    (format stream "~&Running ~d test~:p...~%" (length *tests*))
    (dolist (entry *tests*)
      (let* ((*current-test* (car entry))
             (before (length *failures*)))
        (handler-case (funcall (cdr entry))
          (error (condition)
            (record-failure "unhandled error: ~a" condition)))
        (cond ((= before (length *failures*))
               (incf passed)
               (format stream "  pass  ~a~%" (car entry)))
              (t
               (incf failed)
               (format stream "  FAIL  ~a~%" (car entry))))))
    (dolist (failure (reverse *failures*))
      (format stream "~&    ~a: ~a~%" (car failure) (cdr failure)))
    (format stream "~&~d passed, ~d failed.~%" passed failed)
    (zerop failed)))
