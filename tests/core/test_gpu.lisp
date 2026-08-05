;;;; Tests for the GPU placement interface.
;;;;
;;;; There is no real device backend, so these tests cover the two things that
;;;; are actually true today: with no backend installed the module refuses
;;;; clearly, and with a stub backend the placement bookkeeping is consistent.

(defpackage :neuralisp.tests.core.gpu
  (:use :cl :neuralisp.tests.harness)
  (:import-from :neuralisp.core.tensor
                #:make-tensor #:tensor-equal-p #:tensor-device #:tensor-gpu-pointer)
  (:import-from :neuralisp.core.gpu
                #:*gpu-backend* #:gpu-available-p #:gpu-backend-unavailable
                #:backend-to-device #:backend-from-device #:backend-free
                #:tensor-on-gpu-p #:move-to-gpu #:move-to-cpu
                #:tensor-ensure-on-gpu #:tensor-ensure-on-cpu))
(in-package :neuralisp.tests.core.gpu)

;;; A backend that "transfers" by holding onto the host data, so the interface
;;; can be exercised without a device.

(defclass stub-backend ()
  ((buffers :initform (make-hash-table :test #'eql) :reader stub-buffers)
   (next-handle :initform 0 :accessor stub-next-handle)
   (freed :initform '() :accessor stub-freed)))

(defmethod backend-to-device ((backend stub-backend) tensor)
  (let ((handle (incf (stub-next-handle backend))))
    (setf (gethash handle (stub-buffers backend))
          (copy-seq (neuralisp.core.tensor:tensor-data tensor)))
    handle))

(defmethod backend-from-device ((backend stub-backend) handle shape)
  (make-tensor shape :data (copy-seq (gethash handle (stub-buffers backend)))))

(defmethod backend-free ((backend stub-backend) handle)
  (push handle (stub-freed backend))
  (remhash handle (stub-buffers backend))
  t)

(deftest no-backend-is-installed-by-default
  (check (not (gpu-available-p))))

(deftest move-to-gpu-refuses-without-a-backend
  (let ((*gpu-backend* nil))
    (check-signals gpu-backend-unavailable (move-to-gpu (make-tensor '(2 2))))))

(deftest ensure-on-gpu-refuses-without-a-backend
  (let ((*gpu-backend* nil))
    (check-signals gpu-backend-unavailable (tensor-ensure-on-gpu (make-tensor '(2 2))))))

(deftest tensors-start-on-the-cpu
  (let ((tensor (make-tensor '(2 2))))
    (check (not (tensor-on-gpu-p tensor)))
    (check-equal :cpu (tensor-device tensor))
    (check (null (tensor-gpu-pointer tensor)))))

(deftest move-to-cpu-is-a-no-op-for-host-tensors
  (let ((tensor (make-tensor '(2 2) :initial-element 3)))
    (check (eq tensor (move-to-cpu tensor)))))

(deftest a-round-trip-through-a-backend-preserves-contents
  (let* ((*gpu-backend* (make-instance 'stub-backend))
         (original (make-tensor '(2 2) :data '(1 2 3 4)))
         (on-device (move-to-gpu original)))
    (check (gpu-available-p))
    (check (tensor-on-gpu-p on-device))
    (check (not (tensor-on-gpu-p original)))
    (check (tensor-equal-p original (move-to-cpu on-device)))))

(deftest ensure-on-cpu-releases-the-device-buffer
  (let* ((backend (make-instance 'stub-backend))
         (*gpu-backend* backend)
         (tensor (make-tensor '(2 2) :data '(1 2 3 4))))
    (tensor-ensure-on-gpu tensor)
    (check (tensor-on-gpu-p tensor))
    (tensor-ensure-on-cpu tensor)
    (check (not (tensor-on-gpu-p tensor)))
    (check (null (tensor-gpu-pointer tensor)))
    (check-equal 1 (length (stub-freed backend)))
    (check (tensor-equal-p (make-tensor '(2 2) :data '(1 2 3 4)) tensor))))
