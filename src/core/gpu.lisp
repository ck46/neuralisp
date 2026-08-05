;;;; GPU placement interface for NeuraLisp.
;;;;
;;;; There is no working GPU backend yet.  This module exists to give the rest
;;;; of the system one coherent way to talk about device placement, and to fail
;;;; loudly and specifically when something asks for a device that is not there.
;;;; It loads without CUDA present, which the previous cl-cuda-based version did
;;;; not -- it could not even be READ on a CPU-only machine.
;;;;
;;;; To supply a real backend, bind *GPU-BACKEND* to an object implementing
;;;; BACKEND-TO-DEVICE / BACKEND-FROM-DEVICE / BACKEND-FREE.

(defpackage :neuralisp.core.gpu
  (:use :cl)
  (:import-from :neuralisp.core.tensor
                #:tensor #:tensor-data #:tensor-shape #:tensor-device
                #:tensor-gpu-pointer #:make-tensor #:tensor-copy)
  (:export #:*gpu-backend* #:gpu-available-p #:gpu-backend-unavailable
           #:backend-to-device #:backend-from-device #:backend-free
           #:tensor-on-gpu-p #:move-to-gpu #:move-to-cpu
           #:tensor-ensure-on-gpu #:tensor-ensure-on-cpu))
(in-package :neuralisp.core.gpu)

(defvar *gpu-backend* nil
  "The active GPU backend object, or NIL when no device backend is installed.")

(define-condition gpu-backend-unavailable (error)
  ((operation :initarg :operation :initform nil :reader gpu-backend-unavailable-operation))
  (:report (lambda (condition stream)
             (format stream "No GPU backend is installed~@[ (needed by ~a)~]. ~
Bind NEURALISP.CORE.GPU:*GPU-BACKEND* to a backend object first."
                     (gpu-backend-unavailable-operation condition))))
  (:documentation "Signalled when device placement is requested with no backend."))

(defun gpu-available-p ()
  "True when a GPU backend is installed."
  (and *gpu-backend* t))

(defgeneric backend-to-device (backend tensor)
  (:documentation "Copy TENSOR's contents to the device; return a device handle."))

(defgeneric backend-from-device (backend handle shape)
  (:documentation "Copy the device buffer at HANDLE back into a fresh host tensor of SHAPE."))

(defgeneric backend-free (backend handle)
  (:documentation "Release the device buffer at HANDLE."))

(defun tensor-on-gpu-p (tensor)
  "True when TENSOR's contents currently live on a device."
  (eq :gpu (tensor-device tensor)))

(defun move-to-gpu (tensor)
  "Return a tensor whose contents live on the device.

Signals GPU-BACKEND-UNAVAILABLE when no backend is installed, rather than
returning a tensor that only claims to be on a device."
  (cond ((tensor-on-gpu-p tensor) tensor)
        ((null *gpu-backend*) (error 'gpu-backend-unavailable :operation 'move-to-gpu))
        (t (let ((copy (tensor-copy tensor)))
             (setf (tensor-gpu-pointer copy) (backend-to-device *gpu-backend* tensor)
                   (tensor-device copy) :gpu)
             copy))))

(defun move-to-cpu (tensor)
  "Return a tensor whose contents live in host memory."
  (if (tensor-on-gpu-p tensor)
      (let ((host (backend-from-device *gpu-backend*
                                       (tensor-gpu-pointer tensor)
                                       (tensor-shape tensor))))
        (setf (tensor-device host) :cpu
              (tensor-gpu-pointer host) nil)
        host)
      tensor))

(defun tensor-ensure-on-gpu (tensor)
  "Move TENSOR to the device in place, returning it."
  (unless (tensor-on-gpu-p tensor)
    (when (null *gpu-backend*)
      (error 'gpu-backend-unavailable :operation 'tensor-ensure-on-gpu))
    (setf (tensor-gpu-pointer tensor) (backend-to-device *gpu-backend* tensor)
          (tensor-device tensor) :gpu))
  tensor)

(defun tensor-ensure-on-cpu (tensor)
  "Move TENSOR back to host memory in place, returning it."
  (when (tensor-on-gpu-p tensor)
    (let ((host (backend-from-device *gpu-backend*
                                     (tensor-gpu-pointer tensor)
                                     (tensor-shape tensor))))
      (setf (tensor-data tensor) (tensor-data host))
      (backend-free *gpu-backend* (tensor-gpu-pointer tensor))
      (setf (tensor-gpu-pointer tensor) nil
            (tensor-device tensor) :cpu)))
  tensor)
