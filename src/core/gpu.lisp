(defpackage :neuralisp.core.gpu
  (:use :cl)
  (:import-from :neuralisp.core.tensor
                :tensor
                :tensor-data
                :tensor-size
                :tensor-dtype
                :tensor-shape
                :tensor-gpu-pointer
                :make-tensor)
  (:export :initialize-gpu
           :shutdown-gpu
           :to-gpu
           :from-gpu
           :gpu-allocate
           :gpu-deallocate
           :tensor-on-gpu-p
           :tensor-ensure-on-gpu
           :tensor-ensure-on-cpu
           :move-to-gpu
           :move-to-cpu))
(in-package :neuralisp.core.gpu)

(defstruct gpu-buffer
  (storage (make-array 0 :element-type 'number)
           :type (simple-array number (*))
           :documentation "Backing storage for emulated device memory.")
  (dtype :float64 :type keyword)
  (backend :emulated :type keyword))

(defparameter *gpu-backend* :emulated
  "Symbol describing the active GPU backend. Defaults to :EMULATED when no GPU is available.")

(defparameter *gpu-initialized* nil
  "Tracks whether the GPU subsystem has been initialized.")

(defun initialize-gpu ()
  "Initialize the GPU subsystem. In environments without CUDA support the backend falls back to emulation."
  (setf *gpu-initialized* t)
  (setf *gpu-backend* :emulated)
  *gpu-backend*)

(defun shutdown-gpu ()
  "Shutdown the GPU subsystem and release any cached state."
  (setf *gpu-initialized* nil)
  :ok)

(defun %ensure-initialized ()
  (unless *gpu-initialized*
    (initialize-gpu)))

(defun %ensure-buffer-size (buffer size)
  (let ((storage (gpu-buffer-storage buffer)))
    (unless (= (length storage) size)
      (setf (gpu-buffer-storage buffer) (make-array size :element-type 'number)))))

(defun %copy-host-to-device (tensor buffer)
  (let* ((size (tensor-size tensor))
         (host (tensor-data tensor)))
    (%ensure-buffer-size buffer size)
    (setf (gpu-buffer-dtype buffer) (tensor-dtype tensor))
    (dotimes (i size)
      (setf (aref (gpu-buffer-storage buffer) i) (aref host i)))
    buffer))

(defun %copy-device-to-host (tensor buffer)
  (let* ((size (tensor-size tensor))
         (host (tensor-data tensor))
         (storage (gpu-buffer-storage buffer)))
    (unless (= (length storage) size)
      (error "Device buffer size mismatch: expected ~A, got ~A" size (length storage)))
    (dotimes (i size)
      (setf (aref host i) (aref storage i)))
    tensor))

(defun gpu-allocate (size &key (dtype :float64))
  "Allocate a GPU buffer of SIZE elements using the emulated backend."
  (%ensure-initialized)
  (make-gpu-buffer :storage (make-array size :element-type 'number)
                   :dtype dtype
                   :backend *gpu-backend*))

(defun gpu-deallocate (buffer)
  "Release BUFFER resources."
  (when buffer
    (setf (gpu-buffer-storage buffer) (make-array 0 :element-type 'number)))
  :ok)

(defun to-gpu (tensor)
  "Create a device buffer holding TENSOR's data."
  (%ensure-initialized)
  (let ((buffer (gpu-allocate (tensor-size tensor) :dtype (tensor-dtype tensor))))
    (%copy-host-to-device tensor buffer)
    buffer))

(defun from-gpu (buffer shape &key dtype)
  "Materialize a tensor from BUFFER with SHAPE."
  (let* ((storage (gpu-buffer-storage buffer))
         (data (make-array (length storage) :element-type 'number)))
    (dotimes (i (length storage))
      (setf (aref data i) (aref storage i)))
    (make-tensor :data data :shape shape :dtype (or dtype (gpu-buffer-dtype buffer)))))

(defun tensor-on-gpu-p (tensor)
  "Return T when TENSOR has a device allocation."
  (not (null (tensor-gpu-pointer tensor))))

(defun tensor-ensure-on-gpu (tensor)
  "Ensure TENSOR has an up-to-date device buffer and return the tensor."
  (%ensure-initialized)
  (let ((buffer (tensor-gpu-pointer tensor)))
    (if buffer
        (%copy-host-to-device tensor buffer)
        (setf (tensor-gpu-pointer tensor) (to-gpu tensor))))
  tensor)

(defun tensor-ensure-on-cpu (tensor)
  "Ensure the host copy of TENSOR reflects the current device contents."
  (let ((buffer (tensor-gpu-pointer tensor)))
    (when buffer
      (%copy-device-to-host tensor buffer)))
  tensor)

(defun move-to-gpu (tensor)
  "Move TENSOR to the GPU by ensuring a device allocation exists."
  (tensor-ensure-on-gpu tensor))

(defun move-to-cpu (tensor)
  "Move TENSOR back to CPU memory and release the device allocation."
  (tensor-ensure-on-cpu tensor)
  (when (tensor-gpu-pointer tensor)
    (gpu-deallocate (tensor-gpu-pointer tensor))
    (setf (tensor-gpu-pointer tensor) nil))
  tensor)
