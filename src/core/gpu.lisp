(defpackage :neuralisp.core.gpu
  (:use :common-lisp :cl-cuda)
  (:import-from :neuralisp.core.tensor :tensor :tensor-data :tensor-shape :make-tensor
                :tensor-gpu-pointer)
  (:export :initialize-gpu :shutdown-gpu :to-gpu :from-gpu :gpu-allocate :gpu-deallocate
           :tensor-on-gpu-p :tensor-ensure-on-gpu :tensor-ensure-on-cpu :move-to-gpu :move-to-cpu))
(in-package :neuralisp.core.gpu)

(defun initialize-gpu ()
  "Initialize the GPU and cl-cuda library."
  (setf cl-cuda.all:use-cache-p t) ; Enable caching of compiled CUDA programs
  (dolist (platform (cl-cuda.all:get-platform-ids))
    (dolist (device (cl-cuda.all:get-device-ids platform))
      (format t "Platform: ~A, Device: ~A~%" platform device))))

(defun shutdown-gpu ()
  "Clean up the GPU and cl-cuda library before exiting."
  (cl-cuda.basic:shutdown))

(defun to-gpu (tensor)
  "Send the tensor to GPU memory."
  (let ((tensor-dev-ptr (cublas:allocate (reduce #'* (tensor-shape tensor)))))
    (cublas:with-cublas
      (cublas:send-to tensor-dev-ptr (tensor-data tensor) (reduce #'* (tensor-shape tensor))))
    tensor-dev-ptr))

(defun from-gpu (tensor-dev-ptr shape)
  "Retrieve tensor from GPU memory, given its device-pointer and shape."
  (let ((tensor-data (make-array (reduce #'* shape) :element-type 'single-float)))
    (cublas:with-cublas
      (cublas:retrieve-from tensor-dev-ptr tensor-data (reduce #'* shape)))
    (make-tensor tensor-data shape)))

(defmacro gpu-allocate (var &rest args)
  "Create and allocate GPU memory for a tensor, storing its device-pointer in the given var."
  `(let ((,var (cublas:allocate (reduce #'* ',args))))
     ,var))

(defmacro gpu-deallocate (var)
  "Deallocate GPU memory associated with a tensor's device-pointer."
  `(cublas:deallocate ,var))

(defun tensor-on-gpu-p (tensor)
  "Verify if the provided tensor is stored on the GPU."
  (eql (tensor-data tensor) :gpu))

(defun move-to-gpu (tensor)
  "Move the provided tensor to the GPU."
  (if (tensor-on-gpu-p tensor)
      tensor
      (make-instance 'tensor :data :gpu
                            :shape (tensor-shape tensor))))

(defun move-to-cpu (tensor)
  "Move the provided tensor back to the CPU."
  (if (tensor-on-gpu-p tensor)
      (make-instance 'tensor :data (copy-seq (tensor-data tensor)) :shape (tensor-shape tensor))
      tensor))

(defun tensor-ensure-on-gpu (tensor)
  "Ensures tensor's data is on GPU. If not, sends the data to GPU."
  (unless (tensor-on-gpu-p tensor)
    (setf (tensor-gpu-pointer tensor) (to-gpu tensor))
    (setf (tensor-data tensor) nil))
  tensor)

(defun tensor-ensure-on-cpu (tensor)
  "Ensures tensor's data is on CPU. If not, retrieves the data from GPU."
  (when (tensor-on-gpu-p tensor)
    (setf (tensor-data tensor) (from-gpu (tensor-gpu-pointer tensor) (tensor-shape tensor)))
    (gpu-deallocate (tensor-gpu-pointer tensor))
    (setf (tensor-gpu-pointer tensor) nil))
  tensor)