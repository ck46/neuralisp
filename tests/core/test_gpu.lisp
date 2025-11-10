(defpackage :neuralisp.tests.core.gpu
  (:use :cl :fiveam)
  (:import-from :neuralisp.core.tensor
                :make-tensor
                :tensor-data
                :tensor-gpu-pointer
                :tensor-size)
  (:import-from :neuralisp.core.gpu
                :initialize-gpu
                :shutdown-gpu
                :tensor-ensure-on-gpu
                :tensor-ensure-on-cpu
                :tensor-on-gpu-p
                :move-to-cpu))
(in-package :neuralisp.tests.core.gpu)

(def-suite gpu-suite :description "GPU emulation tests")
(in-suite gpu-suite)

(defun tensor->list (tensor)
  (loop for i below (tensor-size tensor)
        collect (aref (tensor-data tensor) i)))

(test gpu-roundtrip
  (initialize-gpu)
  (let* ((tensor (make-tensor :data '((1 2) (3 4))))
         (original (tensor->list tensor)))
    (tensor-ensure-on-gpu tensor)
    (is-true (tensor-on-gpu-p tensor))
    (is-true (tensor-gpu-pointer tensor))
    ;; zero the host data to ensure the copy back fills it
    (dotimes (i (tensor-size tensor))
      (setf (aref (tensor-data tensor) i) 0))
    (tensor-ensure-on-cpu tensor)
    (is (equal (tensor->list tensor) original))
    (move-to-cpu tensor)
    (is (null (tensor-gpu-pointer tensor))))
  (shutdown-gpu))
