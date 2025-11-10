(defpackage :neuralisp.core.tensor
  (:use :cl)
  (:export :tensor
           :tensor-data
           :tensor-shape
           :tensor-strides
           :tensor-size
           :tensor-dtype
           :tensor-gpu-pointer
           :make-tensor
           :tensor-copy
           :tensor-reshape
           :tensor-broadcast-to
           :tensor-zeros-like
           :tensor-ones-like
           :tensor-add
           :tensor-subtract
           :tensor-multiply
           :tensor-divide
           :tensor-matmul
           :tensor-sum
           :tensor-mean))
(in-package :neuralisp.core.tensor)

;;;; Utility helpers

(defparameter *dtype-order* '(:int :float32 :float64)
  "Ordered list describing promotion precedence for supported numeric types.")

(defun %dtype-index (dtype)
  (position dtype *dtype-order* :test #'eq))

(defun promote-dtype (&rest dtypes)
  "Return the most expressive dtype from DTYPES based on *DTYPE-ORDER*."
  (let ((resolved nil))
    (dolist (dtype dtypes)
      (when dtype
        (let ((idx (%dtype-index dtype)))
          (unless idx
            (error "Unknown dtype ~A." dtype))
          (when (or (null resolved)
                    (> idx (%dtype-index resolved)))
            (setf resolved dtype)))))
    (or resolved :float64)))

(defun dtype-of (value)
  "Infer the dtype keyword for VALUE."
  (typecase value
    (double-float :float64)
    (single-float :float32)
    (float :float64)
    ((or ratio complex) :float64)
    (t :int)))

(defun coerce-value-to-dtype (value dtype)
  "Coerce VALUE to the numeric representation described by DTYPE."
  (case dtype
    (:int (round value))
    (:float32 (coerce value 'single-float))
    (:float64 (coerce value 'double-float))
    (t value)))

(defun compute-size (shape)
  "Return the total number of elements for SHAPE. Scalars have size 1."
  (if (null shape)
      1
      (reduce #'* shape :initial-value 1)))

(defun compute-strides (shape)
  "Compute row-major strides for SHAPE."
  (if (null shape)
      '()
      (let* ((rank (length shape))
             (strides (make-array rank :element-type 'fixnum))
             (running 1))
        (dotimes (i rank)
          (let ((axis (- rank i 1)))
            (setf (aref strides axis) running)
            (setf running (* running (nth axis shape)))))
        (loop for i below rank collect (aref strides i)))))

(defun offset-from-coords (coords strides)
  "Compute the linear offset for COORDS given STRIDES."
  (loop for c in coords
        for s in strides
        sum (* c s)))

(defun normalize-sequence (seq)
  "Return SEQ as a list without modifying nested sequences."
  (cond
    ((listp seq) seq)
    ((vectorp seq) (loop for i below (length seq) collect (aref seq i)))
    ((sequencep seq) (loop for elem across seq collect elem))
    (t (error "Unsupported sequence type ~A." (type-of seq)))))

(defun infer-shape-from-sequence (data)
  "Infer a nested sequence shape.

DATA should be a number or a (possibly nested) sequence."
  (cond
    ((or (numberp data) (complexp data)) '())
    ((and (sequencep data)
          (zerop (length data)))
     (list 0))
    ((sequencep data)
     (let* ((normalized (normalize-sequence data))
            (child-shape (infer-shape-from-sequence (first normalized))))
       (dolist (elem (rest normalized))
         (unless (equal child-shape (infer-shape-from-sequence elem))
           (error "Irregular tensor data: ~A" data)))
       (cons (length normalized) child-shape)))
    (t (error "Cannot infer shape from ~A" data))))

(defun flatten-sequence (data)
  "Flatten nested DATA into a simple list preserving row-major order."
  (cond
    ((or (numberp data) (complexp data)) (list data))
    ((sequencep data)
     (mapcan #'flatten-sequence (normalize-sequence data)))
    (t (error "Unsupported data element ~A" data))))

(defun ensure-shape-compatible (shape size)
  (unless (= (compute-size shape) size)
    (error "Shape ~A is incompatible with data of length ~A" shape size))
  shape)

(defun normalize-axis (axis rank)
  "Normalize AXIS (or list of axes) into an increasing list for tensors with RANK."
  (when (null axis)
    (return-from normalize-axis nil))
  (let ((axes (if (listp axis) axis (list axis)))
        (result '()))
    (dolist (ax axes)
      (let* ((normalized (if (< ax 0) (+ rank ax) ax)))
        (when (or (< normalized 0) (>= normalized rank))
          (error "Axis ~A out of bounds for rank ~A" ax rank))
        (pushnew normalized result :test #'=)))
    (sort result #'<)))

(defun broadcast-shape (&rest shapes)
  "Compute the broadcasted shape for SHAPES."
  (let ((result '()))
    (dolist (shape shapes)
      (let* ((rev (reverse shape))
             (max-len (max (length rev) (length result))))
        (loop for i from 0 below max-len do
          (let* ((da (if (< i (length rev)) (nth i rev) 1))
                 (db (if (< i (length result)) (nth i result) 1))
                 (dim (cond
                        ((= da db) da)
                        ((= da 1) db)
                        ((= db 1) da)
                        (t (error "Shapes ~A cannot be broadcast with ~A" shape (reverse result))))))
            (if (< i (length result))
                (setf (nth i result) dim)
                (push dim result))))))
    (reverse result)))

(defun compute-broadcast-strides (input-shape input-strides target-shape)
  "Return strides matching TARGET-SHAPE for broadcasting INPUT." 
  (let* ((len-target (length target-shape))
         (len-input (length input-shape))
         (offset (- len-target len-input))
         (result (make-array len-target :element-type 'fixnum)))
    (dotimes (axis len-target)
      (let* ((dim-target (nth axis target-shape))
             (dim-input (if (< axis offset)
                            1
                            (nth (- axis offset) input-shape)))
             (stride (cond
                       ((< axis offset) 0)
                       ((= dim-input dim-target)
                        (nth (- axis offset) input-strides))
                       ((= dim-input 1) 0)
                       (t (error "Cannot broadcast shape ~A to ~A" input-shape target-shape)))))
        (setf (aref result axis) stride)))
    (loop for i below len-target collect (aref result i))))

(defun iterate-indices (shape fn)
  "Call FN with each index tuple for SHAPE in row-major order."
  (labels ((walk (axis prefix)
             (if (= axis (length shape))
                 (funcall fn (nreverse prefix))
                 (dotimes (i (nth axis shape))
                   (walk (1+ axis) (cons i prefix))))))
    (if (null shape)
        (funcall fn '())
        (walk 0 '()))))

;;;; Tensor class

(defclass tensor ()
  ((data :initarg :data
         :accessor tensor-data
         :type (simple-array number (*))
         :documentation "Flattened row-major storage for the tensor.")
   (shape :initarg :shape
          :accessor tensor-shape
          :type list
          :documentation "List describing the tensor's extents. Scalars use NIL.")
   (strides :initarg :strides
            :accessor tensor-strides
            :type list
            :documentation "Row-major strides for fast offset computation.")
   (size :initarg :size
         :accessor tensor-size
         :type fixnum
         :documentation "Total number of tensor elements.")
   (dtype :initarg :dtype
          :accessor tensor-dtype
          :type keyword
          :documentation "Logical dtype associated with the tensor's values.")
   (gpu-pointer :initarg :gpu-pointer
                :initform nil
                :accessor tensor-gpu-pointer
                :documentation "Backend specific device allocation.")))

(defun %coerce-data-vector (flat-data dtype)
  (let* ((size (length flat-data))
         (array (make-array size :element-type 'number)))
    (loop for i from 0 below size
          for value in flat-data do
            (setf (aref array i) (coerce-value-to-dtype value dtype)))
    array))

(defun %make-tensor-from-vector (data shape dtype)
  (let* ((final-shape (copy-list shape))
         (size (compute-size final-shape))
         (strides (compute-strides final-shape)))
    (ensure-shape-compatible final-shape (length data))
    (make-instance 'tensor
                   :data data
                   :shape final-shape
                   :strides strides
                   :size size
                   :dtype dtype)))

(defun make-tensor (&key data shape initializer (initial-element 0) dtype (copy-data t) (on-gpu nil))
  "Construct a tensor using either raw DATA or SHAPE with INITIALIZER.

DATA may be a tensor, number or nested sequence. When DATA is supplied the
shape is inferred unless SHAPE is explicitly provided. When SHAPE is supplied
without DATA the tensor is filled using INITIALIZER (a function receiving the
index tuple) or INITIAL-ELEMENT.

DTYPE controls numeric coercion and defaults to the promoted dtype from the
input data or INITIAL-ELEMENT. If ON-GPU is true and the GPU subsystem is
available the tensor is uploaded after creation."
  (let ((source-data nil)
        (source-shape nil)
        (source-dtype dtype))
    (cond
      ((typep data 'tensor)
       (setf source-shape (or shape (tensor-shape data)))
       (setf source-dtype (or dtype (tensor-dtype data)))
       (setf source-data (if copy-data
                             (copy-seq (tensor-data data))
                             (tensor-data data))))
      (data
       (cond
         ((arrayp data)
          (unless shape
            (setf shape (array-dimensions data)))
          (setf source-shape shape)
          (let ((flat (loop for i below (array-total-size data)
                            collect (row-major-aref data i))))
            (setf source-dtype (or dtype (reduce #'promote-dtype flat :initial-value :int :key #'dtype-of)))
            (setf source-data (%coerce-data-vector flat source-dtype))))
         ((sequencep data)
          (let* ((flat (flatten-sequence data))
                 (inferred (infer-shape-from-sequence data)))
            (setf source-shape (ensure-shape-compatible (or shape inferred) (length flat)))
            (setf source-dtype (or dtype (reduce #'promote-dtype flat :initial-value :int :key #'dtype-of)))
            (setf source-data (%coerce-data-vector flat source-dtype))))
         ((or (numberp data) (complexp data))
          (setf source-shape (or shape '()))
          (setf source-dtype (or dtype (dtype-of data)))
          (setf source-data (%coerce-data-vector (list data) source-dtype)))
         (t (error "Unsupported tensor data ~A" data))))
      (shape
       (setf source-shape shape)
       (let* ((size (compute-size shape))
              (values (make-array size :element-type 'number))
              (fill-fn (or initializer (lambda (&rest _) initial-element)))
              (index 0)
              (element-dtype dtype))
         (iterate-indices shape
           (lambda (coords)
             (let ((raw (funcall fill-fn coords)))
               (unless element-dtype
                 (setf element-dtype (dtype-of raw)))
               (setf (aref values index)
                     (coerce-value-to-dtype raw (or element-dtype :float64)))
               (incf index))))
         (setf source-data values)
         (setf source-dtype (or element-dtype (dtype-of initial-element) :float64))))
      (t (error "make-tensor requires DATA or SHAPE")))
    (let* ((data-vector (if (and (arrayp source-data)
                                 (typep source-data '(simple-array number (*)))
                                 copy-data)
                            (copy-seq source-data)
                            (or source-data
                                (make-array (compute-size source-shape) :element-type 'number))))
           (tensor (%make-tensor-from-vector data-vector source-shape source-dtype)))
        (when on-gpu
          (let ((fn (and (find-package :neuralisp.core.gpu)
                          (find-symbol "TENSOR-ENSURE-ON-GPU" :neuralisp.core.gpu))))
            (when fn
              (funcall fn tensor))))
        tensor)))

(defun tensor-copy (tensor)
  "Deep copy of TENSOR preserving dtype and shape."
  (make-tensor :data (copy-seq (tensor-data tensor))
               :shape (tensor-shape tensor)
               :dtype (tensor-dtype tensor)))

(defun tensor-reshape (tensor new-shape)
  "Return a reshaped copy of TENSOR with NEW-SHAPE."
  (ensure-shape-compatible new-shape (tensor-size tensor))
  (%make-tensor-from-vector (copy-seq (tensor-data tensor))
                            new-shape
                            (tensor-dtype tensor)))

(defun tensor-broadcast-to (tensor target-shape)
  "Broadcast TENSOR to TARGET-SHAPE, returning a new tensor."
  (let* ((input-shape (tensor-shape tensor))
         (input-strides (tensor-strides tensor))
         (result-strides (compute-strides target-shape))
         (broadcast-strides (compute-broadcast-strides input-shape input-strides target-shape))
         (result-size (compute-size target-shape))
         (result-data (make-array result-size :element-type 'number)))
    (labels ((walk (axis offset-input offset-output)
               (if (= axis (length target-shape))
                   (setf (aref result-data offset-output)
                         (aref (tensor-data tensor) offset-input))
                   (let* ((dim (nth axis target-shape))
                          (stride-input (nth axis broadcast-strides))
                          (stride-output (nth axis result-strides)))
                     (dotimes (i dim)
                       (walk (1+ axis)
                             (+ offset-input (* i stride-input))
                             (+ offset-output (* i stride-output))))))))
      (walk 0 0 0))
    (%make-tensor-from-vector result-data target-shape (tensor-dtype tensor)))

(defun tensor-zeros-like (tensor &key dtype)
  (make-tensor :shape (tensor-shape tensor)
               :initial-element 0
               :dtype (or dtype (tensor-dtype tensor))))

(defun tensor-ones-like (tensor &key dtype)
  (make-tensor :shape (tensor-shape tensor)
               :initial-element 1
               :dtype (or dtype (tensor-dtype tensor))))

(defun %elementwise-binary (tensor-a tensor-b op)
  (let* ((shape-a (tensor-shape tensor-a))
         (shape-b (tensor-shape tensor-b))
         (result-shape (broadcast-shape shape-a shape-b))
         (dtype (promote-dtype (tensor-dtype tensor-a) (tensor-dtype tensor-b)))
         (result-size (compute-size result-shape))
         (result-data (make-array result-size :element-type 'number))
         (result-strides (compute-strides result-shape))
         (broadcast-a (compute-broadcast-strides shape-a (tensor-strides tensor-a) result-shape))
         (broadcast-b (compute-broadcast-strides shape-b (tensor-strides tensor-b) result-shape)))
    (labels ((walk (axis offset-a offset-b offset-r)
               (if (= axis (length result-shape))
                   (let* ((value-a (aref (tensor-data tensor-a) offset-a))
                          (value-b (aref (tensor-data tensor-b) offset-b))
                          (result (funcall op (coerce-value-to-dtype value-a dtype)
                                           (coerce-value-to-dtype value-b dtype))))
                     (setf (aref result-data offset-r)
                           (coerce-value-to-dtype result dtype)))
                   (let* ((dim (nth axis result-shape))
                          (stride-a (nth axis broadcast-a))
                          (stride-b (nth axis broadcast-b))
                          (stride-r (nth axis result-strides)))
                     (dotimes (i dim)
                       (walk (1+ axis)
                             (+ offset-a (* i stride-a))
                             (+ offset-b (* i stride-b))
                             (+ offset-r (* i stride-r))))))))
      (walk 0 0 0 0))
    (%make-tensor-from-vector result-data result-shape dtype))

(defun tensor-add (tensor-a tensor-b)
  (%elementwise-binary tensor-a tensor-b #'+))

(defun tensor-subtract (tensor-a tensor-b)
  (%elementwise-binary tensor-a tensor-b #'-))

(defun tensor-multiply (tensor-a tensor-b)
  (%elementwise-binary tensor-a tensor-b #'*))

(defun tensor-divide (tensor-a tensor-b)
  (%elementwise-binary tensor-a tensor-b (lambda (a b) (/ a b))))

(defun tensor-matmul (tensor-a tensor-b &key (transpose-a nil) (transpose-b nil))
  "Matrix multiply TENSOR-A and TENSOR-B, optionally transposing inputs."
  (let* ((shape-a (tensor-shape tensor-a))
         (shape-b (tensor-shape tensor-b)))
    (unless (= (length shape-a) 2)
      (error "tensor-matmul expects rank-2 tensors for A, got ~A" shape-a))
    (unless (= (length shape-b) 2)
      (error "tensor-matmul expects rank-2 tensors for B, got ~A" shape-b))
    (let* ((rows-a (if transpose-a (second shape-a) (first shape-a)))
           (cols-a (if transpose-a (first shape-a) (second shape-a)))
           (rows-b (if transpose-b (second shape-b) (first shape-b)))
           (cols-b (if transpose-b (first shape-b) (second shape-b))))
      (unless (= cols-a rows-b)
        (error "Inner dimensions mismatch for matmul: ~A and ~A" shape-a shape-b))
      (let* ((dtype (promote-dtype (tensor-dtype tensor-a) (tensor-dtype tensor-b)))
             (result-data (make-array (* rows-a cols-b) :element-type 'number))
             (stride-a (tensor-strides tensor-a))
             (stride-b (tensor-strides tensor-b)))
        (dotimes (i rows-a)
          (dotimes (j cols-b)
            (let ((acc 0d0))
              (dotimes (k cols-a)
                (let* ((offset-a (if transpose-a
                                     (+ (* k (first stride-a)) (* i (second stride-a)))
                                     (+ (* i (first stride-a)) (* k (second stride-a)))))
                       (offset-b (if transpose-b
                                     (+ (* j (first stride-b)) (* k (second stride-b)))
                                     (+ (* k (first stride-b)) (* j (second stride-b)))))
                       (val-a (coerce-value-to-dtype (aref (tensor-data tensor-a) offset-a) dtype))
                       (val-b (coerce-value-to-dtype (aref (tensor-data tensor-b) offset-b) dtype)))
                  (setf acc (+ acc (* val-a val-b)))))
              (setf (aref result-data (+ (* i cols-b) j))
                    (coerce-value-to-dtype acc dtype)))))
        (%make-tensor-from-vector result-data (list rows-a cols-b) dtype))))

(defun tensor-sum (tensor &key (axis nil) (keepdims nil))
  "Sum TENSOR optionally along AXIS. When KEEPDIMS is NIL and the reduction
covers all axes a scalar number is returned."
  (let* ((shape (tensor-shape tensor))
         (rank (length shape)))
    (if (null axis)
        (let* ((total (loop for i below (tensor-size tensor)
                            sum (aref (tensor-data tensor) i)))
               (value (coerce-value-to-dtype total (tensor-dtype tensor))))
          (if keepdims
              (make-tensor :shape (make-list rank 1)
                           :initial-element value
                           :dtype (tensor-dtype tensor))
              value))
        (let* ((axes (normalize-axis axis rank))
               (result-shape (if keepdims
                                  (loop for i from 0 below rank
                                        collect (if (member i axes) 1 (nth i shape)))
                                  (loop for i from 0 below rank
                                        unless (member i axes) collect (nth i shape))))
               (result-size (compute-size result-shape))
               (result-data (make-array (max result-size 1) :initial-element 0 :element-type 'number))
               (result-strides (compute-strides result-shape)))
          (iterate-indices shape
            (lambda (coords)
              (let* ((input-offset (offset-from-coords coords (tensor-strides tensor)))
                     (value (aref (tensor-data tensor) input-offset))
                     (target-coords (if keepdims
                                        (loop for i from 0 below rank
                                              collect (if (member i axes) 0 (nth i coords)))
                                        (loop for i from 0 below rank
                                              unless (member i axes) collect (nth i coords))))
                     (target-offset (if (null target-coords)
                                        0
                                        (offset-from-coords target-coords result-strides))))
                (setf (aref result-data target-offset)
                      (+ (aref result-data target-offset)
                         (coerce-value-to-dtype value (tensor-dtype tensor)))))))
          (if (and (not keepdims) (zerop (length result-shape)))
              (aref result-data 0)
              (%make-tensor-from-vector result-data result-shape (tensor-dtype tensor)))))))

(defun tensor-mean (tensor &key (axis nil) (keepdims nil))
  "Mean of TENSOR optionally along AXIS."
  (let* ((shape (tensor-shape tensor))
         (rank (length shape))
         (axes (normalize-axis axis rank))
         (count (if axes
                    (reduce #'* (mapcar (lambda (ax) (nth ax shape)) axes))
                    (tensor-size tensor)))
         (dtype (promote-dtype (tensor-dtype tensor) :float32)))
    (if axes
        (let ((sum (tensor-sum tensor :axis axes :keepdims keepdims)))
          (if (typep sum 'tensor)
              (let* ((result-data (make-array (tensor-size sum) :element-type 'number))
                     (source (tensor-data sum)))
                (dotimes (i (length source))
                  (setf (aref result-data i)
                        (coerce-value-to-dtype (/ (aref source i) count) dtype)))
                (make-tensor :data result-data :shape (tensor-shape sum) :dtype dtype :copy-data nil))
              (coerce-value-to-dtype (/ sum count) dtype)))
        (let* ((total (tensor-sum tensor :axis nil :keepdims keepdims)))
          (if (typep total 'tensor)
              (let* ((result-data (make-array (tensor-size total) :element-type 'number))
                     (source (tensor-data total)))
                (dotimes (i (length source))
                  (setf (aref result-data i)
                        (coerce-value-to-dtype (/ (aref source i) count) dtype)))
                (make-tensor :data result-data :shape (tensor-shape total) :dtype dtype :copy-data nil))
              (coerce-value-to-dtype (/ total count) dtype))))))
