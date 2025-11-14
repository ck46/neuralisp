(defpackage :neuralisp.cognition.state
  (:use :cl)
  (:import-from :neuralisp.core.tensor
                :tensor
                :tensor-mean)
  (:export :cognitive-node
           :node-id
           :node-type
           :node-data
           :state-graph
           :state-nodes
           :state-edges
           :state-trace
           :state-cycle-count
           :advance-cycle
           :make-state-graph
           :ensure-node
           :get-node
           :collect-node-data
           :nodes-by-type
           :update-node-data
           :add-directed-edge
           :neighbors
           :record-trace
           :state-cycle-trace
           :ensure-control-node
           :register-state-value
           :lookup-state-value
           :update-state-value
           :tensor->symbolic
           :symbolic->tensor-compatible))

(in-package :neuralisp.cognition.state)

(defclass cognitive-node ()
  ((id :initarg :id
       :accessor node-id
       :documentation "Identifier used to retrieve the node from the state graph.")
   (type :initarg :type
         :accessor node-type
         :documentation "Symbol describing the semantic type for the node, e.g. :fact, :memory, :control.")
   (data :initarg :data
         :initform nil
         :accessor node-data
         :documentation "Arbitrary payload carried by the node. This can include tensors, symbolic facts, or metadata.")))

(defclass state-graph ()
  ((nodes :initform (make-hash-table :test 'equal)
          :accessor state-nodes
          :documentation "Lookup table for nodes keyed by their identifier.")
   (edges :initform (make-hash-table :test 'equal)
          :accessor state-edges
          :documentation "Adjacency list describing directed edges between nodes.")
   (trace :initform '()
          :accessor state-trace
          :documentation "Chronological record of module execution describing how the shared cognitive state evolves.")
   (cycle-count :initform 0
                :accessor state-cycle-count
                :documentation "Number of completed cognitive loop iterations.")))

(defun make-state-graph ()
  "Construct an empty STATE-GRAPH instance that can be shared across cognitive modules."
  (make-instance 'state-graph))

(defun ensure-node (graph id &key (type :generic) (data nil))
  "Ensure that a node with identifier ID exists in GRAPH.
If the node is already present it is returned untouched.
Otherwise a new node is created with TYPE and DATA and registered in GRAPH."
  (or (gethash id (state-nodes graph))
      (setf (gethash id (state-nodes graph))
            (make-instance 'cognitive-node :id id :type type :data data))))

(defun get-node (graph id)
  "Fetch the node associated with ID from GRAPH.
Returns NIL if the node does not exist."
  (gethash id (state-nodes graph)))

(defun update-node-data (graph id updater)
  "Apply UPDATER to the node stored under ID.
UPDATER is a function receiving the previous data value and returning the new value.
If the node does not exist it is created with a NIL payload prior to the update."
  (let* ((node (ensure-node graph id))
         (old-value (node-data node))
         (new-value (funcall updater old-value)))
    (setf (node-data node) new-value)
    node))

(defun add-directed-edge (graph source target &key (label :relation))
  "Insert a directed edge with LABEL between SOURCE and TARGET nodes.
If the edge list does not exist it is initialised."
  (let ((edges (gethash source (state-edges graph))))
    (unless edges
      (setf edges '())
      (setf (gethash source (state-edges graph)) edges))
    (push (list :target target :label label) edges)
    (setf (gethash source (state-edges graph)) edges))
  graph)

(defun neighbors (graph source &key label)
  "Return a list of neighbouring nodes from SOURCE optionally filtered by LABEL."
  (let ((edges (copy-list (gethash source (state-edges graph)))))
    (when label
      (setf edges (remove-if-not (lambda (edge)
                                   (eql (getf edge :label) label))
                                 edges)))
    (mapcar (lambda (edge)
              (get-node graph (getf edge :target)))
            edges)))

(defun collect-node-data (graph ids)
  "Return the NODE-DATA for every node identified by IDS, omitting missing nodes."
  (loop for id in ids
        for node = (get-node graph id)
        when node collect (node-data node)))

(defun nodes-by-type (graph type)
  "Gather nodes whose NODE-TYPE matches TYPE."
  (let ((result '()))
    (maphash (lambda (key node)
               (declare (ignore key))
               (when (eql (node-type node) type)
                 (push node result)))
             (state-nodes graph))
    (nreverse result)))

(defun record-trace (graph module info)
  "Append a trace entry describing MODULE's INFO to GRAPH.
INFO is typically an alist or plist capturing salient events so that reflective modules can introspect past behaviour."
  (let ((entry (list :cycle (state-cycle-count graph)
                     :module module
                     :info info
                     :timestamp (get-universal-time))))
    (setf (state-trace graph) (append (state-trace graph) (list entry)))
    entry))

(defun state-cycle-trace (graph cycle)
  "Return all trace entries that were recorded during CYCLE."
  (remove-if-not (lambda (entry)
                   (= (getf entry :cycle) cycle))
                 (state-trace graph)))

(defun advance-cycle (graph)
  "Advance the global cycle counter for GRAPH and return the updated value."
  (incf (slot-value graph 'cycle-count))
  (state-cycle-count graph))

(defun ensure-control-node (graph)
  "Ensure that the control node used to coordinate strategies exists and provide sensible defaults."
  (let ((node (ensure-node graph :control :type :control :data '((:strategy . :reactive)
                                                                 (:confidence . 0.0)))))
    node))

(defun register-state-value (graph key value)
  "Persist VALUE under KEY on the dedicated :control node."
  (let ((node (ensure-control-node graph)))
    (setf (node-data node) (acons key value (remove key (node-data node) :key #'car :test #'eq)))
    value))

(defun lookup-state-value (graph key &optional default)
  "Read KEY from the :control node returning DEFAULT when the key is unknown."
  (let* ((node (ensure-control-node graph))
         (pair (assoc key (node-data node))))
    (if pair (cdr pair) default)))

(defun update-state-value (graph key updater &optional default)
  "Apply UPDATER to the control value stored under KEY, initialising it with DEFAULT if necessary."
  (let* ((current (lookup-state-value graph key default))
         (next (funcall updater current)))
    (register-state-value graph key next)))

(defun tensor->symbolic (embedding)
  "Translate EMBEDDING, which may be a NEURALISP.CORE.TENSOR:TENSOR instance, into a symbolic summary.
When MAGICL-backed tensors are available we compute the mean activation as a robust scalar proxy.
Fallbacks gracefully handle plain lists or scalars which simplifies integration tests."
  (cond
    ((and embedding (typep embedding 'tensor))
     (handler-case
         (let ((mean (tensor-mean embedding)))
           (list :type :tensor-mean :value mean))
       (error ()
         (list :type :tensor-placeholder :value embedding))))
    ((and (listp embedding) (every #'numberp embedding))
     (let ((mean (/ (reduce #'+ embedding) (max 1 (length embedding)))))
       (list :type :numeric-list :value mean :raw embedding)))
    (t (list :type :scalar :value embedding))))

(defun symbolic->tensor-compatible (symbolic embedding)
  "Reconcile SYMBOLIC descriptors with the original EMBEDDING.
If EMBEDDING is a tensor we simply return it, otherwise we derive an updated numeric value from SYMBOLIC."
  (cond
    ((and embedding (typep embedding 'tensor)) embedding)
    ((and (plist-member symbolic :value)
          (numberp (getf symbolic :value)))
     (getf symbolic :value))
    (t embedding)))

(defun plist-member (plist key)
  "Utility helper for checking KEY membership in PLIST."
  (loop for (k v) on plist by #'cddr
        when (eql k key) do (return t)))
