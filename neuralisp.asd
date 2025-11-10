(asdf:defsystem "neuralisp"
  :description "Symbolic-neural hybrid research playground"
  :serial t
  :components
  ((:module "src"
            :serial t
            :components
            ((:module "core"
                      :components ((:file "tensor")
                                   (:file "autograd")
                                   (:file "gpu")))
             (:module "cognition"
                      :serial t
                      :components ((:file "state")
                                   (:file "knowledge")
                                   (:file "memory")
                                   (:file "reasoning")
                                   (:file "learning")
                                   (:file "self-reflection")
                                   (:file "perception")
                                   (:file "actuation")
                                   (:file "loop")
                                   (:file "agent")))))))

(asdf:defsystem "neuralisp/tests"
  :depends-on ("neuralisp")
  :description "Integration tests for the cognitive loop"
  :serial t
  :components
  ((:module "tests"
            :components
            ((:module "cognition"
                      :components ((:file "test_cognitive_loop")))))))
