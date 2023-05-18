#!/bin/bash

# Create directories
mkdir -p src/core
mkdir -p src/layers
mkdir -p src/activations
mkdir -p src/optimizers
mkdir -p src/losses
mkdir -p src/transformers
mkdir -p src/utils
mkdir -p tests/core
mkdir -p tests/layers
mkdir -p tests/activations
mkdir -p tests/optimizers
mkdir -p tests/losses
mkdir -p tests/transformers
mkdir -p examples
mkdir -p docs

# Create files
touch src/core/tensor.lisp
touch src/core/gpu.lisp
touch src/core/autograd.lisp
touch src/layers/base.lisp
touch src/layers/linear.lisp
touch src/layers/convolutional.lisp
touch src/layers/recurrent.lisp
touch src/layers/attention.lisp
touch src/layers/multihead_attention.lisp
touch src/activations/base.lisp
touch src/activations/relu.lisp
touch src/activations/sigmoid.lisp
touch src/optimizers/base.lisp
touch src/optimizers/sgd.lisp
touch src/optimizers/adam.lisp
touch src/losses/base.lisp
touch src/losses/mse.lisp
touch src/losses/cross_entropy.lisp
touch src/transformers/transformer_layer.lisp
touch src/transformers/encoder.lisp
touch src/transformers/decoder.lisp
touch src/utils/data_loader.lisp
touch src/utils/debugging.lisp
touch tests/core/test_tensor.lisp
touch tests/core/test_gpu.lisp
touch tests/core/test_autograd.lisp
touch tests/layers/test_layers.lisp
touch tests/activations/test_activations.lisp
touch tests/optimizers/test_optimizers.lisp
touch tests/losses/test_losses.lisp
touch tests/transformers/test_transformers.lisp
touch examples/example1.lisp
touch examples/example2.lisp
touch docs/getting_started.md
touch docs/tensor.md
touch docs/autograd.md
touch docs/layers.md
touch docs/activations.md
touch docs/optimizers.md
touch docs/losses.md
touch docs/transformers.md
# touch README.md
touch .gitignore
touch neuralisp.asd

# Git setup
git init

# Optional: Configure user details
# (Replace "Your Name" and "your.email@example.com" with your information)
git config user.name "ck46"
git config user.email "prof.chakas@gmail.com"

git add .
git commit -m "Initial commit"