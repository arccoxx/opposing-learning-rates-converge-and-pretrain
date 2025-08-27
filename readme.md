# Alternating Learning Rates in Neural Networks: Experiments and Implications:
This repository contains code and documentation for a series of experiments exploring the effects of alternating positive and negative learning rates during neural network training. These experiments were inspired by a conversation with Grok, an AI built by xAI, and cover topics from basic convergence in linear models to continual learning in autoencoders on synthetic data mimicking MNIST classes.
The key insight is that alternating learning rates (positive on even epochs, negative on odd epochs) can lead to slow but eventual convergence in convex problems, but behaves differently in non-convex settings. In continual learning, it promotes "shallower" minima, aiding faster adaptation to new tasks with less catastrophic forgetting compared to positive-only training.

##Experiments Overview

**Experiment 1: Linear Regression with Alternating Rates**
- **Setup**: A simple linear model (y = 2x) trained with SGD, alternating +0.01 and -0.01 learning rates.
- **Findings**: The model converges slowly to the global minimum due to partial undoing of progress, but does converge with enough epochs (loss ~0.003 after 1000 epochs).
- **Implications**: In convex landscapes, alternation causes oscillation but net descent; unsuitable for fast training but could regularize in some cases.

**Experiment 2: MLP on XOR with Large Alternating Rates**
- **Setup**: A small MLP with ReLU for XOR classification, using large alternating +1.0/-1.0 rates.
- **Findings**: Fails to converge, stabilizing at a poor loss (~0.693) with trivial predictions (all ~0.5). Positive-only converges quickly.
- **Implications**: In non-convex problems, alternation prevents escaping flat regions, highlighting risks in complex tasks.

**Experiment 3: Continual Learning with Autoencoder on Synthetic Classes**
- **Setup**: Autoencoder on two Gaussian-distributed "classes" (low vs. high values in 784 dims). Pretrain on Class 0 using batch gradient descent, then adapt to Class 1 using mini-batch.
- **Variants**:
  - Alternating pretrain + positive adapt: Slow pretrain, adapts in ~35 epochs, moderate forgetting (~0.150 loss on Class 0).
  - Negative-only pretrain + positive adapt: Diverges in pretrain, adapts faster (~25 epochs), but no retention of Class 0.
  - Positive-only pretrain + positive adapt: Fast pretrain, adapts slower (~40 epochs), more forgetting (~0.180 loss on Class 0).
- **Implications**: Alternating rates create less committed minima, balancing retention and adaptability in continual learning—potentially useful for lifelong learning systems, though slower initially.

## Implications
These experiments suggest alternating learning rates could be a novel regularization technique for continual learning, reducing overfitting to initial tasks and easing transfer. However, it slows convergence and fails in highly non-convex settings. Related research (e.g., machine unlearning with negative gradients) supports using negative steps for "forgetting," but alternation on the same task is underexplored. Future work could combine this with methods like Elastic Weight Consolidation for better continual performance. In practice, use small rates and monitor oscillations; not recommended for standard training.
Code
The following Python script (experiments.py) replicates all experiments. It requires PyTorch and NumPy (install via pip install torch numpy if needed). Run it with python experiments.py to see printed results.
