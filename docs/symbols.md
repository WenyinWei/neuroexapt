# Symbol Glossary {#symbols}

This document provides a comprehensive reference for all mathematical symbols and notation used throughout the Neuro Exapt framework.

## Core Information Theory Symbols

### Basic Information Measures

| Symbol | Description | Definition | Units |
|--------|-------------|------------|-------|
| $H(X)$ | Shannon entropy of random variable $X$ | $H(X) = -\sum_{x} p(x) \log p(x)$ | bits/nats |
| $H(X\|Y)$ | Conditional entropy of $X$ given $Y$ | $H(X\|Y) = -\sum_{x,y} p(x,y) \log p(x\|y)$ | bits/nats |
| $I(X;Y)$ | Mutual information between $X$ and $Y$ | $I(X;Y) = H(X) - H(X\|Y)$ | bits/nats |
| $D_{KL}(P\|\|Q)$ | Kullback-Leibler divergence | $D_{KL}(P\|\|Q) = \sum_x p(x) \log \frac{p(x)}{q(x)}$ | bits/nats |

### Network-Specific Variables

| Symbol | Description | Domain | Notes |
|--------|-------------|--------|-------|
| $L_i$ | Output of layer $i$ | $\mathbb{R}^{n_i}$ | Layer representation |
| $O$ | Network output | $\mathbb{R}^{n_o}$ | Final network output |
| $X$ | Network input | $\mathbb{R}^{n_x}$ | Input features |
| $Y$ | Target output | $\mathbb{R}^{n_y}$ | Ground truth labels |
| $\theta$ | Model parameters | $\mathbb{R}^p$ | All learnable parameters |

## Structural Evolution Symbols

### Layer Importance and Redundancy

| Symbol | Description | Range | Formula |
|--------|-------------|-------|---------|
| $I(L_i;O)$ | Importance of layer $i$ | $[0, \infty)$ | $H(O) - H(O\|L_i) \cdot \psi(\text{TaskType})$ |
| $R$ | Network redundancy | $[0, 1]$ | $1 - \frac{\sum_{i=1}^L I(L_i;O)}{H(O) \cdot \exp(-\lambda \cdot \text{Depth})}$ |
| $\psi(\cdot)$ | Task-aware weight function | $\mathbb{R}^+$ | Task-dependent scaling factor |
| $\lambda$ | Depth decay factor | $\mathbb{R}^+$ | Architecture-specific constant |

### Task-Specific Parameters

| Task Type | $\psi$ Value | $\lambda$ (ResNet) | $\lambda$ (Transformer) | $\delta$ Value |
|-----------|--------------|-------------------|------------------------|------------|
| Classification | 1.2 | 0.03 | 0.01 | 0.2 |
| Generation | 0.8 | 0.03 | 0.01 | 0.3 |
| Regression | 1.0 | 0.03 | 0.01 | 0.1 |
| Detection | 1.1 | 0.03 | 0.01 | 0.4 |

### Evolution Dynamics

| Symbol | Description | Default Value | Physical Meaning |
|--------|-------------|---------------|------------------|
| $S$ | Structural entropy | - | Information content of network structure |
| $S^*$ | Optimal structure | - | Information-theoretic optimum |
| $\alpha$ | Information retention coefficient | 0.7 | Weight for information preservation |
| $\beta$ | Structure variation coefficient | 0.3 | Weight for structural regularization |
| $t$ | Training time | $[0, \infty)$ | Continuous time parameter |

## Entropy Control Symbols

### Adaptive Thresholding

| Symbol | Description | Typical Value | Formula Component |
|--------|-------------|---------------|-------------------|
| $\tau$ | Current entropy threshold | $[0.1, 1.0]$ | - |
| $\tau_0$ | Initial threshold | 0.5 | Base threshold value |
| $\gamma$ | Entropy decay rate | 0.05 | Controls threshold reduction |
| $\delta$ | Task complexity coefficient | 0.2 | Scales complexity adjustment |

### Complexity Estimation

| Symbol | Description | Range | Notes |
|--------|-------------|-------|-------|
| $\text{TaskComplexity}$ | Estimated task difficulty | $[0, 1]$ | Normalized complexity score |
| $\text{DatasetSize}$ | Number of training samples | $\mathbb{N}$ | Used in complexity estimation |
| $\text{NumClasses}$ | Number of output classes | $\mathbb{N}$ | For classification tasks |
| $\text{InputDim}$ | Input dimensionality | $\mathbb{N}$ | Feature space size |
| $\text{Depth}$ | Network depth | $\mathbb{N}$ | Number of layers |

## Discrete Parameter Optimization

### Continuous Relaxation

| Symbol | Description | Domain | Purpose |
|--------|-------------|--------|---------|
| $k$ | Discrete parameter | $\mathbb{Z}$ | Actual discrete value (e.g., kernel size) |
| $\theta$ | Continuous relaxation | $\mathbb{R}$ | Learnable continuous parameter |
| $\sigma(\cdot)$ | Sigmoid function | $[0, 1]$ | $\sigma(x) = \frac{1}{1 + e^{-x}}$ |
| $k_{\max}$ | Maximum parameter value | $\mathbb{Z}$ | Upper bound for discrete parameter |
| $k_{\min}$ | Minimum parameter value | $\mathbb{Z}$ | Lower bound for discrete parameter |
| $\lfloor \cdot \rfloor$ | Floor function | $\mathbb{Z}$ | Discretization operation |

### Parameter Ranges

| Parameter Type | Typical Range | Symbol | Notes |
|----------------|---------------|--------|-------|
| Kernel size | $[1, 7]$ | $k_{\text{kernel}}$ | Convolutional layers |
| Stride | $[1, 3]$ | $k_{\text{stride}}$ | Downsampling factor |
| Dilation | $[1, 3]$ | $k_{\text{dilation}}$ | Receptive field expansion |
| Groups | $[1, 8]$ | $k_{\text{groups}}$ | Grouped convolution |

## Mathematical Operations

### Probability and Statistics

| Symbol | Description | Domain | Definition |
|--------|-------------|--------|------------|
| $p(x)$ | Probability mass/density | $[0, 1]$ or $\mathbb{R}^+$ | Probability of event $x$ |
| $p(x\|y)$ | Conditional probability | $[0, 1]$ | Probability of $x$ given $y$ |
| $\mathbb{E}[X]$ | Expected value | $\mathbb{R}$ | $\mathbb{E}[X] = \sum_x x \cdot p(x)$ |
| $\text{Var}[X]$ | Variance | $\mathbb{R}^+$ | $\text{Var}[X] = \mathbb{E}[(X - \mathbb{E}[X])^2]$ |
| $\mathcal{U}(a,b)$ | Uniform distribution | - | Uniform over interval $[a,b]$ |
| $\mathcal{N}(\mu,\sigma^2)$ | Normal distribution | - | Gaussian with mean $\mu$, variance $\sigma^2$ |

### Optimization and Calculus

| Symbol | Description | Context | Meaning |
|--------|-------------|---------|---------|
| $\nabla$ | Gradient operator | Optimization | Vector of partial derivatives |
| $\frac{\partial}{\partial x}$ | Partial derivative | Calculus | Rate of change with respect to $x$ |
| $\frac{d}{dt}$ | Total derivative | Dynamics | Time derivative |
| $\lim_{t \to \infty}$ | Limit as $t$ approaches infinity | Convergence | Asymptotic behavior |
| $\|\cdot\|_2$ | Euclidean norm | Linear algebra | $\|x\|_2 = \sqrt{\sum_i x_i^2}$ |

## Convergence Theory Symbols

### Theoretical Bounds

| Symbol | Description | Interpretation | Context |
|--------|-------------|----------------|---------|
| $C$ | Problem-dependent constant | Bound tightness | Convergence theorem |
| $\mu$ | Contraction coefficient | Convergence rate | Lyapunov analysis |
| $\epsilon$ | Bounded perturbation | Noise/error level | Stability analysis |
| $V(S)$ | Lyapunov function | Energy-like measure | $V(S) = \|S - S^*\|_2^2$ |
| $H_{\max}$ | Maximum entropy bound | Information capacity | Theoretical limit |

### Complexity Classes

| Symbol | Description | Growth Rate | Examples |
|--------|-------------|-------------|----------|
| $\mathcal{O}(n)$ | Linear complexity | $n$ | Information assessment |
| $\mathcal{O}(k^2)$ | Quadratic complexity | $k^2$ | Structural optimization |
| $\mathcal{O}(1)$ | Constant complexity | 1 | Discrete parameter mapping |
| $\mathcal{O}(\log n)$ | Logarithmic complexity | $\log n$ | Tree operations |

## Implementation Constants

### Numerical Stability

| Symbol | Description | Typical Value | Purpose |
|--------|-------------|---------------|---------|
| $\epsilon$ | Small constant | $10^{-10}$ | Avoid division by zero |
| $\epsilon_{\text{log}}$ | Logarithm stabilizer | $10^{-8}$ | Prevent $\log(0)$ |
| $\epsilon_{\text{grad}}$ | Gradient clipping | $10^{-6}$ | Numerical stability |

### Algorithmic Parameters

| Symbol | Description | Range | Tuning Guidelines |
|--------|-------------|-------|-------------------|
| $n_{\text{bins}}$ | Number of histogram bins | $[10, 100]$ | More bins for larger datasets |
| $w_{\text{ma}}$ | Moving average window | $[5, 50]$ | Smooth entropy estimates |
| $f_{\text{eval}}$ | Evaluation frequency | $[1, 10]$ | How often to assess importance |

## Special Notation

### Set Theory and Logic

| Symbol | Description | Example | Usage |
|--------|-------------|---------|-------|
| $\in$ | Element of | $x \in \mathbb{R}$ | Set membership |
| $\subset$ | Subset | $A \subset B$ | Set inclusion |
| $\cup$ | Union | $A \cup B$ | Set union |
| $\cap$ | Intersection | $A \cap B$ | Set intersection |
| $\forall$ | For all | $\forall x \in X$ | Universal quantifier |
| $\exists$ | There exists | $\exists x$ | Existential quantifier |
| $\implies$ | Implies | $A \implies B$ | Logical implication |
| $\iff$ | If and only if | $A \iff B$ | Logical equivalence |

### Functional Notation

| Symbol | Description | Domain/Codomain | Notes |
|--------|-------------|-----------------|-------|
| $f: X \to Y$ | Function from $X$ to $Y$ | - | Function definition |
| $f^{-1}$ | Inverse function | - | When it exists |
| $f \circ g$ | Function composition | $(f \circ g)(x) = f(g(x))$ | Order matters |
| $\arg\max_x f(x)$ | Argument that maximizes $f$ | - | Optimization |
| $\arg\min_x f(x)$ | Argument that minimizes $f$ | - | Optimization |

## Index and Summation Conventions

### Standard Indices

| Index | Typical Range | Usage | Description |
|-------|---------------|-------|-------------|
| $i$ | $1, \ldots, L$ | Layer index | Iterates over layers |
| $j$ | $1, \ldots, N$ | Sample index | Iterates over data samples |
| $k$ | $1, \ldots, K$ | Parameter index | Iterates over parameters |
| $t$ | $0, \ldots, T$ | Time index | Discrete time steps |
| $n$ | $1, \ldots, N$ | General index | Generic counter |

### Summation Notation

| Expression | Meaning | Equivalent |
|------------|---------|------------|
| $\sum_{i=1}^L$ | Sum over layers | $\sum_{i \in \{1,2,\ldots,L\}}$ |
| $\sum_{x \in X}$ | Sum over all $x$ in set $X$ | - |
| $\prod_{i=1}^n$ | Product notation | $\prod_{i \in \{1,2,\ldots,n\}}$ |

## Units and Scales

### Information Units

| Unit | Base | Conversion | Usage |
|------|------|------------|-------|
| bit | $\log_2$ | 1 bit = $\log(2)$ nats | Computer science |
| nat | $\log_e$ | 1 nat = $\frac{1}{\log(2)}$ bits | Natural logarithm |
| dit | $\log_{10}$ | 1 dit = $\log(10)$ nats | Decimal |

### Time Scales

| Scale | Description | Typical Range | Context |
|-------|-------------|---------------|---------|
| Epoch | Training iteration | $[1, 1000]$ | Discrete training |
| Step | Gradient update | $[1, 10^6]$ | Fine-grained updates |
| Continuous time | Mathematical abstraction | $[0, \infty)$ | Theoretical analysis |

---

*This symbol glossary provides comprehensive notation reference for all mathematical expressions in Neuro Exapt. For usage examples, see the @ref theory "Theoretical Foundation" and API documentation.* 