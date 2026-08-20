Below is a translation table organized by theme, then a list of false friends (the real source of confusion), then advice on what to prioritize.

## The basic setup

Almost every ML paper instantiates the same four slots: a parametrized family, a loss, a regularizer, and a first-order method. Once you see this, most of the vocabulary attaches to one of them.

- **Features** — the coordinates of the input $x \in \mathbb{R}^d$. A **feature map** is literally $\phi : \mathcal{X} \to \mathcal{H}$, the same object as in RKHS theory. "Feature engineering" = choosing the coordinates by hand.
- **Label / target / ground truth** — the value $y$ of the function to be learned.
- **Sample / example / instance** — usually a *single* pair $(x_i, y_i)$, not the whole collection. (Statistics uses "sample" for the collection; this clash is constant.)
- **Dataset** — a finite family $\{(x_i,y_i)\}_{i\le n}$, best thought of as the empirical measure $\hat\mu_n = \frac1n\sum_i \delta_{(x_i,y_i)}$, assumed i.i.d. from an unknown $\mu$.
- **Model / architecture / hypothesis class** — a parametrized family $\{f_\theta : \theta\in\Theta\}$, i.e. a subset of a function space *together with* a chosen parametrization (which matters, because optimization happens in $\Theta$, not in the function space). "Model" also sometimes denotes one fitted element; context disambiguates.
- **Parameters / weights** $\theta$ vs. **hyperparameters** — the latter are anything not determined by the optimizer: $\lambda$, step size, depth, width. Parameters of the *procedure* rather than of the function.
- **Loss** $\ell(f(x), y) \ge 0$ — pointwise discrepancy. **Risk** $R(f) = \mathbb{E}_\mu[\ell(f(x),y)]$ — a functional on the hypothesis class. **Empirical risk** $\hat R_n(f)$ — the same integral against $\hat\mu_n$. **Objective / cost / criterion** — what is actually minimized, typically $\hat R_n + \lambda\Omega(\theta)$.
- **Training / fitting / learning** — numerically minimizing that functional over $\Theta$.
- **Regularization** — the penalty $\lambda\Omega(\theta)$; Tikhonov, essentially unchanged. "$L^2$ regularization" = ridge, "$L^1$" = lasso, "weight decay" = $L^2$ implemented as a shrinkage step inside the iteration.
- **Generalization** — control of $R(f) - \hat R_n(f)$; i.e. uniform laws of large numbers over $\mathcal{F}$, empirical process theory.
- **Capacity / expressivity** — the size of $\mathcal{F}$, measured by VC dimension, Rademacher complexity, covering numbers. **Universal approximation** = density in $C(K)$ with sup norm.
- **Overfitting / underfitting** — estimation error dominating vs. approximation error dominating. The **bias–variance decomposition** is a Pythagorean identity for the $L^2$ risk.

## Optimization

This is the most mathematically stable part of the vocabulary; the words mean what you'd expect.

- **Gradient descent** — explicit Euler discretization of the gradient flow $\dot\theta = -\nabla L(\theta)$. **Learning rate** = step size $\eta$.
- **SGD** — $\nabla \hat R_n$ replaced by an unbiased estimator computed from a random subsample. **Minibatch** = that subsample; **batch size** = its cardinality; **epoch** = one full pass through the data.
- **Momentum, Adam, RMSProp** — accelerated and diagonally preconditioned variants.
- **Backpropagation** — the chain rule evaluated in reverse mode: since the codomain is $\mathbb{R}$, associating the product of Jacobians right-to-left costs $O(1)$ forward passes instead of $O(\dim\Theta)$.
- **Convergence** — usually means "the loss curve flattened," not a theorem. Be alert to this.

## Architectures

- **Layer** — one map in a composition. A feedforward net is $f = A_L \circ \sigma \circ A_{L-1} \circ \cdots \circ \sigma \circ A_1$ with $A_i$ affine and $\sigma$ a fixed nonlinearity applied coordinatewise. **Depth** = $L$, **width** = the intermediate dimensions.
- **Activation** — the nonlinearity $\sigma$. **ReLU** $= \max(0,\cdot)$, so a ReLU network is a piecewise-affine map subordinate to a polyhedral decomposition of $\mathbb{R}^d$ — often the most useful mental picture.
- **Logits** — a vector in $\mathbb{R}^k$ pushed to the simplex by **softmax**, which is the Gibbs measure $z \mapsto (e^{z_i}/\sum_j e^{z_j})_i$, a smoothed $\arg\max$. The scalar **logit** is $\log\frac{p}{1-p}$, inverse of the sigmoid.
- **Embedding / representation / latent space** — an intermediate codomain in a factorization $f = g\circ h$; "embedding" means *vector representation*, with no injectivity or immersion implied.
- **Attention** — an input-dependent linear operator: weights come from a softmax of a bilinear form in the inputs, and the output is a weighted average. A kernel smoother whose kernel is itself learned and data-dependent.
- **Normalization (batch/layer norm)** — affine rescaling of intermediate vectors to prescribed empirical mean and variance.

## Probability and statistics as used

- **Likelihood, prior, posterior, MLE, MAP** — standard. The key dictionary entry: minimizing **cross-entropy loss** = maximizing log-likelihood for a categorical model; squared loss = Gaussian likelihood; regularization = log-prior.
- **KL divergence** — relative entropy $D(P\|Q) = \int \log\frac{dP}{dQ}\,dP$.
- **Sample complexity** — the $n$ needed for $\varepsilon$-accuracy with probability $\ge 1-\delta$ (the PAC framework).
- **Distribution shift** — train and test measures differ; the i.i.d. hypothesis fails.
- **Curse of dimensionality** — covering numbers of $[0,1]^d$ grow like $\varepsilon^{-d}$.
- **Accuracy, precision, recall, F1, AUC** — summary functionals of the confusion matrix.

## Experimental vocabulary (no mathematical content, but load-bearing)

- **Baseline** — a comparator method, often deliberately trivial (predict the mean, predict the majority class) or the previous best. The analogue of "the obvious estimate you must beat." A claim without a baseline is meaningless, so this word carries a lot of the epistemic weight.
- **Ablation** — removing one component and re-measuring, to argue necessity. A control experiment; the closest thing to a "only if" direction.
- **Benchmark** — a fixed dataset and metric everyone reports on. **SOTA** = best published number on it.
- **Inference** — *evaluating the trained function on a new input.* Not statistical inference. Very common in engineering contexts.

## False friends

These are where a pure background actively misleads:

| Word | What it does **not** mean | What it means |
|---|---|---|
| **tensor** | multilinear functional; transformation law | a multidimensional array, nothing more |
| **bias** | — | (a) the offset $b$ in $x\mapsto Wx+b$; (b) $\mathbb{E}[\hat\theta]-\theta$; (c) **inductive bias**: the restriction/prior making the problem well-posed |
| **embedding** | injective immersion | any map into $\mathbb{R}^d$, usually learned |
| **kernel** | one thing | RKHS kernel, *or* convolution filter, *or* a GPU subroutine |
| **inference** | estimating parameters | forward evaluation of a fixed model |
| **linear** | linear | very often affine |
| **sample** | the collection | one data point |
| **dimension** | — | often the length of an array axis |
| **normal** | Gaussian, or orthogonal | often just "rescaled" |
| **manifold**, **orthogonal**, **spectral**, **compact**, **smooth** | the definitions | frequently used metaphorically or loosely |
| **optimal** | a minimizer | "best configuration we tried" |

## What to focus on

**1. Learn the four-slot template and use it aggressively.** When reading a paper, ask: what is $\mathcal{F}$, what is $\ell$, what is $\Omega$, what is the optimizer? Nearly every contribution is a modification of exactly one slot, and papers rarely say which. This single habit compresses the literature enormously.

**2. Track shapes, not equations.** Understanding an architecture is mostly bookkeeping about which $\mathbb{R}^{n_1\times\cdots\times n_k}$ maps to which. Write out the types explicitly; the prose will never do it for you, and index conventions are inconsistent between papers.

**3. Recalibrate for a different epistemology.** ML claims are supported by benchmark numbers plus ablations, not proofs. A theorem in a deep learning paper is usually an intuition pump under hypotheses that don't hold in the experiments — which is understood by everyone and stated by nobody. Read theorems for their mechanism, not their scope.

**4. Get the probability-to-loss dictionary automatic.** The correspondence between losses and negative log-likelihoods, and between regularizers and priors, explains why particular objectives are chosen and turns arbitrary-looking formulas into derivations.

**5. Accept imprecision provisionally.** The strongest temptation with your background is to demand definitions before proceeding. Much of the vocabulary is genuinely informal and cannot be sharpened without changing the subject. Formalize later, selectively.

**6. Reading, roughly in order of congeniality to your training:**
- Shalev-Shwartz & Ben-David, *Understanding Machine Learning* — the theory book that reads most like mathematics; best entry point for the vocabulary of risk, capacity, and generalization.
- Wainwright, *High-Dimensional Statistics* — for concentration, empirical processes, and the actual technical machinery.
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning* — for the classical-methods vocabulary that everyone assumes.
- Murphy, *Probabilistic Machine Learning* (2 vols) — encyclopedic; good as a reference for terminology rather than a cover-to-cover read.
- Karpathy's "Neural Networks: Zero to Hero" videos — for the engineering half of the vocabulary (shapes, batching, training loops), which no theory book teaches.

If you want one exercise that fixes the language faster than reading: implement least squares, then logistic regression, then a two-layer network from scratch with explicit gradients. Every term above appears in that exercise, and the ambiguous ones resolve themselves once you've had to write them down as code.