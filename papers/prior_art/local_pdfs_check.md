# Prior-art check: four local PDFs (2026-08-31)

Items checked: (a) regression/ML conditional-expectation estimator inside particle LSV calibration, (b) Girsanov/importance sampling, (c) damped leverage fixed-point iteration with convergence analysis.

HEADLINE: None of the four contains (a) a neural-network or plain-least-squares regression estimator, (b) any Girsanov / importance-sampling variance reduction, or (c) a damped leverage fixed-point with convergence analysis. The one finding that should change how the paper is written is in Bayer et al., who tried global basis-function regression first and abandoned it for a stated technical reason. That is a warning about the method, not prior art on it.

---

## 1. Li (2023), MSc thesis, Imperial College
`papers/pdf/li_2023_thesis_simulations_calibrated_lsv.pdf`

FULL LIST OF CONDITIONAL-EXPECTATION ESTIMATORS COVERED. Exactly three, one per chapter, compared head-to-head in Ch. 5:

1. Regularising kernel / Nadaraya-Watson (Ch. 2, pp. 11-19), from Guyon-Henry-Labordere's book. Gaussian kernel, bandwidth h = S0*N^(-1/5), plus two efficiency devices: evaluate the leverage only on a grid G_t of size max(30*sqrt(t), 15) with cubic interpolation elsewhere, and truncate the kernel sums to particles where the kernel exceeds eta = 1e-3.
2. RKHS ridge regression (Ch. 3, pp. 20-27), a reimplementation of Bayer et al. Solves (K^T K + N*lambda*R) beta_hat = K^T V on L = 40 basis points placed at equally-spaced quantiles of the particle spots.
3. Bin Monte Carlo (Ch. 4, pp. 28-33), from Van der Stoep-Grzelak-Oosterlee (2014). Partitions spot into l bins; both equal-bin-width and equal-bin-frequency conventions described, equal-frequency used in the experiments.

Ch. 6 applies method 1 to an FX model with two shifted-CIR short rates. That is the complete methodological content.

(a) PARTIAL, and only as RKHS. The projection identity is stated explicitly, twice. Page 24: "recall that the conditional expectation E[v_t|S_t = .] can be interpreted as a function that gives the best estimate of the value v_t given the information of S_t: E[v_t|S_t=.] = argmin_{f in L2} E(v_t - f(S_t))^2". Repeated in the Ch. 5 summary (p. 39) as eq. (5.3.1), then immediately replaced by the penalised RKHS problem. So the L2-projection framing exists, but exclusively as a route into RKHS ridge regression. No unpenalised least squares, no basis-function/LSMC estimator, no neural network.

(b) ABSENT. Zero occurrences of "importance", "Girsanov", "change of measure", "variance reduction", or "control variate" in the body (the only control-variate string is the title of Cozma et al. in the bibliography).

(c) ABSENT. No fixed-point iteration on the leverage function at all: all three algorithms are single-pass forward Euler-Maruyama with the leverage recomputed from the current cloud each step.

Useful as evidence: Fig. 5.2 (p. 39) compares accuracy vs N in {500,...,10000} over 50 repetitions in a Heston market, with RKHS degrading worst as N shrinks; RKHS struggles "when the underlying LSVM has a strong correlation between the share price and volatility processes in a market with non-constant volatility surface", a difficulty "not encountered by the other two methods". Practical recommendation: bin Monte Carlo.

---

## 2. Muguruza (2019), "Not so Particular about Calibration"
`papers/pdf/muguruza_2019_not_so_particular_calibration.pdf`

WHAT THE ESTIMATOR ACTUALLY IS: conditional Monte Carlo (Rao-Blackwellisation), not regression. Theorem 4.1 (p. 4): under piecewise-constant leverage, log S_{t_i} conditioned on F^W_{t_{i-1}} ∨ F^Z_{t_i} is exactly Gaussian; Corollary 4.1 replaces the Dirac by that known conditional density (eq. 5) — the "kernel" is the model's own conditional lognormal density with particle-specific model-determined width. No bandwidth, no O(h^2) bias. Prop. 4.1 gives a further closed form for lognormal SV. No learned or regression variant mentioned.

(a) ABSENT except as the baseline being replaced (Sec. 3.1-3.2 criticise Nadaraya-Watson's bias/variance and bandwidth dependence).

(b) ABSENT, but wording needs care: the abstract claims the method "theoretically guarantees a variance reduction without additional computational complexity" — this is CONDITIONING, not a measure change. Position against it rather than treating the variance-reduction space as empty. A second conditioning trick (eqs. 10-11, Appendix A) replaces GHL's Malliavin representation.

(c) ABSENT and structurally inapplicable (single forward predictor-corrector sweep). Stability remark (p. 8): "for large values of vol of vol the algorithm fails to converge ... the domain of convergence of our algorithm is superior to that of the particle method ... The precise theoretical identification of an upper bound in volatility of volatility remains unanswered."

---

## 3. Bayer, Belomestny, Butkovsky, Schoenmakers (2022/2024), RKHS
`papers/pdf/bayer_etal_2022_rkhs_singular_lsv_mckean_vlasov.pdf`

THE ONE THAT MATTERS MOST; THE FINDING IS A CAUTION.

(a) GLOBAL REGRESSION: CONSIDERED, THEN EXPLICITLY ABANDONED. Page 5, exact quote:
"In fact, the starting point of this work was to replace (1.9) by global regression based on, say, L basis functions. However, it turns out that Lipschitz constants of the resulting approximation to the conditional expectations in terms of the particle distribution explode as L -> infinity, unless the basis functions are carefully chosen."
And p. 7 on standard ridge regression on a fixed basis: it "would have to impose restrictions on the regression coefficients leading to a nonconvex constrained optimization problem."

Pages 4-5 articulate the local-vs-global argument (Nadaraya-Watson "cannot take advantage of 'global' information"; as epsilon -> 0 "the dynamics silently collapses to a pure local volatility dynamics"). The MOTIVATION for a neural estimator is fully articulated here; what is missing is any nonlinear/parametric function class realising it.

NEURAL NETWORKS NEVER MENTIONED: zero occurrences of "neural", "network", "deep learning".

Their key statements:
- Regularisation: m^lambda_A = argmin_{f in H}{E(A(Y)-f(X))^2 + lambda||f||_H^2} (eqs. 1.10-1.11, Prop. 3.3); Assumption 2.1 requires twice continuously differentiable kernel with uniformly bounded mixed derivatives.
- Theorem 2.4 (load-bearing): |m^lambda_A(x;mu) - m^lambda_A(y;nu)| <= C1 W1(mu,nu) + C2|x-y| with C1 ~ D_k/lambda^2, i.e. lambda^{-2} blow-up.
- Theorem 2.2: unique strong solution of the regularised MV system at each FIXED lambda > 0.
- Theorem 2.3 (propagation of chaos): rate eps_N = N^{-1/2} (d=1), N^{-1/2}log N (d=2), N^{-1/d} (d>2).
- Theorem 3.5/Cor 3.6 (lambda->0): convergence WITHOUT a rate; rate O(lambda^theta) only under a source condition.
- Remark 2.5 (the opening): well-posedness holds only at fixed lambda > 0; "as lambda -> 0, the Lipschitz constants of m^lambda_A blow up"; weak existence in the limit "remains however an important open problem". Conclusion (p. 26): "the choice of RKHS and the number of basis functions ... is left for future research."

(b), (c): ENTIRELY ABSENT (single-pass Euler scheme, no variance reduction of any kind).

---

## 4. Cozma, Mariapragassam, Reisinger (2017/2021), control variate particle method
`papers/pdf/cozma_mariapragassam_reisinger_2017_control_variate_particle.pdf`

(b) ABSENT. Variance reduction is purely additive control variates, NO measure change (zero occurrences of importance/Girsanov/Radon-Nikodym/reweight/likelihood ratio/tilt in 45 pages). For the conditional expectation (Sec. 3.2): coupled 2-factor LSV particle system sharing Brownian increments plus a finite-element Kolmogorov forward solve as baseline; optimal lambda estimated empirically; reduction factor 1/(1-Corr^2). Reported: 625-fold speed-up; ~4,000 particles with variates matches 2,500,000 without. Gaussian kernel with Silverman-type bandwidth h_N(T) = 1.5 S_0 sigma_LV(S_0,T) sqrt(max(T,0.25)) N^{-1/5}. Closest thing to the variance-reduction goal but correlated-baseline subtraction, not reweighting — so (b) genuinely open across all four.

(c) "Dampened" appears twice meaning a PDE change of variables (p = z^{-beta} phi) for boundary stability near V=0 under Feller violation — unrelated to relaxed fixed-point updates. There IS a real fixed-point iteration but on the PURE LV surface (Appendix E.3, Algorithm 4): undamped multiplicative update sigma_LV <- sigma_LV * Sigma_Target/Sigma_Model, attributed to Reghai (Risk 2006) and an unpublished GDF Suez note (Tur 2014). P. 43: "It is stated but not proved in [38] that the map ... is contracting ... ASSUMING THIS TO BE TRUE ... In practice, convergence is achieved particularly fast (between 10 and 20 iterations)." No relaxation parameter, no proof. The LSV leverage calibration itself (Algorithm 1) is a single forward sweep — no leverage fixed point to damp. Sec. 3.1: existence/uniqueness of the MV SDE "not established theoretically"; propagation of chaos "not proven for the present case".

(a) ABSENT (trust-region least squares fits CIR++/Heston parameters, unrelated; "quadratic basis functions" means P2 finite elements).

---

## What this means for the novelty claim

(a) clear in these four. The L2-projection identity is written down in two of them (Li p. 24, Bayer p. 6) and the local-vs-global argument made forcefully in Bayer pp. 4-5, but the only function classes instantiated are RKHS-with-ridge, bins, and Muguruza's exact conditional density.

THE THING THAT SHOULD SHAPE THE PAPER: Bayer et al. tried global basis-function regression first and dropped it because the Wasserstein-Lipschitz constant of nu -> m(.;nu) blows up as the basis grows — precisely what their well-posedness and propagation-of-chaos proofs need. A neural network is a global approximator with a far larger effective basis and no lambda||f||_H^2 control, so the obvious referee objection to a plain-L2 neural estimator is already in print, by the people closest to this problem. Either inherit a regularisation controlling the same constant (weight decay does not obviously buy a Wasserstein-Lipschitz bound), or frame the contribution as numerical with the theory explicitly open — and say so before a referee does. Their lambda->0 gap (Remark 2.5) and "choice of RKHS/basis ... future research" are the two adjacent open problems claimable.

(b) clear: Muguruza reduces variance by conditioning, Cozma et al. by control variates; neither touches the sampling measure.

(c) clear, with a soft target: only leverage-adjacent fixed point (Cozma App. E.3, pure-LV) is undamped, rests on an explicitly unproved contraction assumption, traced to an unpublished internal industry note.

SCOPE CAVEAT: absences in these four documents, not the literature. See web sweep for the broader check.
