# Baseline comparison set

One card per method. Every method runs through the same harness (`bench/`): same scenario, same particle budgets N in {1e3, 1e4, 1e5}, same time grid, same seeds, same CIR full-truncation stepper; only the conditional-expectation estimator swaps out via `make_estimator(name)`. Ground truth for the acceptance tests is the Heston market scenario family (semi-analytic Heston pricer through the grid Dupire class); the SSVI grid stays for the paper figures.

Metrics per method, identical rows: surface repricing error (IV bps) at fixed budget, wing error, robustness across the (vol-of-vol, mixing) grid, runtime, and the knob-sensitivity curve (h for kernels, lambda for ridge/RKHS, architecture for the network). Experiment design mirrors Li (2023), adding our estimators.

Status of each card: owner, registry name, implementation state.

---

## Card 1: Kernel Nadaraya-Watson (Guyon and Henry-Labordere 2012)

**Owner:** neural-particle-code session. **Registry:** `nw` (Gaussian/Silverman, existing) plus a `nw-ghl` quartic variant to add.

**Source and equations.** Guyon and Henry-Labordere, "Being particular about calibration", Risk, Jan 2012 (SSRN 1885032). Estimator: Nadaraya-Watson ratio
E[V | S = K] ~ sum_i V_i K_h(S_i - K) / sum_i K_h(S_i - K).

**Kernel and bandwidth, with a source discrepancy to resolve.** Muguruza (2019, section 3.1) reports that the original article recommends a *quartic* kernel with a rule-of-thumb bandwidth, and follows those guidelines for his benchmark implementation. Cozma, Mariapragassam and Reisinger (SIAM J. Fin. Math. 2019, arXiv:1701.06001) implement a *Gaussian* kernel with the Silverman-type rule they attribute to GHL:

    h_N(T) = eta * S0 * sigma_LV(S0, T) * sqrt(max(T, T_min)) * N^(-1/5),  eta = 1.5, T_min = 0.25.

Li (2023) uses h = S0 * N^(-1/5) with truncation eta = 1e-3 and grid |G_t| = max(30 sqrt(t), 15). Reisinger and Tsianni (2023) note the AMISE-optimal c * N^(-1/5) rate but find h = S0 N^(-1/5) / 100 more accurate in their regularised setting. TODO: quote GHL's own kernel and constants directly; the Risk paper and the CRC book (Nonlinear Option Pricing, ch. 11) are both paywalled, so this awaits Osian pulling either through institutional access. Until then the card cites Muguruza's and Cozma's readings, which disagree on the kernel family.

**Tuning knob and protocol.** Bandwidth h. Primary run at the Cozma rule; sensitivity curve over h in {rule/10, rule/3, rule, 3*rule, 10*rule}.

**Documented failure modes to probe.** O(h^2) bias and bandwidth-sensitive variance (Muguruza 2019 section 3.2); drastic accuracy loss under a plain Silverman rule even at 1e5 to 2e6 particles (Bain, Mariapragassam, Reisinger 2019, appendix A.1 discussion); loss of convergence at large vol-of-vol (the Fig. 3 cross).

**Acceptance criterion.** Reproduce Li (2023) Table 2.4 (kernel, Heston market: kappa 1.5768, theta 0.0484, xi 0.5751, rho -0.7, v0 0.1024; N = 1e5, M = 1000) at bandwidth h0: average absolute IV error 1.44% (simple LSVM) and 1.18% (complex LSVM), within the run-to-run band; his h0/3 and h0/10 rows (0.71 to 0.75%) pin the bandwidth-sensitivity curve.

## Card 2: Exact conditional-Gaussian method (Muguruza 2019)

**Owner:** neural-particle-code session. **Registry:** `muguruza` (needs the per-step vol-path hook in `calibrate/explicit.py`).

**Source and equations.** Muguruza, "Not so Particular about Calibration: Smile Problem Resolved" (SSRN 3461545, arXiv:1909.13366). Corollary 4.1 only (per Osian's decision): with d_i(K) = (mu_i - log K) / sqrt((1 - rhohat^2) sigma_i^2),

    E[V_ti D | S_ti = K] / E[D | S_ti = K]
      = E[ D V_ti exp(-d_i(K)^2 / 2) / sqrt(sigma_i^2) ] / E[ D exp(-d_i(K)^2 / 2) / sqrt(sigma_i^2) ],

where mu_i and sigma_i^2 are the per-step conditional mean and variance of log S given the vol path over [t_{i-1}, t_i] with leverage frozen at S_{t_{i-1}} (his Theorem 4.1). Kernel-free and bandwidth-free by construction.

**Tuning knob and protocol.** None (that is the point). The sensitivity axis degenerates; report the absence.

**Structural caveat to probe.** The formula needs the conditional-Gaussian step structure: log-spot Gaussian given the vol path, leverage frozen within the step. Exact for the Heston test case as the step size shrinks; probe step-size sensitivity (M sweep) where the frozen-leverage approximation is the only error source.

**Acceptance criterion.** Reproduce the variance-reduction claim of his section 7 qualitatively: match the kernel benchmark's calibrated smile at equal N with visibly lower estimator variance across repetition bands.

## Card 3: RKHS ridge (Bayer, Belomestny, Butkovsky, Schoenmakers)

**Owner:** this session. **Registry:** `rkhs` (`estimators/rkhs.py`).

**Source and equations.** arXiv:2203.01160 (Finance and Stochastics 2024). Regularised conditional expectation m_lambda_A = (C_nu + lambda I)^(-1) c_nu_A, their (1.10). Practical algorithm (their (4.3) and (5.9)): choose L centres c_j at the j*100/(L+1) percentiles of the particle cloud, kernel matrix K_ij = k(X_i, c_j), Gram R_jl = k(c_j, c_l), solve

    (K^T K + N * lambda * R) beta = K^T G,   mhat(x) = sum_j beta_j k(x, c_j),

with G_i = V_i. Their numerical section: Gaussian kernel with variance 0.1, L = 100, lambda = 1e-9, M = 500 steps, N = 1e6, Heston kappa 2.19, theta 0.17023, xi 1.04, rho -0.83, S0 = 1, v0 = 0.0045, CIR variance floored at 1e-3. Li (2023, Algorithm 2) runs the same construction with L = 40, Gaussian pdf kernel with variance 5 (pdf normalisation rescales the effective lambda), lambda = 1e-5 (BS market) and 1e-7 (Heston market).

**Why this card is mandatory.** Bayer et al. explicitly abandoned global regression on L fixed basis functions (Lipschitz constants explode as L grows, their section 1) and flagged fixed-basis ridge as leading to nonconvex constrained problems. Our frozen-body ridge head is fixed-basis ridge with a learned basis; this head-to-head is the comparison a referee demands first.

**Tuning knob and protocol.** lambda (primary), kernel variance and L (secondary, held at source values). Sensitivity curve over lambda in {1e-9 ... 1e-3} decades.

**Documented failure modes to probe.** Li (2023): the simple LSVM with rho = -0.5 fails for lambda >= 1e-6, and no lambda in the tested range works at rho = -0.7 with N = 1e5, M = 1000. Probe exactly that cross.

**Acceptance criterion.** Reproduce Bayer et al. Fig. 1 (calibrated smile at N in {1e3, 1e4, 1e5}) on their Heston parameters within the repetition band, and Li's failure at rho = -0.7.

## Card 4: Equal-frequency bins (van der Stoep et al. via Li 2023)

**Owner:** this session. **Registry:** `bins` (`estimators/bins.py`).

**Source and equations.** Binning per van der Stoep, Grzelak, Oosterlee (2014); parameters and results as run by Li (2023). Partition the cloud into l equal-frequency buckets by sorted lnx; the estimate on each bucket is the (weighted) mean of V within it; grid points map to buckets by the empirical bucket edges (piecewise-constant estimator).

**Tuning knob and protocol.** Bucket count l in {20, 50, 200} (Li's grid). Primary run l = 20.

**Documented failure modes to probe.** Piecewise-constant bias in the wings where buckets are wide; degradation as l grows at fixed N (variance) and as l shrinks (bias); no smoothness for the leverage function without post-hoc interpolation.

**Acceptance criterion.** Li (2023) Table 4.2, Heston market, N = 1e5, l = 20: average absolute IV error 0.91% (simple LSVM) and 1.01% (complex LSVM); match within the repetition band.

## Card 5: PURBF (Hakala 2019)

**Owner:** neural-particle-code session (offer stands to move it here; the source PDF is now in papers/pdf and staged for the knowledge graph).

**Source and equations.** Hakala, "Applied Machine Learning for Stochastic Local Volatility Calibration", Frontiers in Artificial Intelligence 2:4, 2019. Partition-of-unity RBF with C << N centres:

    PURBF(x) = sum_j w_j K_hj(x - c_j) / sum_j K_hj(x - c_j),

weights by ridge-regularised normal equations w = (A^T A - lambda id)^(-1) A^T y with A_ij = K_hj(x_i - c_j) (his (3); the sign on lambda is as printed, implement as +lambda). Centres: min(x_i), max(x_i), plus a random subset of the cloud, pruned when |c_i - c_j| / h_j falls below a global pruning constant; per-centre local widths; his best variant sets the width from the 5 nearest neighbours. Rule-of-thumb global width h = (4 sigma^5 / (3 n))^(1/5).

**Positioning note (use in section 8).** Hakala rules out MLPs because "the training is much more involved in the MLP case", and writes "We could envision to use a pretrained MLP to get the solution without training. We postpone this approach for potential future use." The warm-started frozen-body head is exactly that postponed approach; the PURBF comparison closes his loop.

**His best configuration (section 4, verified against the typeset PDF).** C = 40 units ("sufficiently versatile for the number of particles we want to use (2,048)"), lambda = 0.2, pruning on, local widths from the 5 nearest neighbours (a 3-NN variant also appears in Fig. 7), vol-of-variance mixing 66%. The pruning constant is the symbol Theta in min_i(|c_i - c_j| / h_j) <= Theta and its value is never stated; treat it as a free knob. The printed solution w = (A^T A - lambda id)^(-1) A^T y has a minus sign inconsistent with his ridge loss LSR = (1/2N) sum (y_i - RBF(x_i))^2 + lambda sum w_j^2; implement the standard +lambda.

**Tuning knob and protocol.** Number of centres C, nearest-neighbour count for local widths, pruning constant Theta; primary run at his best configuration above.

**Documented failure modes to probe.** Centre placement sensitivity in thin wings; pruning-constant sensitivity.

**Acceptance criterion.** Necessarily qualitative: the paper contains no numeric error tables; all comparisons are figures on proprietary Leonteq FX snapshots (EUR/USD 6M/5Y, USD/JPY 5Y, EUR/BRL 3Y). Criterion: at his configuration on our Heston market with N = 2,048, PURBF with 5-NN widths visibly dominates plain and local-linear kernel regression in the wings without oscillation in the bulk (his Figs. 6, 7, 9 pattern); freeze our first accepted run as the numeric regression target thereafter.

## Card 6: PDE / Fokker-Planck reference (anchor, not competitor)

**Owner:** separate "PDE-method" session, own branch. Low-dimensional Heston-LSV forward Fokker-Planck solve, used as the accuracy anchor for the explicit-scheme comparisons. Not wired into the estimator registry.

## Card 7: Control variates (excluded this round)

Cozma, Mariapragassam, Reisinger (arXiv:1701.06001). Composable overlay rather than rival estimator (their headline: 2e4 particles with control variates matching 2.5e6 without). Related-work mention only, per Osian's decision.

---

## Treatment arms (ours, for reference)

`nn` (per-slice network, warm-started), `ridge` (frozen-body ridge head, shrink-to-previous), `spline` (strongest classical head in 1D), implicit-scheme `GlobalRidge`. Cards not needed; they are the treatment, the cards above are the controls.
