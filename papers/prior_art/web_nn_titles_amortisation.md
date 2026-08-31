# Prior-art web sweep: NN regression, title collisions, warm-starting (2026-08-31)

## (1) NN least-squares regression of E[V|S] inside particle LSV calibration

CLEAN ABSENCE. No paper, preprint, thesis, or public code trains a neural network with a plain L2 loss on the particle cloud to estimate E[V_t | X_t = x] inside the Guyon-Henry-Labordere loop. Every neural LSV paper either parametrises the leverage function directly against option prices, solves a PDE/density problem, or amortises the whole calibration map. The regression slot itself is still held by kernels, bins, and RKHS.

Two findings matter most:

### Hakala (2019), "Applied Machine Learning for Stochastic Local Volatility Calibration", Frontiers in AI 2, art. 4
https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2019.00004/full

Same problem, framed identically ("the task is to find a function R(S) = E(V_t|S_t = S) based on the samples as observed pairs"). Benchmarks Nadaraya-Watson, local linear kernel regression, RBF, and partition-of-unity RBF; concludes PURBF with regularisation and pruning wins. Considers neural networks and rules them out, verbatim:

> "In the last couple of years popularity of multilayer perceptron (MLP) and deep versions thereof grew enormously. For our application we rule out these architectures as the training is much more involved in the MLP case with a many remaining questions about a suitable number of hidden units, number of layers, type of activation functions. We could envision to use a pretrained MLP to get the solution without training. We postpone this approach for potential future use."

The idea has been named and deferred in print, never executed. Confirms the gap is real; also hands a reviewer a "considered and rejected in 2019" objection. The paper needs to beat PURBF, not just Nadaraya-Watson, and answer the training-cost objection directly.

### van der Stoep, Grzelak & Oosterlee (2014), "The Heston stochastic-local volatility model: efficient Monte Carlo simulation", IJTAF 17(7)
https://ir.cwi.nl/pub/22747/22747D.pdf

Remark 3.2 is the linear-basis ancestor: estimate E[psi^2(V)|S] by L2 projection onto orthogonal polynomials — exactly least-squares regression with a linear function class. Rejected on positivity grounds:

> "Although intuitive and straightforward, the regression-based alternative possesses the drawback that the Feller condition must be satisfied to guarantee a positive conditional expectation for the whole range of arguments... Since in practice the Feller condition is often violated, regression-based methods require additional tuning like including high-order polynomials or constraining of regression coefficients. As such model improvements need to be done on case-by-case basis we consider the nonparametric approach as preferable"

They use binning instead. Sharpest technical warning: an unconstrained L2 regressor of a conditional variance can go negative exactly where the true conditional expectation approaches zero, and that divides into the leverage. Present the positivity parametrisation (softplus / log-learning) explicitly as answering this objection, with citation.

### Rest of the neural LSV landscape (no collision)
- Cuchiero, Khosrawi & Teichmann (2020, arXiv:2005.02505): networks parametrise the leverage directly, trained on prices with deep-hedging variance reduction; call E[alpha_t^2|S_t=s] "a very challenging and still open problem"; concede no speed competition with the particle method. Most likely paper for a reviewer to confuse with ours — distinguish early (calibration objective vs regression estimator inside the fixed point).
- Hoshisashi, Phelan & Barucca (ICAIF 2025, doi:10.1145/3768292.3770350; methodology arXiv:2604.13723): PINN on Fokker-Planck with density normalisation and Dupire consistency; PDE route, no particles, no regression.
- Wang, Despres, Dureau & Buet-Golfouse (arXiv:2608.01217): neural operator amortising quotes -> calibration triple. Caution: its related-work sentence loosely lumps Cuchiero et al. and Hakala together; do not inherit the conflation.
- Brunner (github.com/ejbrun/NN_Calibration_Stochastic_Local_Volatility): price-map learning, not conditional expectation.
- Muguruza (arXiv:1909.13366): closed-form exact alternative, no NN.

Non-neural estimator baselines for comparison: GHL Nadaraya-Watson; Bayer et al. arXiv:2203.01160 (kernel ridge; presented by Bayer as "A kernel regression approach to local stochastic volatility models", Milstein 2025); Reisinger & Tsianni arXiv:2302.00434, arXiv:2504.14343; recent weak-error survey arXiv:2506.10817 mentions no neural network.

Adjacent literatures checked and clean: deep backward BSDE schemes (Hure-Pham-Warin and descendants, e.g. arXiv:2603.14721) and deep Longstaff-Schwartz use NN L2 conditional-expectation regression as standard, but nobody ports it to the LSV leverage loop nor proposes it as future work there. Neural McKean-Vlasov solvers do something else: arXiv:2501.00780 (trajectory networks matched via Ito calculus, no finance), DeepSPoC arXiv:2408.16403 (fits empirical distribution, not conditional expectation), arXiv:2512.14967 (NN conditional expectations in MV-FBSDEs, never LSV calibration).

## (2) "Neural Particle Method" title collision — HIGH RISK

Exact-name uses:
- Wessels, Weissenfels & Wriggers, CMAME 368 (2020) 113127, arXiv:2003.10208 — owns the acronym NPM; code gitlab.com/henningwessels/npm.
- "An Improved Neural Particle Method for Complex Free Surface Flow Simulation Using PINNs", Mathematics 11(8):1805 (2023) — "INPM".
- Shibukawa, Ozaki & Berthet, "The compressible Neural Particle Method...", arXiv:2508.16916 (Aug 2025), physics.flu-dyn.
- Kim, Son, Kim & Lee, "A Physics-Informed, Global-in-Time Neural Particle Method for the Spatially Homogeneous Landau Equation", arXiv:2603.10874 (Mar 2026), math.NA — most damaging: recent, in numerical analysis, neural particle discretisation of a nonlinear kinetic PDE; a numerics-literate reader will assume a connection.

Close variants established elsewhere: Neural Particle Filter (arXiv:1508.06818), Neural Particle Smoothing (NAACL 2018, arXiv:1804.10747), Neural Particle Image Velocimetry (arXiv:2101.11950), Robust Neural Particle Identification (arXiv:2212.07274), Neural Particle Automata (arXiv:2601.16096), physics-informed neural particle flow filtering (arXiv:2606.10959, arXiv:2602.23089). "Deep particle" space also occupied: DeepParticle (arXiv:2111.01356 — learns invariant measures from an interacting particle method via Wasserstein minimisation, uncomfortably close conceptually), DeepSPoC, NeuralMPM (arXiv:2408.15753), Deep JKO (arXiv:2311.06700), ParticleWNN (arXiv:2305.12433).

MATHEMATICAL FINANCE: CLEAN. No use of "neural particle method" or close variants in quant finance (arXiv title search all categories, journals, software). Recommendation: prefer a name encoding the finance-specific object — "neural leverage calibration" and "neural Markovian projection" were both checked and are unused as titles.

## (3) Warm-starting / amortisation

Little beyond what's known, plus one citable precedent:
- Hakala 2019 contains the seed of the amortisation idea ("pretrained MLP to get the solution without training ... postpone[d] for potential future use"), never followed up.
- Cuchiero-Khosrawi-Teichmann calibrate maturity by maturity with earlier parameters frozen and feeding forward, but use a separate network per interval; no statement about warm-starting weights across maturities found.
- Cozma-Mariapragassam-Reisinger reuse a PDE solve from a simpler submodel as a control variate — reuse of computation, not of learning.
- Sequential slice-by-slice bootstrapping is structural in all implementations; nobody frames it as amortisation.

CLEAN ABSENCE: nothing warm-starts a learned conditional-expectation estimator across time slices within a run, and nothing reuses fitted regressors across calibration runs, other than arXiv:2608.01217 amortising across quote surfaces. Both open.
