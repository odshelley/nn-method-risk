# Prior-art web sweep: damped leverage iteration and Girsanov/IS (2026-08-31)

## (1) Damped / under-relaxed fixed-point iteration for L — NO PUBLISHED ANALYSIS

Nobody has published an analysis of the leverage iteration Phi: L -> sigma_Dupire/sqrt(E^L[V|X=.]) — no linearisation, no spectral radius, no contraction constant, no damped/under-relaxed variant specific to it. Contraction is ASSERTED, never proved. Adjacent work by relevance:

### The one damping/acceleration precedent
Nastasi, Pallavicini & Sartorelli (2017, rev. 2020), "Smile Modelling in Commodity Markets", arXiv:1808.09685. Only paper accelerating a leverage-type calibration fixed point: ANDERSON ACCELERATION (citing Anderson 1965; Walker & Ni 2011), empirical convergence curves on ICE Coffee and CME data (Figs. 2-4), 0.1bp in vol space in 15-30 iterations. Caveats: their fixed point updates a local-vol function against model implied vols (Reghai-style level-and-skew correction), NOT the conditional-expectation map; zero convergence theory.

### The one rigorous convergence theorem (different operator)
Acciaio, Marini & Pammer (2023, rev. 2025), "Calibration of the Bass Local Volatility model", arXiv:2311.14567, SIAM J. Fin. Math. For the Conze-Henry-Labordere fixed-point equation of Bass LV: existence, uniqueness up to translation, LINEAR CONVERGENCE at rate q in (0,1) in W_infinity (Thm 1.3/1.4, Prop 3.12), via contraction in L^infinity except along constants. Different operator (no conditional expectation of variance, no SV component) so it does not transfer — but the template for what an analysis of Phi would look like, and the obvious thing to position against.

### The practitioner iteration, undamped, unanalysed
Wyns & in 't Hout (2016), arXiv:1609.00232: exactly the iteration, undamped, as inner loop in each ADI step (Sec. 7), Q=2 fixed, structure credited to Tataru & Fisher (2010). No analysis; validation only that calibration error <= temporal discretisation error. Tail note: they regularise the NW quotient as E_{n,i} = (sum psi^2(v_j)P_{n,i,j} + eps*psi^2(eta))/(sum P_{n,i,j} + eps), eps=1e-8, pulling towards the mean-reversion level where the density is thin; negative numerator/denominator freezes E_{n,i} = E_{n-1,i}.
Wyns & Du Toit (2016), arXiv:1611.02961: same undamped inner iteration; all convergence analysis is of the FV spatial discretisation.

### The "stated but not proved" quote
Cozma, Mariapragassam & Reisinger, arXiv:1701.06001, Appendix E.3: Picard forward induction with multiplicative update; "It is stated but not proved in [38] that the map Phi is contracting ... Assuming this to be true..." ([38] = Reghai, "The hybrid most likely path", Risk 2006). Converges in 10-20 iterations in practice.
FALSE FRIEND WARNING: the same paper's "dampened Kolmogorov forward PDE" (Algorithm 2, Sec. 4) is a damping change of variable on the DENSITY to stabilise the PDE solve near V=0 — not damping of the iteration. Pre-empt referee confusion.

### Documented failure mode in the wings
Saporito, Yang & Zubelli (2019), arXiv:1711.03023: replaces iteration with Tikhonov; their analysis is of the regularisation. Published evidence of where the standard scheme breaks: Ren-Madan-Qian-style benchmark "is not stable and it fails to converge for large logmoneyness"; relative residuals 27.82% (benchmark) vs 7.93% (theirs) on real SPX over log-moneyness [-3,3]; explicitly not a boundary artefact.

### Fixed point for well-posedness, not numerics
Abergel & Tachet (2010), arXiv:0911.3664: contraction argument in a Banach algebra for short-time existence of the calibrated PIDE — function-space well-posedness, not the numerical iteration.

### Known adjacent strands (no iteration analysis)
Well-posedness of the calibrated MV SDE: Jourdain-Zhou; Lacker-Shkolnikov-Zhang arXiv:1905.06213; Reisinger-Tsianni arXiv:2302.00434, arXiv:2504.14343; Mustapha arXiv:2406.14074. Optimal transport sidesteps the fixed point via a convex HJB dual: Guo-Loeper-Obloj-Wang, Math. Finance 32(1) 2022. Recent survey arXiv:2512.19821 mentions neither iteration convergence nor damping.

## (2) Girsanov / importance sampling in particle LSV calibration — CLEAN ABSENCE

No paper applies IS or a Girsanov change of measure inside particle-method LSV calibration. Nothing tilts the spot measure leaving variance paths unchanged; nothing does importance-weighted estimation of E[V|X]. All variance reduction in this space is control variates or Rao-Blackwellisation.

- Cozma et al. arXiv:1701.06001: control variates (calibrated pure-LV + 2-factor Heston LSV), applied separately to numerator and denominator of the conditional-expectation estimator; thin-density regions patched by smooth extrapolation downstream, not sampled better.
- Cuchiero-Khosrawi-Teichmann arXiv:2005.02505: hedging / deep-hedging control variates, not IS.
- Muguruza arXiv:1909.13366 — the closest thing to "weighted particles" and a distinction to draw head-on: Theorem 4.1 conditions on the volatility filtration (Romano-Touzi mixing) so every particle contributes an analytic Gaussian weight at EVERY strike, replacing the kernel. This is conditional Monte Carlo / Rao-Blackwellisation, NOT a measure change: sampling measure untouched, no Radon-Nikodym weights, no attempt to move particles into the tails (though wings improve incidentally). Remark 4.1 claims guaranteed variance reduction.

### Structural prior art to CITE, from a different application
dos Reis, Smith & Tankov, "Importance sampling for McKean-Vlasov SDEs", Appl. Math. Comput. 453 (2023), arXiv:1803.09320. Two IS schemes for MV-SDEs with interacting particles: (i) complete measure change (tilt in coefficients and expectation simultaneously); (ii) DECOUPLING: estimate the law under the original measure, freeze the law component, then simulate the resulting ordinary SDE under the IS measure. The decoupling algorithm is precisely the structure of "change measure for the spot while leaving the mean-field component alone". Optimal tilt via large deviations + Pontryagin. Decoupling costs 2-3x plain MC; up to 3 orders of magnitude variance reduction (Kuramoto model). NO finance, NO LSV, NO calibration anywhere (verified on full text). Follow-ups (all rare-event, never calibration): arXiv:2207.06926, arXiv:2208.03225, arXiv:2307.05149.

### Caveats
Primary sources unverifiable (paywalled/403): GHL "Smile Calibration Problem Solved" (SSRN 1885032), Henry-Labordere SSRN 1493306, the Nonlinear Option Pricing book, Ren-Madan-Qian (Risk 2007), Tataru-Fisher (Bloomberg internal). Every follow-up describes GHL's limitations as kernel/bandwidth/variance and attributes no IS device to them — treat "not in GHL" as very likely but unverified. The paywalled practitioner sources are the most plausible home for undocumented damping heuristics.
False friend: Scharth & Kohn, "Particle efficient importance sampling", J. Econometrics 190(1) 2016 — IS for likelihood estimation in SV models with leverage EFFECT (spot-vol correlation); unrelated.

## Positioning takeaways
(1) A spectral/linearisation analysis of Phi with a damping prescription would be new: contraction asserted not proved (Reghai via Cozma), iteration run undamped with fixed small counts (Wyns & in 't Hout Q=2), one theory-free acceleration attempt (Anderson, Nastasi et al.), one rigorous linear-rate theorem on a different operator (Acciaio et al.).
(2) The tail failure mode is repeatedly named but every fix is downstream regularisation or a control variate; nobody changes the sampling measure. The decoupling idea exists in the general MV-SDE literature (dos Reis-Smith-Tankov) — a citation to make, not a threat, since they never touch calibration.

Extracted full texts for grepping: /Users/osianshelley/.claude/jobs/84f2fa54/tmp/{1701.06001,1909.13366,1609.00232,1611.02961,2311.14567,1803.09320}.txt, at.txt, comm.txt, szz.txt, surv.txt.
