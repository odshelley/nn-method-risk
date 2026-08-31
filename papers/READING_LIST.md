# LSV Calibration: Annotated Reading List

Compiled 2026-08-31 from a four-way parallel web search (particle method, PDE/Fokker–Planck methods, machine-learning methods, surveys and foundations). Roughly 45 unique resources; each section was researched and annotated independently, so a handful of load-bearing papers (Gyöngy 1986, Abergel–Tachet 2010, Jourdain–Zhou, Lacker–Shkolnikov–Zhang, Djete, Mustapha, Bayer et al. RKHS, Saporito–Yang–Zubelli, Reisinger–Tsianni) appear in more than one section, annotated from that section's angle.

## How the four sections fit together

The calibration condition is the same everywhere: by Gyöngy's mimicking theorem, an LSV model reprices all vanillas iff the leverage satisfies L²(t,S)·E[V_t | S_t=S] = σ²_Dupire(t,S). Every method is a way of computing the conditional expectation E[V_t | S_t=S], which depends on the law of the solution itself and makes the problem a McKean–Vlasov fixed point:

- **Particle method** (Section 1): estimate it by kernel regression over an interacting particle system, in one forward Monte Carlo pass. Scales to hybrids and many factors; pays in Monte Carlo noise and kernel/bandwidth choices.
- **PDE / Fokker–Planck** (Section 2): read it off the gridded joint density from the 2D forward Kolmogorov equation, with an inner fixed-point iteration per time step. Deterministic and accurate in 2D; hits the curse of dimensionality beyond three factors, and the near-zero-density wings and the v=0 boundary (Feller violation) need care.
- **Machine learning** (Section 3): either parameterise the leverage directly with networks and bypass the conditional expectation (Cuchiero–Khosrawi–Teichmann GAN), learn the whole quotes→leverage map offline as a neural operator, enforce Fokker–Planck as a PINN residual, or replace the model class entirely (neural SDEs, signatures). Semimartingale optimal transport (Guo–Loeper–Wang) is the principled non-neural optimisation baseline.
- **Surveys and foundations** (Section 4): Dupire, Gyöngy, the original LSV mixing papers, the Homescu survey, freely available Imperial theses, and path-dependent volatility as the current competitor to LSV.

A recurring theme worth knowing before reading anything: well-posedness of the calibrated McKean–Vlasov equation is still open in full generality. Short-time existence (Abergel–Tachet 2010), global existence for finite-state volatility (Jourdain–Zhou), stationary solutions (Lacker–Shkolnikov–Zhang), existence under non-regular coefficients plus propagation of chaos (Djete 2022), and strong existence-uniqueness with finite-state volatility plus propagation of chaos (Mustapha 2024) are the milestones; continuous-state Heston-LSV remains unresolved without regularisation.

## Suggested entry points

- Method itself: Guyon & Henry-Labordère, Risk 2012, then *Nonlinear Option Pricing* Ch. 11.
- PDE route: Ren–Madan–Qian 2007 (free PDF linked in Section 2/4), then Wyns–Du Toit for modern FV-ADI numerics.
- ML route: Cuchiero–Khosrawi–Teichmann 2020, then the 2026 neural-operator paper for the current cost frontier.
- Theory: Lacker–Shkolnikov–Zhang 2019 for the "inverting the Markovian projection" framing, then Mustapha 2024.

PDFs of the freely available papers are in `pdf/` alongside this file.

---

# SECTION 1 — Particle Method


15 entries in four groups: (1) foundational particle-method papers, (2) well-posedness / propagation-of-chaos theory for the calibrated McKean-Vlasov SDE, (3) numerical analysis of the particle scheme, (4) refinements, extensions, implementation. Links marked "verified" were fetched directly from the arXiv abstract page.

---

## 1. Foundations

### The Smile Calibration Problem Solved
**Julien Guyon, Pierre Henry-Labordere - 2011 (SSRN working paper, July 2011)**
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1885032

The original statement of the particle method for LSV calibration. Starting from Gyongy's mimicking theorem, the calibration condition forces the leverage function to equal the Dupire local volatility divided by the root conditional mean square of the stochastic volatility factor given the spot. Because that conditional expectation is taken under the law of the solution itself, the resulting SDE is nonlinear in the sense of McKean, and the authors solve it by simulating an interacting particle system in which the conditional expectation is replaced by a kernel-weighted empirical average across particles. Also covers hybrid models and gives a Malliavin representation of the effective local volatility. Cite this for the method itself; the authors' own keywords are "nonlinear SDEs, particle method, calibration, Malliavin calculus."

### Being Particular About Calibration
**Julien Guyon, Pierre Henry-Labordere - 2012, Risk Magazine 25(1), p. 88**
https://www.risk.net/markets/2135540/being-particular-about-calibration

The practitioner-facing Risk version of the SSRN paper above. Same content, more compressed and aimed at desk implementation: it shows how to calibrate exactly any multi-factor LSV model to market smiles in a single Monte Carlo pass, with no PDE solve and no iterative outer loop. In practice this is the canonical citation in the quant literature, and "the particle method" almost always means this pair of papers. Worth having both: the Risk article as the standard reference, the SSRN version for the derivations.

### Nonlinear Option Pricing
**Julien Guyon, Pierre Henry-Labordere - 2013, Chapman & Hall/CRC Financial Mathematics Series (ISBN 9781466570337; 2024 reprint 9781032919393)**
https://www.routledge.com/9781466570337

The book-length treatment and the best self-contained exposition. Part III covers McKean nonlinear SDEs and the particle method in full: calibration of LSV models to vanillas with and without stochastic interest rates, the Markovian projection machinery underneath, and the practical questions the papers skip (kernel choice, bandwidth, particle count, time-stepping, and handling the low-density wings where the kernel estimate degrades). If you want one reference explaining why the method works rather than just how to run it, this is it.

---

## 2. Well-posedness of the calibrated McKean-Vlasov equation

### A Nonlinear Partial Integro-Differential Equation from Mathematical Finance
**Frederic Abergel, Remi Tachet - 2010, Discrete and Continuous Dynamical Systems A 27(3), 907-917**
https://www.aimsciences.org/article/doi/10.3934/dcds.2010.27.907 (preprint: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1508490)

The earliest rigorous result on the LSV calibration equation, working on the PDE side rather than the SDE side. The Fokker-Planck equation for the calibrated model is a nonlinear integro-differential PDE whose nonlocality comes from a *quotient of two integral terms*, an operator not even defined on all bounded continuous functions. Using a fixed-point argument plus a priori estimates they prove short-time existence. This short-time-only result is the benchmark everything else in this section tries to improve on, and it is why the field described the global problem as open for so long.

### Existence of a Calibrated Regime Switching Local Volatility Model and New Fake Brownian Motions
**Benjamin Jourdain, Alexandre Zhou - 2016/2017 (arXiv), published Mathematical Finance 30(2), 2020** *(verified)*
https://arxiv.org/abs/1607.00077 - journal: https://onlinelibrary.wiley.com/doi/abs/10.1111/mafi.12231

The first global existence result for a genuinely calibrated LSV model. The trick is to discretise the volatility factor: the stochastic volatility driver is a jump process with *finitely many* values (regime switching), which makes the associated Fokker-Planck system tractable, and they extend an argument of Figalli to get existence. A by-product is a new family of "fake Brownian motions" (martingales with Brownian marginals but non-Brownian dynamics), a nice illustration of how underdetermined the calibration constraint is. The finite-state restriction propagates into most later work, including Mustapha below.

### Inverting the Markovian Projection, with an Application to Local Stochastic Volatility Models
**Daniel Lacker, Mykhaylo Shkolnikov, Jiacheng Zhang - 2019 (arXiv), Annals of Probability 48(5), 2020** *(verified)*
https://arxiv.org/abs/1905.06213 - journal: https://projecteuclid.org/journals/annals-of-probability/volume-48/issue-5/Inverting-the-Markovian-projection-with-an-application-to-local-stochastic/10.1214/19-AOP1420.full

The deepest theoretical treatment, and the one that names the problem correctly: LSV calibration is the *inversion* of Gyongy's Markovian projection. They study two-dimensional McKean-Vlasov SDEs in which the conditional law of the second component given the first enters the equation for the first, prove strong existence of *stationary* solutions, and prove strong uniqueness in an important special case. The stationarity requirement is the significant caveat for finance applications, but the techniques (fixed point on the space of conditional laws, plus regularity of the projected coefficient) set the template for the field.

### Non-regular McKean-Vlasov Equations and Calibration Problem in Local Stochastic Volatility Models
**Mao Fabrice Djete - 2022 (arXiv v1 Aug 2022), revised October 2024** *(verified)*
https://arxiv.org/abs/2208.09986

Attacks the problem at the generality it actually needs: McKean-Vlasov equations whose drift and volatility coefficients are *not* continuous in the Wasserstein topology on the measure argument, which is precisely the failure mode of the calibration coefficient (a ratio involving a conditional expectation). Establishes existence under these weak conditions and shows the N-particle system approximates the limit via propagation of chaos, then specialises to deduce existence of calibrated LSV models for suitable parameter choices. The most direct theoretical justification available for the particle method as actually implemented.

### Strong Existence and Uniqueness of a Calibrated Local Stochastic Volatility Model
**Scander Mustapha - 2024 (arXiv, June 2024)** *(verified)*
https://arxiv.org/abs/2406.14074

Current state of the art on the SDE side. Proves *strong* existence and uniqueness for the two-dimensional McKean-Vlasov SDE whose volatility depends on the conditional distribution of the volatility factor given the log-price, establishing well-posedness of a two-factor LSV model calibrated to European call prices. Follows Jourdain-Zhou in assuming the volatility driver takes finitely many values. Crucially, it also establishes propagation of chaos for the particle system, the theoretical justification for the Guyon-Henry-Labordere algorithm that had been missing for over a decade.

---

## 3. Numerical analysis of the particle scheme

### A Reproducing Kernel Hilbert Space Approach to Singular Local Stochastic Volatility McKean-Vlasov Models
**Christian Bayer, Denis Belomestny, Oleg Butkovsky, John Schoenmakers - 2022 (arXiv Mar 2022), revised January 2024; Finance and Stochastics** *(verified)*
https://arxiv.org/abs/2203.01160

Directly addresses the regularization question the original particle method leaves informal. Instead of an ad hoc kernel-density estimate of the conditional expectation, they regularize via a reproducing kernel Hilbert space projection, prove the regularized McKean-Vlasov model is well-posed, and prove propagation of chaos for the corresponding particle system. Numerically the regularized model reproduces local-volatility option prices essentially exactly. Read this if you care about *which* regularization of the conditional expectation to use and what it costs in bias.

### Convergence of the Euler-Maruyama Particle Scheme for a Regularised McKean-Vlasov Equation Arising from the Calibration of Local-Stochastic Volatility Models
**Christoph Reisinger, Maria Olympia Tsianni - 2023 (arXiv Feb 2023, revised Aug 2023)**
https://arxiv.org/abs/2302.00434

The first full discretisation error analysis of the method as implemented. Working with regularised coefficients (well-posedness of the unregularised problem was open at the time), they prove the regularised model is well-posed and that Euler-Maruyama converges strongly to the particle system with rate 1/2 in the step size, with the error's dependence on the regularisation parameters made explicit. That explicit dependence is the practically useful part: it tells you how bandwidth and time step trade off, confirmed empirically on a Heston-type LSV calibration.

### Numerical Analysis of a Particle System for the Calibrated Heston-type Local Stochastic Volatility Model
**Christoph Reisinger, Maria Olympia Tsianni - 2025 (arXiv, April 2025)**
https://arxiv.org/abs/2504.14343

The follow-up handling the model people actually use, Heston-type LSV, where the square-root variance process brings only 1/2-Holder regularity in the diffusion coefficient on top of mean-field terms in both drift and diffusion. They establish well-posedness for a fixed but arbitrarily small kernel bandwidth and prove *strong* propagation of chaos, giving convergence of the particle system under a condition on the Feller ratio and up to a critical time. Numerics use log-Euler for the spot and full-truncation Euler for the CIR variance. The Feller-ratio condition and the critical time are the honest caveats: convergence is not unconditional.

---

## 4. Refinements, extensions, and implementation

### Calibration of a Hybrid Local-Stochastic Volatility Stochastic Rates Model with a Control Variate Particle Method
**Andrei Cozma, Matthieu Mariapragassam, Christoph Reisinger - 2017 (arXiv), SIAM Journal on Financial Mathematics 10(1), 2019**
https://arxiv.org/abs/1701.06001 - journal: https://doi.org/10.1137/17M1114570

The main variance-reduction paper and the standard reference for four-factor FX LSV with stochastic rates. Builds directly on Guyon-Henry-Labordere and layers on control variates from three cheaper calibrated models: a pure local volatility model, a two-factor Heston-type LSV model (both with deterministic rates), and the CIR short rates. Variance reduction matters here because the kernel estimator's noise is what limits particle counts in production. Calibrated on real EUR/USD data with runtime comparable to a PDE calibration of the two-factor LSV model alone, which is the benchmark practitioners care about. Not tied to a specific diffusion choice.

### Not So Particular About Calibration: Smile Problem Resolved
**Aitor Muguruza - 2019 (arXiv Sept 2019; also SSRN 3461545)** *(verified)*
https://arxiv.org/abs/1909.13366 - SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3461545

The most pointed critique of the original method's weak spot. The particle method's accuracy depends on a kernel function and a bandwidth, and getting these wrong produces instability, especially in the wings and at short maturities. Muguruza proposes a Monte Carlo LSV calibration that is closed-form and exact in the conditional-expectation step, removing dependence on both kernel and bandwidth, with a guaranteed variance reduction at no extra computational cost. Applies to all stochastic volatility models including the non-Markovian rough volatility family, which the standard particle method handles awkwardly. Tested on several hybrid (rough) LSV models.

### Calibration of Local-Stochastic and Path-Dependent Volatility Models to Vanilla and No-Touch Options
**Alan Bain, Matthieu Mariapragassam, Christoph Reisinger - 2019 (arXiv, November 2019)**
https://arxiv.org/abs/1911.00877

Extends particle calibration beyond vanillas to barrier-type instruments, where LSV models earn their keep in FX. The key device is a *two-states* particle method estimating the Markovian projection of the variance onto the pair (spot, running maximum), combined with a forward PIDE (from Hambly et al. 2016) that prices up-and-out calls across all strikes, barriers and maturities in one solve. Worked through for a Heston-type LSV model with local vol-of-vol and for two path-dependent volatility models. On EUR/USD data all three calibrate inside the market no-touch bid-ask, a demanding test.

### Path-Dependent Volatility
**Julien Guyon - 2014 (SSRN, September 2014); Risk Magazine**
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2425048

Applies the particle method outside the LSV setting proper. PDV models make instantaneous volatility a function of the path of past returns; like local volatility models they are complete and fit the market smile exactly, and like stochastic volatility models they generate rich implied-vol dynamics. The exact smile calibration is done with the particle method, since the calibration condition again involves a conditional expectation given a non-Markovian state. Relevant as evidence that the particle machinery is a general tool for calibrating any model whose calibration constraint is a Markovian-projection identity, not an LSV-specific hack.

### Cross-Dependent Volatility
**Julien Guyon - 2015 (SSRN, June 2015); Risk Magazine, March 2016**
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2615162 - Risk: https://www.risk.net/derivatives/2451464/cross-dependent-volatility

The multi-asset extension. In CDV models each asset's volatility depends on the current or past levels of the *other* assets, so calibrating to index/basket smiles requires conditional expectations across assets. All calibration procedures here use the particle method, and calibrating the implied "local in basket" CDV needs a novel *fixed-point* particle method, i.e. an outer fixed-point iteration wrapped around the particle simulation. The economic punchline is separately interesting: steep basket skews can be produced by cross-dependency of volatilities alone, without any correlation skew.

### A New Calibration of the Heston Stochastic Local Volatility Model and Its Parallel Implementation on GPUs
**Ana Maria Ferreiro-Ferreiro, Jose A. Garcia-Rodriguez, Luis Souto, Carlos Vazquez - 2020, Mathematics and Computers in Simulation 177, 467-486**
https://doi.org/10.1016/j.matcom.2020.03.019

The GPU/implementation reference, though note it is a deliberate alternative to the particle method rather than an acceleration of it. Instead of the usual two-step procedure (calibrate the stochastic volatility parameters, then bootstrap the leverage function), it calibrates the whole parameter set plus the leverage function jointly, using multi-CPU parallel global optimisation driving multi-GPU Monte Carlo pricers. Because it never relies on analytical pricing formulas it can calibrate directly against exotics. Useful as a performance and design contrast: it buys flexibility and joint calibration at the cost of the particle method's single-pass exactness on vanillas.

---

## Reading path

For the method itself: the Risk 2012 article, then Part III of *Nonlinear Option Pricing*. For the theory, the honest chronology is Abergel-Tachet (short time only), then Jourdain-Zhou (global, finite-state), then Lacker-Shkolnikov-Zhang (stationary solutions, the "inverting the Markovian projection" framing), then Djete and Mustapha (2022-2024, the closest thing to a justification of the algorithm as run). For implementation: Reisinger-Tsianni 2023/2025 for discretisation error and the Feller-ratio caveat, Cozma-Mariapragassam-Reisinger for variance reduction, and Muguruza before committing to a kernel and bandwidth at all.

## Two gaps worth flagging

No result yet gives global strong well-posedness for a *continuous-state* volatility factor without regularisation. Every entry in section 2 restricts to finitely many volatility states, to stationary solutions, or to a regularised coefficient.

The propagation-of-chaos rates in section 3 are established only up to a critical time under a Feller condition, so they do not cover long-dated calibration in low-Feller Heston regimes.

---

# SECTION 2 — PDE / Fokker–Planck Methods


Ordered from theoretical foundation, through the practitioner line of work, to numerics and well-posedness theory. All arXiv/Springer/AIMS/SSRN/Wiley links verified to resolve; caveats on the paywalled and grey-literature items noted at the end.

---

## 1. Mimicking the one-dimensional marginal distributions of processes having an Itô differential

**Author:** István Gyöngy
**Year:** 1986
**Venue:** Probability Theory and Related Fields 71(4), 501–516
**Link:** https://link.springer.com/article/10.1007/BF00699039

The theoretical foundation of the entire LSV calibration condition. Gyöngy proves that for an Itô process `dξ_t = β_t dt + δ_t dW_t` there exists a Markov diffusion with deterministic coefficients `σ²(t,x) = E[δ_t δ_tᵀ | ξ_t = x]` matching the one-dimensional marginals for every t. Applied to an LSV model `dS_t = L(t,S_t)√V_t S_t dW_t`, this gives the calibration identity `L²(t,S) · E[V_t | S_t = S] = σ²_Dup(t,S)`: the leverage function must equal Dupire local volatility divided by the root conditional expectation of the stochastic variance. Every PDE method below is a numerical scheme for computing that conditional expectation from a joint density.

---

## 2. The vol smile problem

**Author:** Alexander Lipton
**Year:** 2002
**Venue:** Risk Magazine, February 2002, 61–65
**Link:** https://www.risk.net/derivatives/equity-derivatives/1530435/vol-smile-problem

The origin of the "universal volatility model" (UVM): a single framework nesting local volatility, stochastic volatility and jumps, with an LSV specification whose local volatility component is calibrated so the model reproduces the vanilla surface exactly. Lipton sets out the forward-PDE route — solve the 2D forward Kolmogorov equation for the joint density `p(t,S,V)`, then extract `E[V_t|S_t=S]` by numerical quadrature in the V direction at each spot level. This is the canonical citation for one-factor forward-PDE calibration of LSV models, as distinct from the later particle/Monte-Carlo route.

---

## 3. Calibrating and pricing with embedded local volatility models

**Authors:** Yong Ren, Dilip Madan, Michael Qian Qian
**Year:** 2007
**Venue:** Risk Magazine, September 2007, 138–143
**Link:** https://www.risk.net/derivatives/equity-derivatives/1500232/calibrating-and-pricing-embedded-local-volatility-models

The paper that turned the Gyöngy identity into a workable production algorithm, and the near-universal reference for the fixed-point / iterative 2D forward Kolmogorov PDE method. On a time-stepping grid they solve the 2D forward Kolmogorov (Fokker–Planck) equation for the joint density of spot and variance, and at each step compute the leverage from the just-computed density via `L(t,S) = σ_Dup(t,S) / sqrt( ∫v p(t,S,v)dv / ∫p(t,S,v)dv )`, then advance the density with that leverage. Because L at step n+1 depends on the density at step n+1, the step is implicit and is resolved by an inner fixed-point iteration (two or three passes usually suffice for small Δt). They also give the first quanto corrections in local volatility models. This is what practitioners mean by "the Ren–Madan–Qian scheme".

---

## 4. Stochastic Local Volatility (Bloomberg white paper)

**Authors:** Grigore Tataru, Travis Fisher
**Year:** 2010 (Version 1, 5 February 2010)
**Venue:** Bloomberg L.P., Quantitative Development Group technical report
**Link:** https://www.scribd.com/document/405131295/Bloomberg-Stochastic-Local-Volatility — no official Bloomberg-hosted public URL exists; this mirror is the commonly reachable copy. Citable secondary exposition of the same methodology: Yong Kuen Zhu, *Implementation of Local Stochastic Volatility Model in FX Derivatives*, in Wystup (ed.), Springer 2017 — https://link.springer.com/chapter/10.1007/978-3-662-54486-0_4

The reference practitioner note for FX LSV. It specifies the model as `dS/S = ... + L(t,S)√V_t dW` with a Heston or lognormal variance factor and a *mixing fraction* η interpolating between pure local volatility (η=0) and pure stochastic volatility, and details the forward-PDE bootstrap for L. Its distinctive practical contribution is calibrating the term structure of η from normalised risk-reversal moves — that is, from smile *dynamics*, since vanillas are already matched by construction through L. De facto industry standard and the basis of several commercial SLV calibrators; worth citing despite being grey literature.

---

## 5. Pricing of vanilla and first generation exotic options in the local stochastic volatility framework: survey and new results

**Authors:** Alexander Lipton, Andrey Gal, Andris Lasis
**Year:** 2013 preprint; Quantitative Finance 14(11), 2014
**Link:** https://arxiv.org/abs/1312.5693

Lipton's own survey of the LSV framework a decade after the UVM paper, and the best single entry point to that line of work. Directly relevant to numerics: the authors are explicit that standard ADI schemes struggle on the LSV forward and backward equations (mixed derivative terms, the degenerate variance boundary when the Feller condition fails, non-smooth leverage coefficients) and propose a Galerkin–Ritz alternative. They also develop semi-analytical benchmarks, including a closed-form series expansion in powers of the spot–variance correlation, which are valuable for validating a PDE calibration engine since exact LSV reference prices are otherwise unavailable.

---

## 6. Calibration of the Heston stochastic local volatility model: A finite volume scheme

**Authors:** Bernd Engelmann, Frank Koster, Daniel Oeltz
**Year:** 2011 working paper; published 2021
**Venue:** International Journal of Financial Engineering 8(1)
**Link:** https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1823769 (journal: https://www.worldscientific.com/doi/10.1142/S2424786320500486)

The first careful numerical-analysis treatment of the Heston-LSV calibration PDE. Their central point is that standard finite differencing of the 2D forward Kolmogorov equation misbehaves in exactly the two places that matter: the mixed second derivative from spot–variance correlation, and the region near v=0 when the Feller condition `2κθ ≥ ξ²` is violated, where the transition density is singular. They propose a finite volume scheme after a suitable transformation of the model equation, mass-conserving by construction, and validate on real market data. Important because loss of mass in the density biases the conditional expectation and hence the calibrated leverage directly.

---

## 7. A Finite Volume – Alternating Direction Implicit Approach for the Calibration of Stochastic Local Volatility Models

**Authors:** Maarten Wyns, Jacques Du Toit
**Year:** 2016 preprint; International Journal of Computer Mathematics, 2017
**Link:** https://arxiv.org/abs/1611.02961 (NAG technical report version: https://support.nag.com/doc/techrep/pdf/tr3_16.pdf)

The most complete modern reference for the finite-difference/finite-volume + ADI machinery behind forward-PDE LSV calibration. They give a finite volume discretisation of general 1D and 2D forward Kolmogorov equations that, unlike finite differences, requires no transformation of the PDE — decisive here because the leverage function in the coefficients is typically non-smooth, being itself the output of the previous calibration step. Time stepping uses the Hundsdorfer–Verwer ADI scheme (stable in the presence of the mixed derivative), with an inner iteration each step to resolve the nonlinearity from the implicit dependence of L on the current density. Mass conservation is proved rather than hoped for.

---

## 8. An adjoint method for the exact calibration of Stochastic Local Volatility models

**Authors:** Maarten Wyns, Karel J. in 't Hout
**Year:** 2016 preprint; Journal of Computational Science, 2017
**Link:** https://arxiv.org/abs/1609.00232

A conceptual sharpening of the previous entry, best read alongside it. The observation is that the usual approach discretises the *continuous* calibration condition and therefore calibrates only approximately: after discretisation the SLV model does not reprice vanillas exactly. Instead they impose calibration at the *semidiscrete* level, using an adjoint semidiscretisation of the forward Kolmogorov equation to derive the leverage function making semidiscrete SLV European values match semidiscrete LV values exactly. This yields a large nonlinear ODE system, solved with ADI time stepping plus inner iterations. The "discretise first, then calibrate the discrete system" pattern is the same idea that makes adjoint approaches attractive elsewhere in model calibration.

---

## 9. The calibration of stochastic local-volatility models: An inverse problem perspective

**Authors:** Yuri F. Saporito, Xu Yang, Jorge P. Zubelli
**Year:** 2017 preprint; published 2019
**Venue:** Computers & Mathematics with Applications 77(12), 3054–3067
**Link:** https://arxiv.org/abs/1711.03023

Reframes leverage calibration as an ill-posed inverse problem rather than a forward recursion, and applies Tikhonov regularisation with a stability analysis. The motivating pathology is real: `E[V_t|S_t=S]` is a ratio of integrals of the joint density, so in regions of low probability density (deep spot wings, extreme variance states) the denominator is near zero and the naive quotient explodes numerically, producing spurious spikes in the leverage surface that then feed back into the next time step. Their regularised formulation is stable there, respects the integrity of the raw quoted data, and avoids the implied-surface interpolation the Dupire-then-divide route requires. The natural reference if you want calibration posed as a well-defined optimisation with a regularisation parameter rather than as a fixed-point iteration.

---

## 10. A nonlinear partial integro-differential equation from mathematical finance

**Authors:** Frédéric Abergel, Rémi Tachet des Combes
**Year:** 2010
**Venue:** Discrete and Continuous Dynamical Systems – Series A, 27(3), 907–917
**Link:** https://www.aimsciences.org/article/doi/10.3934/dcds.2010.27.907 (preprint: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1508490)

The first well-posedness result for the calibrated LSV Fokker–Planck equation, and the paper everyone cites when noting the problem is not known to be globally well posed. Substituting the Gyöngy leverage into the forward equation gives a nonlinear, *nonlocal* PIDE whose nonlinearity is a quotient of two integrals of the unknown density in the variance direction; that quotient is not even defined for all bounded continuous functions, so standard parabolic theory does not apply. Via a fixed-point argument with a priori Schauder-type estimates they prove existence of a classical solution on a bounded domain in R² with Dirichlet conditions — but only for short time and under a smallness condition on the second derivative of the volatility factor function. Global existence remained open. This is why the practitioner fixed-point iteration can and does break down at large mixing fractions or long maturities.

---

## 11. Existence of a calibrated regime switching local volatility model and new fake Brownian motions

**Authors:** Benjamin Jourdain, Alexandre Zhou
**Year:** 2016 preprint; published 2020
**Venue:** Mathematical Finance 30(2), 501–546
**Link:** https://arxiv.org/abs/1607.00077

The first *global-in-time* well-posedness result for a calibrated LSV model, obtained by restricting the volatility factor to finitely many values. When the stochastic volatility factor is a constant-in-time random variable with finite support, and the range of its square is not too large, they prove existence for the associated Fokker–Planck equation; they then extend to regime-switching local volatility, where the factor is a jump process taking finitely many values with spot-dependent jump intensities. The smallness condition on the spread of the variance states is the mathematical counterpart of the practitioner observation that calibration destabilises as vol-of-vol or mixing fraction is pushed up. The "fake Brownian motion" construction is a by-product of independent probabilistic interest.

---

## 12. Inverting the Markovian projection, with an application to local stochastic volatility models

**Authors:** Daniel Lacker, Mykhaylo Shkolnikov, Jiacheng Zhang
**Year:** 2019
**Venue:** Annals of Probability 48(5), 2189–2211
**Link:** https://arxiv.org/abs/1905.06213

The state-of-the-art probabilistic formulation of the calibration problem. The calibration condition is precisely a request to *invert* the Markovian projection (Gyöngy's map), which turns the LSV model into a 2D McKean–Vlasov SDE where the conditional law of the variance component given the spot component enters the spot dynamics. They prove strong existence of stationary solutions and strong uniqueness in an important special case. For a PDE-oriented reader the value is that it clarifies what object the nonlinear Fokker–Planck equation is the forward equation *of*, and why the nonlocality is intrinsic rather than an artefact of the formulation.

---

## 13. Non-regular McKean–Vlasov equations and calibration problem in local stochastic volatility models

**Author:** Mao Fabrice Djete
**Year:** 2022 (revised 2024)
**Link:** https://arxiv.org/abs/2208.09986

Pushes the existence theory to McKean–Vlasov equations whose coefficients are *not* continuous in the measure variable for the Wasserstein topology — exactly the regularity failure caused by the conditional-expectation quotient in the LSV calibration condition. Djete obtains existence results, an N-particle approximation and propagation of chaos, and thereby constructs calibrated LSV models. Also the theoretical justification for the particle-method calibration route, the main competitor to the forward-PDE route.

---

## 14. Strong existence and uniqueness of a calibrated local stochastic volatility model

**Author:** Scander Mustapha
**Year:** 2024
**Link:** https://arxiv.org/abs/2406.14074

The most recent well-posedness result in this line. For a 2D McKean–Vlasov SDE whose volatility depends on the conditional distribution of one component given the other, with the volatility-driving factor taking finitely many values (following Jourdain–Zhou), Mustapha establishes strong existence *and* uniqueness, confirming well-posedness of a two-factor LSV model calibrated to European call prices. He also proves propagation of chaos for the associated particle system, giving theoretical backing for the Guyon–Henry-Labordère algorithm. Useful for a literature review because it marks how far the theory has actually got: strong well-posedness only under a finite-support restriction on the volatility factor, so genuinely continuous-state Heston-LSV calibration remains theoretically open.

---

## 15. Calibration of Local Volatility Model with Stochastic Interest Rates by Efficient Numerical PDE Method

**Authors:** Julien Hok, Shih-Hau Tan
**Year:** 2018 preprint; Decisions in Economics and Finance, 2019
**Link:** https://arxiv.org/abs/1803.03941

The hybrid-model extension of the forward-PDE approach, relevant for long-dated equity/FX where rates volatility is no longer negligible. They solve a forward PDE for the discounted probability density across state variables using an ADI scheme (an extension of Peaceman–Rachford), and give an efficient grid-based algorithm for all corrective terms that stochastic rates introduce into the local volatility function. Read alongside Deelstra & Rayée, *Local Volatility Pricing Models for Long-Dated FX Derivatives* (https://arxiv.org/abs/1204.0633, Applied Mathematical Finance 20(4), 2013), which derives the corresponding local volatility identities via Gyöngy's mimicking property under stochastic domestic and foreign rates.

---

## Note on the competing (non-PDE) route

For contrast rather than as part of the PDE line: the main alternative is the particle method of Guyon & Henry-Labordère, which estimates `E[V_t|S_t=S]` by kernel regression over an interacting particle system instead of from a gridded joint density. Its four-factor hybrid extension with variance reduction is Cozma, Mariapragassam & Reisinger, *Calibration of a Hybrid Local-Stochastic Volatility Stochastic Rates Model with a Control Variate Particle Method* (https://arxiv.org/abs/1701.06001, SIAM J. Financial Math. 10(1), 181–213, 2019). The practical trade-off: PDE methods are deterministic, fast and accurate in two dimensions but hit the curse of dimensionality beyond three factors, whereas the particle method scales to hybrids at the cost of Monte Carlo noise in the leverage estimate.

---

## Link-quality caveat

All arXiv, Springer, AIMS, SSRN and Wiley links above were verified to resolve. The two Risk Magazine articles (Lipton 2002; Ren–Madan–Qian 2007) are paywalled on risk.net with no legitimate open version, so those links reach abstract/paywall pages only. The Tataru–Fisher Bloomberg note has never had an official public URL; the Scribd mirror listed is the reachable copy, and the Springer chapter by Zhu is the better citable substitute if you need a stable reference for that methodology.

---

# SECTION 3 — Machine Learning Methods


Fifteen entries, grouped into five clusters: (1) the direct neural-leverage-function line, (2) neural SDE / market-model approaches, (3) operator-learning and PINN approaches to the LSV fixed point, (4) kernel and signature approaches, (5) surrogate-based deep calibration adjacent to LSV pipelines. All arXiv links verified to resolve.

---

## 1. A generative adversarial network approach to calibration of local stochastic volatility models

**Authors:** Christa Cuchiero, Wahid Khosrawi, Josef Teichmann
**Year:** 2020 (v1 May 2020, v3 Sep 2020) · **Venue:** *Risks* 8(4), 101
**Link:** https://arxiv.org/abs/2005.02505 · Code: https://github.com/wahido/neural_locVol

The canonical ML-for-LSV paper. The leverage function is parameterised directly by a family of feed-forward neural networks (one per time interval) and trained on market option prices, which sidesteps both the ad hoc interpolation of the implied volatility surface and the McKean–Vlasov conditional expectation entirely — there is no particle estimate of E[V|S=s] anywhere in the loop. The framing is a neural SDE trained adversarially: the generator produces volatility surfaces via the LSV SDE, and the discriminator quantifies distance to market prices. The practical enabler is a variance reduction technique based on hedging and deep hedging, which makes model prices and implied vols accurate enough to differentiate through using only small path batches; demonstrated on a SABR-type LSV model with stability analysis across many smile samples.

## 2. Robust pricing and hedging via neural SDEs

**Authors:** Patryk Gierjatowicz, Marc Sabate-Vidales, David Šiška, Lukasz Szpruch, Žan Žurič
**Year:** 2020 (arXiv) / 2022 · **Venue:** *Journal of Computational Finance* 26(3)
**Link:** https://arxiv.org/abs/2007.04154 · Code: https://github.com/msabvid/robust_nsde

Rather than fixing a parametric form for drift and diffusion, both are overparameterised neural networks, giving a model class rich enough to calibrate consistently under both the risk-neutral and real-world measures. The connection to causal optimal transport is what makes this more than a fitting exercise: it lets them compute robust *bounds* on exotic prices over the set of models consistent with observed vanillas, together with the corresponding hedges. For LSV work this is the reference for how much exotic-price uncertainty survives after perfect vanilla calibration, which is precisely the question the leverage function alone cannot answer.

## 3. Arbitrage-free neural-SDE market models

**Authors:** Samuel N. Cohen, Christoph Reisinger, Sheng Wang
**Year:** 2021 (arXiv) / 2023 · **Venue:** *Applied Mathematical Finance* 30(1)
**Link:** https://arxiv.org/abs/2105.11053

Models the joint dynamics of the whole liquid vanilla book nonparametrically, with neural networks for the drift and diffusion of a low-dimensional factor SDE. Two constraints are imposed structurally rather than penalised: a derived HJM-type drift condition rules out dynamic arbitrage, and because static arbitrage constraints are linear inequalities in prices, the admissible factor state space is a convex polytope the network is kept inside. Notably for this survey, they validate against data generated from a Heston stochastic local volatility model — so it doubles as a market-model alternative to calibrating an LSV leverage function when the object of interest is the surface's dynamics rather than its level. A companion paper applies it to option-book risk (https://arxiv.org/abs/2202.07148).

## 4. Amortizing the Calibration Triple: A Projection-Consistent Neural Operator for Local-Stochastic Volatility

**Authors:** Xiaozhen Wang, Anaïs Després, Martin Dureau, Francois Buet-Golfouse
**Year:** 2026 (submitted 2 Aug 2026) · **Venue:** arXiv preprint
**Link:** https://arxiv.org/abs/2608.01217

The most recent and most directly targeted at the computational bottleneck. A neural operator (both DeepONet and FNO variants) is trained to map finite quote sets to the whole calibration triple at once: the arbitrage-constrained implied volatility surface, its Dupire local variance, and the LSV leverage together with the conditional moment satisfying the projection identity. The point is that the slow, noisy, sequential McKean–Vlasov fixed point is amortised offline, so online calibration is a single operator evaluation: latency drops from 98.5 ms to 0.6 ms, with local-volatility error improved 36% and leverage error 7–16% against baselines, and exotic pricing errors close to a particle implementation. Caveat: very recent, so treat the benchmark numbers as unreplicated.

## 5. Probability-Density-Consistent Physics-Informed Neural Networks for Stochastic Local Volatility Model Calibration

**Authors:** Kentaro Hoshisashi, Carolyn E. Phelan
**Year:** 2025 · **Venue:** ICAIF 2025 (6th ACM International Conference on AI in Finance), Singapore, p. 456
**Link:** https://dl.acm.org/doi/10.1145/3768292.3770350 (DOI resolves; ACM blocks automated fetch)

A PINN formulation in which the network is made consistent with the joint density rather than only with prices, so the Fokker–Planck side of the LSV fixed point is enforced as a physics residual instead of being solved by an outer PDE iteration. Benchmarked against the classical approach of iteratively solving Fokker–Planck and updating the leverage function, with robustness tested on EUR/USD FX option quotes over 127 business days in Jan–Jun 2025. This is the closest thing in the literature to a like-for-like comparison of a neural method against the standard production PDE pipeline on real FX data.

## 6. A Reproducing Kernel Hilbert Space approach to singular local stochastic volatility McKean-Vlasov models

**Authors:** Christian Bayer, Denis Belomestny, Oleg Butkovsky, John Schoenmakers
**Year:** 2022 (v1) / 2024 (v2) · **Venue:** arXiv preprint (math.PR / q-fin.CP)
**Link:** https://arxiv.org/abs/2203.01160

The LSV calibration equation is a *singular* McKean–Vlasov SDE, and its well-posedness is genuinely open in general. This paper regularises the singular conditional-expectation coefficient using RKHS methods — kernel regression in place of the kernel-density particle estimate — and then proves the regularised model is well-posed and satisfies propagation of chaos, which the raw problem does not obviously do. Numerically the regularised scheme replicates option prices from typical local volatility models essentially exactly. This is the bridge entry between the particle-method literature and the ML literature: the estimator is a learned function, but the guarantees are probabilistic.

## 7. Signature-based models: theory and calibration

**Authors:** Christa Cuchiero, Guido Gazzani, Sara Svaluto-Ferro
**Year:** 2022 (arXiv) / 2023 · **Venue:** *SIAM Journal on Financial Mathematics*
**Link:** https://arxiv.org/abs/2207.13136 · Code: https://github.com/GuidoGazzani-ai/sigsde_calibration

Asset dynamics are linear functionals of the time-extended signature of a primary process (Brownian motion up to a general continuous multidimensional semimartingale). Universality means classical models — including LSV — are approximated arbitrarily well, and because the model is *linear in its parameters* given precomputed signature samples, calibration decomposes into offline sampling plus a standard, well-conditioned optimisation. That is the key contrast with neural-leverage approaches: no nonconvex network training and no differentiating through a Monte Carlo simulation. Calibrated to both time series and implied volatility surfaces, on synthetic stochastic-vol data and S&P 500 market data.

## 8. Joint calibration to SPX and VIX options with signature-based models

**Authors:** Christa Cuchiero, Guido Gazzani, Janka Möller, Sara Svaluto-Ferro
**Year:** 2023 (v1) / 2024 (v2), published 2025 · **Venue:** *Mathematical Finance* (2025)
**Link:** https://arxiv.org/abs/2301.13235 · Code: https://github.com/GuidoGazzani-ai/jointcalib_sigsde

Extends the signature framework to the joint SPX/VIX problem, the standard stress test for any smile-calibration method. The technical lever is that the truncated signature of a polynomial diffusion is itself a polynomial diffusion, which yields closed-form VIX², so both log-price and VIX² are linear in signature elements and Fourier pricing applies. They report highly accurate joint calibration with neither jumps nor rough volatility, which matters because most competing solutions to the joint problem require one or the other.

## 9. Pricing and calibration in the 4-factor path-dependent volatility model

**Authors:** Guido Gazzani, Julien Guyon
**Year:** 2024 (arXiv) / 2025 · **Venue:** *Quantitative Finance* 25(3)
**Link:** https://arxiv.org/abs/2406.02319

The 4-factor Markovian version of Guyon–Lekeufack path-dependent volatility, where instantaneous vol is a function of a weighted sum of past returns and the square root of a weighted sum of past squared returns. This is the natural rival to LSV for smile dynamics: it fits the smile from *endogenous* path dependence rather than an exogenous leverage function, which is why exotic prices behave differently from an LSV model fitted to the same vanillas. Includes a neural network approximation of the VIX as a function of the Markovian factors *and the model parameters*, replacing the nested Monte Carlo that made the original calibration intractable.

## 10. Joint deep calibration of the 4-factor PDV model

**Authors:** Fabio Baschetti, Giacomo Bormetti, Pietro Rossi
**Year:** 2025 (submitted 12 Jul 2025) · **Venue:** arXiv preprint
**Link:** https://arxiv.org/abs/2507.09412

Learns SPX implied volatilities, VIX futures, and VIX call prices simultaneously with neural networks, reducing the pricing functionals to matrix-vector products evaluated on the fly and cutting joint calibration from hours to seconds. Worth reading alongside entry 9 as the cleanest demonstration that the surrogate approach transfers from parametric stochastic-vol models to path-dependent ones — the same argument applies to LSV pipelines where the expensive object is a nested or particle simulation.

## 11. Calibration of local-stochastic volatility models by optimal transport

**Authors:** Ivan Guo, Grégoire Loeper, Shiyi Wang
**Year:** 2019 (arXiv) / 2022 · **Venue:** *Mathematical Finance* / SIAM J. Financial Math. line of work
**Link:** https://arxiv.org/abs/1906.06478

Reformulates LSV calibration as a semimartingale optimal transport problem: find the diffusion matching observed vanilla marginals while minimising a transport cost, solved via the dual HJB rather than a McKean–Vlasov fixed point. This is the main non-neural optimisation-based competitor and the right baseline for any ML method — it gives a convex-duality-based well-posed problem where the particle method gives a fixed point that may not have a solution. The survey "Optimal Transport for Model Calibration" (Guo, Loeper, Obłój, Wang, SSRN 3876854) covers the extensions to joint VIX/SPX, path-dependent options, and stochastic rates.

## 12. Deep learning volatility: a deep neural network perspective on pricing and calibration in (rough) volatility models

**Authors:** Blanka Horvath, Aitor Muguruza, Mehdi Tomas
**Year:** 2019 (arXiv) / 2021 · **Venue:** *Quantitative Finance* 21(1), 11–27
**Link:** https://arxiv.org/abs/1901.09647

The reference for the two-step surrogate paradigm: learn the *pricing map* (parameters → implied vol surface) offline, then calibrate with a standard optimiser against the fast surrogate, achieving full-surface calibration in milliseconds. Not an LSV paper, but it defines the design pattern used by entries 4, 10 and 15, and it is the paper to cite for why one should learn the pricing map rather than the inverse map (parameters as a function of market data) — the inverse map is not well-posed when the calibration problem has near-degenerate directions.

## 13. Deep Local Volatility

**Authors:** Marc Chataigner, Stéphane Crépey, Matthew Dixon
**Year:** 2020 · **Venue:** *Risks* 8(3), 82
**Link:** https://arxiv.org/abs/2007.10462 · Code: https://github.com/mChataign/DupireNN

Neural interpolation of vanilla prices that yields the full Dupire local volatility surface as a by-product, with no-arbitrage enforced either by hard architectural constraints or by soft penalties in the loss. Directly relevant to LSV because the Dupire surface is the input to the leverage-function fixed point, and a non-arbitrage-free or noisy local vol surface is a common failure source upstream of LSV calibration. Their honest finding is a trade-off: hard constraints are the only fail-safe route but numerically crush the network's representational power, while soft constraints give much better prices and implied vols at the cost of sporadic residual arbitrage.

## 14. Beyond Surrogate Modeling: Learning the Local Volatility via Shape Constraints

**Authors:** Marc Chataigner, Areski Cousin, Stéphane Crépey, Matthew Dixon, Djibril Gueye
**Year:** 2021 · **Venue:** *SIAM Journal on Financial Mathematics* 12(3), SC58–SC69
**Link:** https://arxiv.org/abs/2212.09957

The follow-up to entry 13, comparing neural network and Gaussian process regression for learning local volatility under shape constraints. The Gaussian process route is the one to note for LSV work: it delivers a *posterior* over the local volatility surface, so the calibration uncertainty inherited by the downstream leverage function can be quantified rather than assumed away.

## 15. Deep Generative Calibration on Stochastic Volatility Models with Applications in FX Barrier Options

**Authors:** Moyna Ma, Carmine Ventre, Renzo Tiranti, Aiming Chen
**Year:** 2025 · **Venue:** ACM SAC 2025 (40th ACM/SIGAPP Symposium on Applied Computing)
**Link:** https://dl.acm.org/doi/10.1145/3672608.3707874 · Also: https://kclpure.kcl.ac.uk/portal/en/publications/deep-generative-calibration-on-stochastic-volatility-models-with-

A two-step industrial-flavoured application: learn the pricing map with a neural network, then run a traditional calibration algorithm against it, taking FX barrier option calibration from hours to seconds. The generative component is used to expand the training set to cover a broader range of market conditions than historical quotes contain, which addresses the main practical weakness of surrogate calibration — surrogates degrade off the training distribution exactly when markets move. Useful as the applied/FX counterpoint to the more theoretical entries.

---

## Also worth knowing (not full entries)

- **Mao Fabrice Djete, "Non-regular McKean-Vlasov equations and the calibration problem in local stochastic volatility models"** (https://arxiv.org/abs/2208.09986, 2022/2024) — pure theory, no ML, but it is the existence result for calibrated LSV models under minimal continuity assumptions, plus propagation of chaos for the N-particle approximation. This is what any neural or particle method is implicitly assuming has a solution.
- **Hakala, "Applied Machine Learning for Stochastic Local Volatility Calibration"** (*Frontiers in AI* 2019, https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2019.00004/full) — early and comparatively lightweight: radial basis function estimators for the SLV calibration step. Historically first in the "ML for SLV" line but superseded by entries 1 and 5.
- **Deep self-consistent learning of local volatility** (Wang, Shaa, Privault, https://arxiv.org/abs/2201.07880) — approximates both market prices and local volatility with networks coupled through Dupire's PDE.

## Reading path suggestion

If the goal is to understand what ML actually buys over the particle method: read **1** for the idea (drop the conditional expectation entirely), **6** for why the underlying problem is delicate, **4** for the current state of the art on cost, and **5** for the only careful head-to-head against the classical Fokker–Planck pipeline on real quotes. Read **2** and **11** for the two principled framings — causal optimal transport and semimartingale optimal transport — that tell you what the calibration problem's non-uniqueness actually costs you in exotic prices.

---

# SECTION 4 — Surveys, Foundations, Comparisons


All links checked. Where a paywall or bot-block prevented full-text retrieval I say so and give a free alternative.

---

## 1. Pricing with a Smile — the local volatility foundation

**Author:** Bruno Dupire
**Year:** 1994
**Type:** Journal article (practitioner, *Risk*, Jan 1994, pp. 18–20)
**Link:** https://www.semanticscholar.org/paper/Pricing-with-a-Smile-Dupire/03798655e555ca39ed845e4399f745b3d0d11681 (scanned copy also at https://www.scribd.com/document/101527176/Dupire-Pricing-With-a-Smile-1994)

Dupire shows that a continuum of European option prices across strikes and maturities pins down a unique one-factor diffusion, with local variance recovered from the Dupire forward equation as a ratio of option-price derivatives. This is the object every LSV calibration ultimately targets: the leverage function is defined so the LSV model's Markovian projection reproduces exactly this Dupire surface. Read it for the forward-PDE formulation, which is also the computational backbone of the PDE calibration route.

---

## 2. Mimicking the one-dimensional marginal distributions of processes having an Itô differential

**Author:** István Gyöngy
**Year:** 1986
**Type:** Journal article (*Probability Theory and Related Fields* 71(4), 501–516)
**Link:** https://www.semanticscholar.org/paper/Mimicking-the-one-dimensional-marginal-of-processes-Gy%C3%B6ngy/d5dc0ef6fff9cfbd8717a0bf62c44de618be6708

Gyöngy constructs, for a general Itô process, a Markov diffusion with identical one-dimensional time marginals, whose coefficients are conditional expectations of the original coefficients given the state. This is the theorem that makes LSV calibration well-defined: an LSV model matches all European vanillas exactly iff its leverage function equals the Dupire local volatility divided by the root conditional mean square of the stochastic variance given spot. Every calibration scheme in the literature — particle, PDE/Fokker–Planck, optimal transport, neural — is a numerical scheme for that conditional expectation.

---

## 3. Pricing exotics under the smile — the original LSV mixing idea

**Authors:** Mark Jex, Robert Henderson, David Wang
**Year:** 1999
**Type:** Practitioner article (*Risk*, November 1999, pp. 72–75)
**Link:** https://www.smartquant.com/references/OptionPricing/option14.pdf (verified: free full text)

The paper that introduced the LSV construction as used on desks: a stochastic volatility process captures the gross features and dynamics of the smile, and a second deterministic (local) component is layered on and calibrated so the market smile is matched exactly. Short and non-technical, but the canonical citation for the mixing formulation, and it states the motivation — exotics pricing where pure local volatility gives the wrong forward smile dynamics — more crisply than most modern treatments.

---

## 4. The vol smile problem — the universal volatility model

**Author:** Alexander Lipton
**Year:** 2002
**Type:** Practitioner article (*Risk* 15(2), 61–65)
**Link:** https://www.risk.net/derivatives/equity-derivatives/1530435/vol-smile-problem (paywalled; circulated copy at https://www.scribd.com/document/57405163/Vol-Smile-Problem-Lipton-Risk-02)

Lipton surveys local volatility, jump-diffusion, stochastic volatility and their combinations for the FX smile, and proposes the "universal volatility model" — LSV with jumps — as the best-performing variant. With Jex–Henderson–Wang this is the standard pair of citations for the origin of LSV. Useful as the framing document for why the industry converged on LSV rather than on either pure component.

---

## 5. Calibrating and pricing with embedded local volatility models — the PDE/Fokker–Planck route

**Authors:** Yong Ren, Dilip Madan, Michael Qian Qian
**Year:** 2007
**Type:** Practitioner article (*Risk*, September 2007, pp. 138–143)
**Link:** https://staff.fnwi.uva.nl/a.khedher/winterschool/16RenMadanQian.pdf (verified: free full text, title page confirmed)

Ren, Madan and Qian give the first practical calibration of the leverage function by solving the two-dimensional forward Fokker–Planck PDE for the joint spot-variance density and reading the conditional expectation off it, marching forward in time. They also give quanto corrections in local volatility models. This is the reference implementation of the PDE branch and the benchmark against which the particle method is always compared (one 2-D nonlinear PDE solve, versus a particle system that scales better in dimension).

---

## 6. Nonlinear Option Pricing — Chapters 11 and 12

**Authors:** Julien Guyon, Pierre Henry-Labordère
**Year:** 2013
**Type:** Book (Chapman & Hall/CRC Financial Mathematics Series; reissued by Routledge 2024)
**Link:** https://www.routledge.com/Nonlinear-Option-Pricing/Guyon-Henry-Labordere/p/book/9781032919393

Chapter 11, "Calibration of Local Stochastic Volatility Models to Market Smiles", is the best self-contained treatment of the smile calibration problem: it sets up the McKean–Vlasov SDE that the calibrated LSV model satisfies, then develops the particle method with kernel-regression estimation of the conditional expectation, including bandwidth choice and the bias/variance tradeoff. Chapter 12 extends the same machinery to local correlation and the FX triangle calibration problem. If the researcher reads one thing on calibration methodology, this is it; the underlying journal version is Guyon & Henry-Labordère, "The Smile Calibration Problem Solved" (SSRN 1885032).

---

## 7. Local Stochastic Volatility Models: Calibration and Pricing

**Author:** Cristian Homescu
**Year:** 2014
**Type:** Survey / working paper
**Link:** https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2448098 (SSRN abstract page; SSRN returns 403 to automated fetches, so full text not pulled, but the record is live and the PDF downloads in a browser)

The most comprehensive practitioner-facing literature overview of LSV calibration available free. It covers calibration procedures and then focuses on the numerical detail of pricing via forward and backward PDEs/PIDEs derived from the LSV model — grid construction, boundary conditions, ADI splitting — exactly the material usually omitted from journal papers. Treat it as an annotated bibliography plus implementation guide rather than a source of new results.

---

## 8. Pricing of vanilla and first generation exotic options in the local stochastic volatility framework: survey and new results

**Authors:** Alexander Lipton, Andrey Gal, Andris Lasis
**Year:** 2013 arXiv; published *Quantitative Finance* 14(11), 2014
**Type:** Survey with new results
**Link:** https://arxiv.org/abs/1312.5693 (verified: free full text, 59 pages)

A survey of SV and LSV models covering vanillas and first-generation exotics (barriers, one-touches), with emphasis on semi-analytical machinery. The new contribution is a Galerkin–Ritz alternative to standard ADI finite differences, plus closed-form series expansions in powers of the spot-variance correlation for the zero-correlation base case, benchmarked against analytical solutions. The value is the set of benchmark solutions: something exact to validate a calibration/pricing implementation against.

---

## 9. The calibration of stochastic-local volatility models: an inverse problem perspective

**Authors:** Yuri F. Saporito, Xu Yang, Jorge P. Zubelli
**Year:** 2017 arXiv; published *Computers & Mathematics with Applications* 77(12), 2019
**Type:** Journal article, methodological comparison
**Link:** https://arxiv.org/abs/1711.03023 (verified: free full text)

Rather than treating leverage-function calibration as a fixed-point/projection problem, this poses it as a regularized inverse problem solved directly against quoted option prices, avoiding the intermediate step of interpolating an implied volatility surface and differentiating it. The payoff is stability in regions of low joint density of spot and instantaneous variance, precisely where particle and PDE methods produce noisy or blown-up leverage estimates. Read it for a clear diagnosis of why the standard approaches are ill-conditioned in the wings, even if you do not adopt the Tikhonov machinery.

---

## 10. Well-posedness of the calibrated LSV equation (three-paper overview)

**Type:** Journal articles — brief orientation only

- **Abergel & Tachet, "A nonlinear partial integro-differential equation from mathematical finance", *Discrete and Continuous Dynamical Systems* 27(3), 907–917, 2010.** https://www.aimsciences.org/article/doi/10.3934/dcds.2010.27.907 (preprint: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1508490). Proves **short-time** existence for the nonlinear PIDE governing the calibrated joint density, via a fixed-point argument and a priori estimates. The nonlocality is a quotient of two integrals, which is why the equation is not defined for all bounded continuous data — the analytic source of the blow-up that practitioners see as leverage-function instability.
- **Lacker, Shkolnikov & Zhang, "Inverting the Markovian projection, with an application to local stochastic volatility models", *Annals of Probability* 48(5), 2189–2211, 2020.** https://arxiv.org/abs/1905.06213. Studies the 2-D McKean–Vlasov SDE in which the conditional law of the variance given spot enters the spot equation. Proves strong existence of **stationary** solutions and strong uniqueness in an important special case. The framing sentence worth quoting: these equations are used pervasively in industry calibration "despite the very limited theoretical understanding".
- **Djete, "Non-regular McKean–Vlasov equations and calibration problem in local stochastic volatility models", 2022 (rev. 2024).** https://arxiv.org/abs/2208.09986. Handles McKean–Vlasov equations whose coefficients are not continuous in the measure argument for the Wasserstein topology; obtains existence, proves propagation of chaos for the N-particle approximation, and deduces existence of a calibrated LSV model for suitable parameters. Currently the strongest general existence result for the calibration problem.

---

## 11. Numerical analysis of a particle system for the calibrated Heston-type local stochastic volatility model

**Authors:** Christoph Reisinger, Maria Olympia Tsianni
**Year:** 2025
**Type:** Journal article (numerical analysis)
**Link:** https://arxiv.org/abs/2504.14343 (verified: free full text)

The paper that connects the well-posedness literature to the algorithm actually run on desks: it analyses the particle method *as implemented*, with a kernel estimator for the conditional expectation, which turns the calibration into a McKean–Vlasov SDE with non-standard coefficients and 1/2-Hölder diffusion. For fixed but arbitrarily small bandwidth it establishes well-posedness, proves strong propagation of chaos under a Feller-ratio condition up to a critical time, and gets strong convergence of Euler–Maruyama at rate 1/2 in time up to a log factor. This is the reference for what is and is not guaranteed about particle calibration, and the Feller-ratio condition explains the empirical fragility at high vol-of-vol.

---

## 12. Simulations of calibrated Local Stochastic Volatility models

**Author:** Minyuan Li (supervisor: Wolfgang Stockinger)
**Year:** 2023
**Type:** MSc thesis, Imperial College London, MSc Mathematics and Finance 2022–23
**Link:** https://www.imperial.ac.uk/media/imperial-college/faculty-of-natural-sciences/department-of-mathematics/math-finance/212188218---Minyuan-Li---LI_MINYUAN_02278349.pdf (verified: free PDF)

A freely available, readable thesis that discusses and compares the existing methods for estimating the leverage/conditional-expectation term in calibrated LSV models via particle simulation, covering both theoretical and numerical difficulties of the calibrated dynamics. Written with input from the JP Morgan QRFX team, so the treatment is grounded in what an FX desk actually does. Two companion Imperial MSc theses are worth having alongside it: Martin Weirich, *Fokker-Planck Calibration of one Factor Stochastic Local Volatility* (https://www.imperial.ac.uk/media/imperial-college/faculty-of-natural-sciences/department-of-mathematics/math-finance/212268623---Martin-Weirich---WEIRICH_MARTIN_02274905.pdf) for the PDE branch, and Muaz Chowdhury, *Mixed Local Volatility Models* (https://www.imperial.ac.uk/media/imperial-college/faculty-of-natural-sciences/department-of-mathematics/math-finance/239242778---Muaz-Chowdhury---Chowdhury_Muaz_02291903.pdf) for leverage-function calibration in the mixed-LV setting.

---

## 13. Path-dependent volatility as the alternative to LSV (2023–2025)

**Authors:** Julien Guyon & Jordan Lekeufack; then Guido Gazzani & Julien Guyon
**Years:** 2023, 2024–2025
**Type:** Journal article plus follow-up papers
**Links:**
- Guyon & Lekeufack, "Volatility is (mostly) path-dependent", *Quantitative Finance* 23(9), 1221–1258, 2023: https://doi.org/10.1080/14697688.2023.2221281 (SSRN preprint: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4174589)
- Gazzani & Guyon, "Pricing and calibration in the 4-factor path-dependent volatility model", 2024 (rev. Feb 2025): https://arxiv.org/abs/2406.02319
- "Joint deep calibration of the 4-factor PDV model", 2025: https://arxiv.org/pdf/2507.09412

The empirical claim is that instantaneous volatility is largely explained by a weighted sum of past returns plus the square root of a weighted sum of past squared returns, giving a model that is genuinely path-dependent rather than Markovian in spot alone. The 4-factor version restores Markovianity in an augmented state, and Gazzani & Guyon use a pathwise neural approximation of the VIX to handle joint SPX/VIX calibration and to sample VIX paths cheaply. This matters as LSV's principal current competitor: PDV reaches joint SPX/VIX calibration, the standard failure case for LSV, without a leverage-function fixed point. For joint calibration more broadly the two other active lines are Gaussian polynomial volatility (Abi Jaber et al., *Mathematical Finance* 2025, https://arxiv.org/abs/2212.08297) and signature-based models (https://arxiv.org/abs/2301.13235); a scale-invariance analysis specific to LSV under joint SPX/VIX constraints is at https://arxiv.org/pdf/2302.08819.

---

### Two smaller items worth knowing about, not full entries

- **"Effective stochastic local volatility models", *Quantitative Finance* 23(12), 2023** (https://www.tandfonline.com/doi/abs/10.1080/14697688.2023.2271514): a semi-analytical "effective volatility" approximation allowing direct calibration of the leverage function for a broad class of LSV models, sidestepping both Monte Carlo and Fourier pricing. Relevant if calibration speed is the binding constraint.
- **"Amortizing the Calibration Triple: A Projection-Consistent Neural Operator for Local-Stochastic Volatility"** (https://arxiv.org/html/2608.01217, 2026): learns an operator mapping quotes plus an SV backbone to the arbitrage-free implied surface, its Dupire local volatility, the LSV leverage, and the conditional moment in the projection identity, with DeepONet and FNO implementations enforcing the Dupire and projection constraints. Newest thing in the space; recent and unvetted rather than endorsed.

### Caveats on link resolution

The Homescu and Guyon–Lekeufack SSRN pages both return HTTP 403 to automated fetching (SSRN blocks bots); the records are live and the PDFs download normally in a browser. Lipton 2002 (*Risk*) and the Guyon–Henry-Labordère book are publisher-paywalled; for those the canonical publisher link plus a freely circulating copy is given where one exists. Everything else on this list resolves to free full text.
