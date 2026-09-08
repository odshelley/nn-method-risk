# Proof Package

Target: contraction property of the damped fixed-point iteration of the implicit scheme
(`paper/notes.tex` §5, Algorithm 2; `paper/notes_convergence.tex`; code
`src/neural_particle_method/calibrate/implicit.py`).

## Claim

**Original claim (as posed).** The damped iteration of the implicit scheme,
$$
L^{n+1} = (1-\alpha)\,L^{n} + \alpha\,\Phi(L^{n}), \qquad \alpha \in (0,1],
$$
is a contraction, hence has a unique fixed point $L^*$ (the calibrated leverage) to which $L^n$ converges geometrically. The suggested route: the implicit scheme is close to the explicit scheme at early times, and the damping factor supplies the rest.

**What is proved below (corrected claims).** For the *time-discretised, population-level* map with a bounded leverage domain and a regularised conditional-expectation estimator:

- **Theorem 1.** If $\Phi$ is *causal* (slice $k$ of $\Phi(L)$ depends only on slices $0,\dots,k-1$ of $L$) and slice-wise Lipschitz, then $\Phi$ has exactly one fixed point $L^*$, which is the output of the explicit scheme run with the same estimator; for **every** $\alpha\in(0,1]$ the damped map $\Phi_\alpha$ is a contraction in a time-weighted norm, and the undamped iteration terminates exactly in $K$ steps. Damping is not needed and only slows the rate.
- **Proposition 3 (no-go).** Under Lipschitz hypotheses alone damping can never turn a non-contraction into a contraction. Any proof in which $\alpha$ is load-bearing must use a one-sided (dissipativity) condition, which is the nonlinear form of the spectral condition $\operatorname{Re}\lambda<1$ in `notes_convergence.tex`.
- **Theorem 2.** For the implemented map $\Psi = \Phi + D + G$ (causal part, same-slice feedback $D$ created by the global network's temporal smoothing, small anti-causal remainder $G$), if the feedback is one-sided Lipschitz with constant $\kappa<1$, then $\Psi_\alpha$ is a contraction in the weighted norm for all $\alpha$ below an explicit threshold, with an explicit rate. Here damping is load-bearing, and the early-time closeness to the explicit output is the weight that neutralises the causal part.
- **Theorem 3.** With bounded (non-Lipschitz) Monte Carlo and training noise the iterates enter and stay in an explicit neighbourhood of the fixed point.

## Status

**PROVABLE AFTER WEAKENING / EXTRA ASSUMPTION.**

The claim is not currently justified for the continuum map $\Phi$ of eq. (phimap) in the notes (open well-posedness; see Literature). It is provable for the discretised, regularised map under Assumptions A and B below. Assumption A (slice Lipschitz stability) is known for kernel-ridge estimators and open for the neural estimator. Assumption B (one-sided feedback constant $\kappa<1$) is the structural hypothesis carrying all of the damping content; it is not established for the implemented scheme and is flagged as measurable.

## Assumptions

Fixed data: horizon $T>0$, $K\in\mathbb{N}$ steps, $\Delta := T/K$, $t_k := k\Delta$. A bounded Dupire surface, $0\le \sigma_{\mathrm{Dup}}\le \sigma_{\max}<\infty$ on the region used. A leverage cap $L_{\max}>0$ and a variance floor $f_{\min}>0$ (code: `L_max = 4.0`, floor `1e-4`).

- **(A0) Slice spaces.** For each $k\in\{0,\dots,K-1\}$, $H_k$ is a real Hilbert space of functions of the log-spot with norm $\|\cdot\|_k$ in which pointwise clipping $u\mapsto \min(\max(u,0),L_{\max})$ is $1$-Lipschitz. Examples: $H_k=L^2(\nu_k)$ for a fixed reference measure $\nu_k$, or $H_k=\mathbb{R}^{G}$ with the Euclidean norm for the code's fixed grid of $G=81$ points.
- **(A1) Domain.** $\mathcal{L} := \{L=(L_0,\dots,L_{K-1}) : L_k\in H_k,\ 0\le L_k\le L_{\max}\ \text{a.e.}\}$.
- **(A2) Causality.** There are maps $\varphi_k : H_0\times\cdots\times H_{k-1}\to H_k$ with $\Phi(L)_k=\varphi_k(L_0,\dots,L_{k-1})$ for all $L\in\mathcal{L}$, $\varphi_0$ constant, and $\Phi(\mathcal{L})\subseteq\mathcal{L}$.
- **(A3) Slice Lipschitz stability.** There are constants $\Lambda_{kj}\ge 0$ ($0\le j<k\le K-1$) with
$$
\|\Phi(L)_k-\Phi(M)_k\|_k\ \le\ \sum_{j<k}\Lambda_{kj}\,\|L_j-M_j\|_j \qquad (L,M\in\mathcal{L}).
$$
Put $\Lambda := \max_{k,j}\Lambda_{kj}$.

Assumption A := (A0)–(A3). Assumption B is stated before Theorem 2.

**Where (A2), (A3) come from.** The population-level discrete map is
$$
\Phi(L)_k \;=\; \operatorname{clip}_{[0,L_{\max}]}\!\Big(\frac{\sigma_{\mathrm{Dup}}(t_k,\cdot)}{\sqrt{\max\big(R_k(\mu_k^{L}),\,f_{\min}\big)}}\Big),
$$
where $\mu_k^{L}$ is the law of $(\ln X_{t_k},V_{t_k})$ under the Euler chain of `heston_step` driven by $L_0,\dots,L_{k-1}$ (the step from $t_j$ to $t_{j+1}$ uses $L_j$ only; `simulate_slices`, `calibrate_explicit`), and $R_k$ is the regression operator returning the estimate of $\mathbb{E}[V_{t_k}\mid \ln X_{t_k}=\cdot\,]$ from the law. (A2) is then a fact, not an assumption: $\mu_0=\delta_{(\ln s_0,v_0)}$ does not depend on $L$, and $\mu_k^{L}$ depends on $L_0,\dots,L_{k-1}$ only. (A3) follows from three ingredients:

1. **Algebraic step (proved).** For $y,y'\ge 0$, $\big|\max(y,f_{\min})^{-1/2}-\max(y',f_{\min})^{-1/2}\big|\le \tfrac12 f_{\min}^{-3/2}|y-y'|$, because $y\mapsto\max(y,f_{\min})^{-1/2}$ is continuous, piecewise $C^1$, with derivative $0$ on $y<f_{\min}$ and $-\tfrac12 y^{-3/2}$ on $y>f_{\min}$, so it is Lipschitz with constant $\sup|{\rm derivative}|=\tfrac12 f_{\min}^{-3/2}$. Multiplying by $0\le\sigma_{\mathrm{Dup}}\le\sigma_{\max}$ and clipping (1-Lipschitz by (A0)) gives $\|\Phi(L)_k-\Phi(M)_k\|_k\le \tfrac12\sigma_{\max}f_{\min}^{-3/2}\,\|R_k(\mu_k^L)-R_k(\mu_k^M)\|_k$.
2. **Estimator stability (assumed; known for kernel ridge).** $\|R_k(\mu)-R_k(\mu')\|_k\le C_R\, d(\mu,\mu')$ for a metric $d$ on laws (e.g. $W_1$ or $W_2$). Bayer, Belomestny, Butkovsky and Schoenmakers prove this for the RKHS ridge estimator: "we prove that under suitable conditions, $m^\lambda_A(x;\nu)$ is Lipschitz in both arguments, that is, w.r.t. the standard Euclidean norm in $x$ and the Wasserstein-1-norm in $\nu$" [Bayer et al. 2024, chunk 4]. For the exact conditional expectation this fails: the dependence on the law is "non-regular (i.e., non-Lipschitz/non-continue) for the Wasserstein distance" [Djete 2024, chunk 0]. For the softplus network trained by Adam, no such result exists.
3. **Chain stability (assumed; standard under bounded Lipschitz coefficients).** $d(\mu_k^L,\mu_k^M)\le \sum_{j<k}c_{kj}\|L_j-M_j\|_j$. For Lipschitz-regularised coefficients this is the one-step/moment stability used by Reisinger and Tsianni (their Propositions 3 and 4) [Reisinger–Tsianni 2023]. It requires the leverage to be bounded (given by $L_{\max}$) and, for $W_2$-stability of the Euler pushforward, uniformly Lipschitz in $x$; on the code's finite grid with linear interpolation (`LeverageField.at`) this is automatic with a grid-dependent constant.

Then $\Lambda_{kj}=\tfrac12\sigma_{\max}f_{\min}^{-3/2}C_R\,c_{kj}$.

## Notation

- $\beta := 1-\alpha$. $\Phi_\alpha := (1-\alpha)\,\mathrm{Id}+\alpha\Phi$; likewise $\Psi_\alpha$.
- Weighted norm, for $\gamma\ge 1$: $\ \|L\|_\gamma := \max_{0\le k<K}\gamma^{-k}\|L_k\|_k$. Sup norm $\|L\|_\infty := \max_k\|L_k\|_k$. One has $\|L\|_\gamma\le\|L\|_\infty\le\gamma^{K-1}\|L\|_\gamma$.
- Weighted causal constant $\ \Lambda_\gamma := \max_k\sum_{j<k}\Lambda_{kj}\gamma^{\,j-k}$.
- $N\in\mathbb{R}^{K\times K}$: the strictly lower-triangular nonnegative matrix with entries $N_{kj}=\Lambda_{kj}$ for $j<k$, zero otherwise. $N^{K}=0$.
- $e^n_k := \|L^n_k-L^*_k\|_k$, $e^n := (e^n_0,\dots,e^n_{K-1})^\top$; vector inequalities are componentwise.
- $L^{\mathrm{expl}}$: output of the explicit scheme (Algorithm 1) run with the same estimator $R_k$ and the same cap and floor.

## Proof Strategy

Banach fixed-point theorem in a time-weighted norm (a discrete Bielecki norm). The weight $\gamma^{-k}$ makes the causal part of the map as small as desired (Lemma 2); this is the rigorous form of "the implicit scheme is pinned to the explicit scheme at early times". Damping is then analysed separately: under Lipschitz information alone it cannot help (Proposition 3); under a one-sided condition on the same-slice feedback it does, with a threshold and a rate that coincide with the linear spectral algebra of `notes_convergence.tex` (Theorem 2, Remark 2.2). Noise is handled by the standard perturbation lemma for contractions (Theorem 3).

## Dependency Map

1. **Lemma 1** (explicit scheme is the unique fixed point of the causal map) uses (A1), (A2) only.
2. **Lemma 2** (weighted Lipschitz constant of the causal part) uses (A3) and the geometric series.
3. **Lemma 3** (completeness of $(\mathcal{L},\|\cdot\|_\gamma)$) uses (A0), (A1).
4. **Theorem 1** uses Lemmas 1–3 and Banach's fixed-point theorem; the slice-wise bound uses only nonnegativity of $N$ and the binomial theorem for commuting matrices.
5. **Proposition 3** is elementary (a one-dimensional example).
6. **Lemma 4** (one-sided step) uses the Hilbert structure of $H_k$ (A0), Cauchy–Schwarz, and $\sqrt{1+y}\le 1+y/2$ for $y\ge -1$.
7. **Theorem 2** uses Lemmas 2–4 and Banach's theorem; **Corollary 2.1** uses Theorem 2 and Lemma 1.
8. **Theorem 3** uses Theorem 2 and a geometric-series bound.
9. Boundary cases handled explicitly: slice $k=0$ (no dependence on $L$), $\alpha=1$ (finite termination), $\kappa\ge 1$ (damping cannot help), $\gamma\to\infty$ trade-off against anti-causal coupling.

## Proof

### Lemma 1 (the explicit scheme is the unique fixed point of the causal map)

Assume (A1), (A2). Define $L^*$ recursively by $L^*_0:=\varphi_0$ and $L^*_k:=\varphi_k(L^*_0,\dots,L^*_{k-1})$ for $k\ge 1$. Then:

(i) $L^*\in\mathcal{L}$ and $\Phi(L^*)=L^*$; (ii) $L^*$ is the only fixed point of $\Phi$ in $\mathcal{L}$; (iii) $L^*=L^{\mathrm{expl}}$; (iv) for every $L\in\mathcal{L}$ and $n\ge 1$, $\Phi^n(L)_k=L^*_k$ for all $k<n$; in particular $\Phi^K\equiv L^*$ is constant.

*Proof.* (i) By induction on $k$: $L^*_0=\varphi_0=\Phi(L)_0\in[0,L_{\max}]$ for any $L\in\mathcal{L}$; if $L^*_0,\dots,L^*_{k-1}$ lie in $[0,L_{\max}]$, pad them with zero slices to obtain some $L\in\mathcal{L}$, and then $L^*_k=\varphi_k(L^*_0,\dots,L^*_{k-1})=\Phi(L)_k\in[0,L_{\max}]$ because $\Phi(\mathcal{L})\subseteq\mathcal{L}$. Hence $L^*\in\mathcal{L}$, and $\Phi(L^*)_k=\varphi_k(L^*_0,\dots,L^*_{k-1})=L^*_k$ by construction.
(ii) If $\Phi(L)=L$ then $L_0=\varphi_0=L^*_0$, and if $L_j=L^*_j$ for all $j<k$ then $L_k=\varphi_k(L_0,\dots,L_{k-1})=\varphi_k(L^*_0,\dots,L^*_{k-1})=L^*_k$. Induction on $k$ gives $L=L^*$.
(iii) The explicit scheme (`calibrate_explicit`) computes, at step $k$, the estimate on the cloud generated by $L_0,\dots,L_{k-1}$, applies the floor, the Dupire ratio, and the cap, and sets $L_k$ to the result. That is exactly the recursion defining $L^*$ (with the population law in place of the particle cloud; the particle-level difference is part of the noise in Theorem 3).
(iv) Induction on $n$. For $n=1$: $\Phi(L)_0=\varphi_0=L^*_0$. If the claim holds for $n$, then for $k<n+1$, $\Phi^{n+1}(L)_k=\varphi_k\big(\Phi^n(L)_0,\dots,\Phi^n(L)_{k-1}\big)$, and all arguments have index $<n$, so they equal $L^*_0,\dots,L^*_{k-1}$, whence $\Phi^{n+1}(L)_k=L^*_k$. $\square$

*Remark 1.1.* Existence and uniqueness of the discrete fixed point require no Lipschitz property at all; they are a consequence of causality. This is the discrete counterpart of the observation in `notes.tex` §4 that "the explicit scheme contains no fixed point": the explicit scheme *is* the fixed point, computed by forward substitution. The convergence question for the implicit scheme is therefore not about existence but about the rate, and about robustness to the non-causal coupling introduced by the global network (Theorem 2).

### Lemma 2 (weighted Lipschitz constant of the causal part)

Assume (A3). For $\gamma>1$ and $L,M\in\mathcal{L}$,
$$
\|\Phi(L)-\Phi(M)\|_\gamma\ \le\ \Lambda_\gamma\,\|L-M\|_\gamma,\qquad \Lambda_\gamma\le\frac{\Lambda}{\gamma-1}.
$$

*Proof.* Write $d:=L-M$. For each $k$,
$$
\gamma^{-k}\|\Phi(L)_k-\Phi(M)_k\|_k\le\sum_{j<k}\Lambda_{kj}\gamma^{-k}\|d_j\|_j=\sum_{j<k}\Lambda_{kj}\gamma^{\,j-k}\,\big(\gamma^{-j}\|d_j\|_j\big)\le\Big(\sum_{j<k}\Lambda_{kj}\gamma^{\,j-k}\Big)\|d\|_\gamma .
$$
Taking the maximum over $k$ gives the first inequality. For the second, $\sum_{j<k}\Lambda_{kj}\gamma^{j-k}\le\Lambda\sum_{m=1}^{k}\gamma^{-m}<\Lambda\sum_{m\ge 1}\gamma^{-m}=\Lambda/(\gamma-1)$. $\square$

*Remark L2 (uniformity in $\Delta$).* If (A3) is available in the squared Volterra form $\|\Phi(L)_k-\Phi(M)_k\|_k^2\le C\sum_{j<k}\Delta\,\|L_j-M_j\|_j^2$ (the form produced by an $L^2$ stability estimate for the Euler chain), then with the exponential weight $\|L\|_{\gamma,\exp}:=\max_k e^{-\gamma t_k}\|L_k\|_k$ the same computation gives $e^{-2\gamma t_k}\|\Phi(L)_k-\Phi(M)_k\|^2_k\le C\|d\|^2_{\gamma,\exp}\sum_{m\ge1}\Delta e^{-2\gamma m\Delta}\le \frac{C}{2\gamma}\|d\|^2_{\gamma,\exp}$, using $\Delta/(e^{2\gamma\Delta}-1)\le 1/(2\gamma)$. So $\Lambda_\gamma\le\sqrt{C/(2\gamma)}$, uniformly in $\Delta$. This is the discrete Bielecki–Gronwall structure behind the short-time existence results (Abergel–Tachet: "by considering a discretized version (in time and space) of the Fokker–Planck equation ... the authors use a fixed point argument for establishing an existence result in short-time" [Djete 2024, chunk 1]). Whether $C$ is finite for the exact conditional expectation is exactly the open question.

### Lemma 3 (completeness)

For every $\gamma\ge1$, $(\mathcal{L},\|\cdot\|_\gamma)$ is a complete metric space, and $\mathcal{L}$ is convex.

*Proof.* $\prod_k H_k$ with $\|\cdot\|_\gamma$ is a Banach space: it is a finite product of Hilbert spaces and $\|\cdot\|_\gamma$ is equivalent to the product max-norm (factor $\gamma^{K-1}$). $\mathcal{L}$ is closed: if $L^{(m)}\to L$ in $\|\cdot\|_\gamma$ then $L^{(m)}_k\to L_k$ in $H_k$ for each $k$; in $L^2(\nu_k)$ a subsequence converges a.e., so $0\le L_k\le L_{\max}$ a.e.; in $\mathbb{R}^G$ the bounds are closed conditions. A closed subset of a Banach space is complete. Convexity: $[0,L_{\max}]$ is convex pointwise. $\square$

### Theorem 1 (damped iteration on the exact causal map)

Assume A. Let $\alpha\in(0,1]$ and $\gamma>1+\Lambda$. Then:

(a) $\Phi_\alpha$ maps $\mathcal{L}$ into itself and is a contraction on $(\mathcal{L},\|\cdot\|_\gamma)$ with constant
$$
r_\gamma(\alpha)\;:=\;1-\alpha\,(1-\Lambda_\gamma)\;\le\;1-\alpha\Big(1-\frac{\Lambda}{\gamma-1}\Big)\;<\;1 .
$$
(b) For every $L^0\in\mathcal{L}$ the iterates $L^{n+1}=\Phi_\alpha(L^n)$ converge to $L^*=L^{\mathrm{expl}}$, with
$$
\|L^n-L^*\|_\gamma\le r_\gamma(\alpha)^n\,\|L^0-L^*\|_\gamma,\qquad \|L^n-L^*\|_\infty\le\gamma^{K-1}\,r_\gamma(\alpha)^n\,\|L^0-L^*\|_\infty .
$$
(c) Slice-wise, $e^{n}\le(\beta I+\alpha N)^n e^{0}$ componentwise, i.e.
$$
e^n_k\ \le\ \sum_{m=0}^{\min(n,k)}\binom{n}{m}\beta^{\,n-m}\alpha^{m}\,(N^m e^0)_k .
$$
In particular $e^n_0=\beta^n e^0_0$, $e^n_k=O(n^k\beta^n)$, and for $\alpha=1$ one has $e^n_k=0$ for all $k<n$: the undamped iteration terminates exactly after $K$ steps.

*Proof.* (a) Self-map: for $L\in\mathcal{L}$, $\Phi_\alpha(L)=(1-\alpha)L+\alpha\Phi(L)$ is a convex combination of two elements of the convex set $\mathcal{L}$ (Lemma 3, (A2)). Contraction: for $L,M\in\mathcal{L}$, by the triangle inequality and Lemma 2,
$$
\|\Phi_\alpha(L)-\Phi_\alpha(M)\|_\gamma\le(1-\alpha)\|L-M\|_\gamma+\alpha\|\Phi(L)-\Phi(M)\|_\gamma\le\big(1-\alpha+\alpha\Lambda_\gamma\big)\|L-M\|_\gamma .
$$
Since $\gamma>1+\Lambda$, Lemma 2 gives $\Lambda_\gamma\le\Lambda/(\gamma-1)<1$, so $r_\gamma(\alpha)<1$ for $\alpha>0$.

(b) Banach's fixed-point theorem on the complete space $(\mathcal{L},\|\cdot\|_\gamma)$ (Lemma 3) yields a unique fixed point of $\Phi_\alpha$ and the geometric estimate. The fixed points of $\Phi_\alpha$ and of $\Phi$ coincide ($\Phi_\alpha(L)=L\iff\alpha(\Phi(L)-L)=0\iff\Phi(L)=L$ as $\alpha>0$), so the limit is $L^*$, which equals $L^{\mathrm{expl}}$ by Lemma 1(iii). The sup-norm bound follows from $\|\cdot\|_\infty\le\gamma^{K-1}\|\cdot\|_\gamma$ and $\|\cdot\|_\gamma\le\|\cdot\|_\infty$.

(c) Using $L^*=\Phi(L^*)$, for each $k$,
$$
L^{n+1}_k-L^*_k=\beta\,(L^n_k-L^*_k)+\alpha\,\big(\Phi(L^n)_k-\Phi(L^*)_k\big),
$$
so by the triangle inequality and (A3), $e^{n+1}_k\le\beta e^n_k+\alpha\sum_{j<k}\Lambda_{kj}e^n_j$, i.e. $e^{n+1}\le(\beta I+\alpha N)e^n$ componentwise. Since $\beta I+\alpha N$ has nonnegative entries it preserves componentwise inequalities, and induction gives $e^n\le(\beta I+\alpha N)^n e^0$. As $I$ and $N$ commute, the binomial theorem gives $(\beta I+\alpha N)^n=\sum_{m=0}^{n}\binom nm\beta^{n-m}\alpha^mN^m$. Row $k$ of $N^m$ vanishes for $m>k$ (a product of $m$ strictly lower-triangular matrices has zero rows $0,\dots,m-1$), which truncates the sum at $m\le k$. The special cases: $k=0$ leaves only $m=0$; the polynomial-times-geometric bound follows from $\binom nm\le n^m$; for $\alpha=1$, $\beta=0$ leaves only $m=n$, and $(N^ne^0)_k=0$ for $k<n$. $\square$

*Remark 1.2 (what Theorem 1 says about damping).* The rate $r_\gamma(\alpha)=1-\alpha(1-\Lambda_\gamma)$ is decreasing in $\alpha$: on the exact causal map, damping strictly slows convergence, and $\alpha=1$ is optimal (finite termination). The linearisation $D\Phi(L^*)$ is strictly lower block-triangular, so its spectrum is $\{0\}$, and the spectral condition of `notes_convergence.tex` holds trivially with $\lambda=0$. The empirical benefit of damping must therefore come from the non-causal part of the implemented map, which is the subject of Theorem 2.

### Proposition 3 (Lipschitz information alone cannot make damping essential)

Let $(E,\|\cdot\|)$ be a normed space, $C\subseteq E$ convex, and $\Psi:C\to C$ Lipschitz with constant $\lambda$. Then $\Psi_\alpha:=(1-\alpha)\mathrm{Id}+\alpha\Psi$ has Lipschitz constant at most $1-\alpha+\alpha\lambda$, and this bound is attained: for $E=C=\mathbb{R}$, $\Psi(x)=\lambda x$, the constant of $\Psi_\alpha$ is exactly $|1-\alpha+\alpha\lambda|=1-\alpha+\alpha\lambda$. Consequently, a Lipschitz constant $\lambda$ guarantees that $\Psi_\alpha$ is a contraction if and only if $\lambda<1$, and then the constant is minimised at $\alpha=1$.

*Proof.* The bound is the triangle inequality. In the example, $\Psi_\alpha(x)=(1-\alpha+\alpha\lambda)x$ is linear with the stated constant; $1-\alpha+\alpha\lambda<1\iff\alpha(\lambda-1)<0\iff\lambda<1$ for $\alpha>0$, and it is decreasing in $\alpha$ when $\lambda<1$. $\square$

*Consequence.* "Closeness to the explicit method plus damping" cannot yield a proof in which $\alpha$ matters if the closeness is expressed only through Lipschitz constants. The additional structure needed is a *one-sided* Lipschitz (dissipativity) condition, as in Assumption B, which is what the linear spectral algebra $\mu=1+\alpha(\lambda-1)$ of `notes_convergence.tex` implicitly uses (it needs $\operatorname{Re}\lambda<1$, a one-sided statement, not $|\lambda|<1$).

### Assumption B (structure of the implemented map)

The implemented outer map, idealised at population level and with the network trained to its minimiser, is a map $\Psi:\mathcal{L}\to\mathcal{L}$ admitting a decomposition
$$
\Psi(L)_k\;=\;\Phi(L)_k\;+\;D_k(L_k)\;+\;G(L)_k,\qquad 0\le k<K,
$$
where

- **(B1)** $\Phi$ satisfies Assumption A (causal part);
- **(B2)** $D_k:\{u\in H_k:0\le u\le L_{\max}\}\to H_k$ is the *same-slice feedback*, with constants $\kappa\in\mathbb{R}$, $\lambda_D\ge0$ independent of $k$:
$$
\langle D_k(u)-D_k(u'),\,u-u'\rangle_k\le\kappa\,\|u-u'\|_k^2,\qquad\|D_k(u)-D_k(u')\|_k\le\lambda_D\|u-u'\|_k ;
$$
- **(B3)** $G$ is the *anti-causal remainder*, with weighted Lipschitz constant $g_\gamma$: $\|G(L)-G(M)\|_\gamma\le g_\gamma\|L-M\|_\gamma$.

Provenance in the code. The global network takes $(t/T,\,x)$ as input and is smooth in $t$; the slice at grid time $t_k$ is read off as $f_\theta(t_k,\cdot)$ (`calibrate_implicit`, evaluation on `field` slice times), while the pooled training data live at $t_1,\dots,t_K$ and the data at $t_{k+1}$ were generated with $L_k$. So $\Psi(L)_k$ depends on $L_k$ (this is $D_k$) and, through temporal smoothing, on $L_{k+1},L_{k+2},\dots$ (this is $G$). This backward flow of information through the shared parameters is what `notes.tex` §5 identifies as "the actual implicitness". Necessarily $-\lambda_D\le\kappa\le\lambda_D$ by Cauchy–Schwarz (if the domain contains two distinct points).

For linear $D_k=A_k$ on $H_k=\mathbb{R}^G$: $\kappa=\max_k\lambda_{\max}\big(\tfrac12(A_k+A_k^\top)\big)$ (the numerical-range abscissa) and $\lambda_D=\max_k\|A_k\|_2$. This is the quantity `notes.tex` §8.5 proposes to measure by power iteration on the frozen-body head Jacobian; the symmetric part is what must be measured, not the spectrum.

### Lemma 4 (one-sided step)

Under (B2), for $u,u'$ in the domain of $D_k$, $d:=u-u'$, $w:=D_k(u)-D_k(u')$, and $\alpha\in(0,1]$,
$$
\|\beta d+\alpha w\|_k\ \le\ \sqrt{q(\alpha)}\;\|d\|_k,\qquad q(\alpha):=1-2\alpha(1-\kappa)+\alpha^2c,\qquad c:=1-2\kappa+\lambda_D^2 ,
$$
with $q(\alpha)\ge0$ and $c\ge(1-\kappa)^2$. Moreover, if $\kappa<1$,
$$
\sqrt{q(\alpha)}\ \le\ 1-\alpha(1-\kappa)+\tfrac12\alpha^2c .
$$

*Proof.* Expand in the Hilbert norm: $\|\beta d+\alpha w\|^2=\beta^2\|d\|^2+2\alpha\beta\langle w,d\rangle+\alpha^2\|w\|^2\le\big(\beta^2+2\alpha\beta\kappa+\alpha^2\lambda_D^2\big)\|d\|^2$, using $\beta\ge0$ and (B2). With $\beta=1-\alpha$: $\beta^2+2\alpha\beta\kappa+\alpha^2\lambda_D^2=1-2\alpha+\alpha^2+2\alpha\kappa-2\alpha^2\kappa+\alpha^2\lambda_D^2=q(\alpha)$. Nonnegativity: $q(\alpha)\ge\beta^2-2\alpha\beta\lambda_D+\alpha^2\lambda_D^2=(\beta-\alpha\lambda_D)^2\ge0$ because $\kappa\ge-\lambda_D$. Also $c=1-2\kappa+\lambda_D^2\ge1-2\kappa+\kappa^2=(1-\kappa)^2$ since $\lambda_D\ge|\kappa|$. For the last inequality, put $y:=q(\alpha)-1\ge-1$; then $\sqrt{1+y}\le1+y/2$ because $(1+y/2)^2=1+y+y^2/4\ge1+y$ and $1+y/2\ge0$ for $y\ge-1$ (for $y\ge -2$ in fact). Substituting $y=-2\alpha(1-\kappa)+\alpha^2c$ gives the claim. $\square$

### Theorem 2 (damped iteration on the implemented map)

Assume B with $\kappa<1$. Let $c:=1-2\kappa+\lambda_D^2$ and suppose $\gamma>1$ is such that
$$
s_\gamma\;:=\;(1-\kappa)-\Lambda_\gamma-g_\gamma\;>\;0 .
$$
Then for every $\alpha\in(0,1]$ with $\alpha<2s_\gamma/c$:

(a) $\Psi_\alpha$ maps $\mathcal{L}$ into itself and is a contraction on $(\mathcal{L},\|\cdot\|_\gamma)$ with constant
$$
\rho_\gamma(\alpha)\;:=\;\sqrt{q(\alpha)}+\alpha\big(\Lambda_\gamma+g_\gamma\big)\;\le\;1-\alpha\Big(s_\gamma-\tfrac12\alpha c\Big)\;<\;1 .
$$
(b) $\Psi$ has a unique fixed point $L^\sharp\in\mathcal{L}$, and $\|L^n-L^\sharp\|_\gamma\le\rho_\gamma(\alpha)^n\|L^0-L^\sharp\|_\gamma$ for every $L^0\in\mathcal{L}$.
(c) The simplified rate bound is minimised at $\alpha^\star=\min(1,s_\gamma/c)$; if $s_\gamma\le c$ this gives $\rho_\gamma(\alpha^\star)\le1-s_\gamma^2/(2c)$.

*Proof.* (a) Self-map as in Theorem 1(a), using $\Psi(\mathcal{L})\subseteq\mathcal{L}$ and convexity. Fix $L,M\in\mathcal{L}$, $d:=L-M$. For each $k$,
$$
\Psi_\alpha(L)_k-\Psi_\alpha(M)_k=\underbrace{\beta d_k+\alpha\big(D_k(L_k)-D_k(M_k)\big)}_{(\mathrm{i})}+\alpha\underbrace{\big(\Phi(L)_k-\Phi(M)_k\big)}_{(\mathrm{ii})}+\alpha\underbrace{\big(G(L)_k-G(M)_k\big)}_{(\mathrm{iii})} .
$$
By Lemma 4, $\|(\mathrm{i})\|_k\le\sqrt{q(\alpha)}\|d_k\|_k$, hence $\gamma^{-k}\|(\mathrm{i})\|_k\le\sqrt{q(\alpha)}\|d\|_\gamma$. By Lemma 2, $\max_k\gamma^{-k}\|(\mathrm{ii})\|_k\le\Lambda_\gamma\|d\|_\gamma$. By (B3), $\max_k\gamma^{-k}\|(\mathrm{iii})\|_k\le g_\gamma\|d\|_\gamma$. The triangle inequality and the maximum over $k$ give $\|\Psi_\alpha(L)-\Psi_\alpha(M)\|_\gamma\le\rho_\gamma(\alpha)\|d\|_\gamma$. The upper bound on $\rho_\gamma(\alpha)$ is Lemma 4's last inequality plus the definition of $s_\gamma$: $\sqrt{q(\alpha)}+\alpha(\Lambda_\gamma+g_\gamma)\le1-\alpha(1-\kappa)+\tfrac12\alpha^2c+\alpha(\Lambda_\gamma+g_\gamma)=1-\alpha(s_\gamma-\tfrac12\alpha c)$. This is $<1$ precisely when $s_\gamma-\tfrac12\alpha c>0$, i.e. $\alpha<2s_\gamma/c$, which is the stated range.
(b) Banach's theorem on $(\mathcal{L},\|\cdot\|_\gamma)$ (Lemma 3) applied to $\Psi_\alpha$; fixed points of $\Psi_\alpha$ and $\Psi$ coincide as in Theorem 1(b).
(c) $\alpha\mapsto1-\alpha s_\gamma+\tfrac12\alpha^2c$ is a convex quadratic with minimiser $s_\gamma/c$ and minimum value $1-s_\gamma^2/(2c)$; restricting to $\alpha\le1$ gives the stated $\alpha^\star$. $\square$

*Remark 2.1 (choice of $\gamma$; the trade-off).* $\Lambda_\gamma\le\Lambda/(\gamma-1)$ decreases in $\gamma$, but an anti-causal coupling of slice $k$ to slice $j>k$ with Lipschitz constant $g_{kj}$ contributes $g_{kj}\gamma^{\,j-k}$ to $g_\gamma$, which increases in $\gamma$. If $G$ couples only slices within $p$ steps ahead with total constant $g$, then $g_\gamma\le g\gamma^{p}$ and the hypothesis $s_\gamma>0$ reads $(1-\kappa)>\Lambda/(\gamma-1)+g\gamma^p$, solvable in $\gamma$ if and only if $g$ is small enough relative to $1-\kappa$ and $\Lambda$. This is the quantitative content of "the implicit map must be close to a causal map": anti-causal coupling must be small in a sense that competes with the causal Lipschitz constants, and the weighted norm is the device that arbitrates between them.

*Remark 2.2 (consistency with the linear spectral algebra).* Take $H_k=\mathbb{R}$, $\Phi=G=0$, $D_k(u)=\lambda u$ with $\lambda<1$ real. Then $\kappa=\lambda$, $\lambda_D=|\lambda|$, $c=(1-\lambda)^2$, and $q(\alpha)=(1-\alpha+\alpha\lambda)^2$, so $\sqrt{q(\alpha)}=|1+\alpha(\lambda-1)|$ is exactly the eigenvalue map $\mu=1+\alpha(\lambda-1)$ of `notes_convergence.tex`; the threshold $2s_\gamma/c=2(1-\lambda)/(1-\lambda)^2$ agrees with eq. (threshold) there, and the optimal $\alpha^\star=1/(1-\lambda)$ makes $\mu=0$. For a normal linear $D$ with complex spectrum the sharp constant is $\sup_\lambda|1+\alpha(\lambda-1)|$, whereas Lemma 4 uses only the numerical-range abscissa $\kappa$ and the norm $\lambda_D$, which is the "numerical range, not spectrum" caveat of `notes_convergence.tex` §2.2 made explicit; for non-normal $D$ the numerical-range version is the one that gives a contraction in the given norm.

*Remark 2.3 (what damping buys, precisely).* If $\lambda_D>1$ but $\kappa<1$, the undamped map $\Psi$ need not be a contraction in any norm on the slice (Proposition 3 shows Lipschitz data cannot rescue it), yet Theorem 2 gives a contraction for small $\alpha$. This is the only mechanism, within this framework, by which damping is load-bearing, and it depends on the sign structure of the same-slice feedback, not on its size.

### Corollary 2.1 (distance of the implicit fixed point from the explicit output; early slices are pinned first)

Under the hypotheses of Theorem 2, with $L^*=L^{\mathrm{expl}}$ from Lemma 1,
$$
\|L^\sharp-L^*\|_\gamma\ \le\ \frac{\|D(L^*)+G(L^*)\|_\gamma}{s_\gamma-\tfrac12\alpha c},\qquad\text{hence}\qquad\|L^\sharp_k-L^*_k\|_k\le\gamma^{k}\,\frac{\|D(L^*)+G(L^*)\|_\gamma}{s_\gamma-\tfrac12\alpha c}\quad(0\le k<K),
$$
where $D(L)_k:=D_k(L_k)$. The right-hand side is the residual of the explicit output under the implemented map, scaled by the reciprocal contraction gap; the slice-$k$ bound grows geometrically in $k$, which is the precise form of the statement that the implicit fixed point is closest to the explicit output at early times.

*Proof.* $L^\sharp-L^*=\Psi_\alpha(L^\sharp)-\Phi_\alpha(L^*)=\big[\Psi_\alpha(L^\sharp)-\Psi_\alpha(L^*)\big]+\alpha\big[\Psi(L^*)-\Phi(L^*)\big]$, and $\Psi(L^*)-\Phi(L^*)=D(L^*)+G(L^*)$. Taking $\|\cdot\|_\gamma$, Theorem 2(a) gives $\|L^\sharp-L^*\|_\gamma\le\rho_\gamma(\alpha)\|L^\sharp-L^*\|_\gamma+\alpha\|D(L^*)+G(L^*)\|_\gamma$, and $1-\rho_\gamma(\alpha)\ge\alpha(s_\gamma-\tfrac12\alpha c)>0$. Rearranging and using $\|L_k\|_k\le\gamma^k\|L\|_\gamma$ gives both bounds. $\square$

### Theorem 3 (bounded noise: Monte Carlo and training error)

Assume the hypotheses of Theorem 2 and let the computed iteration be
$$
L^{n+1}=\Psi_\alpha(L^n)+\alpha\,\eta_n,\qquad\|\eta_n\|_\gamma\le\eta\ \ \text{for all }n,
$$
where $\eta_n$ collects the finite-particle error of the cloud and the optimisation error of the network at iteration $n$ (it enters with the factor $\alpha$ because the code forms $(1-\alpha)L+\alpha\widehat\Phi$ with the noisy $\widehat\Phi$). If $L^n\in\mathcal{L}$ for all $n$ (guaranteed by the clipping in the code), then
$$
\|L^n-L^\sharp\|_\gamma\ \le\ \rho_\gamma(\alpha)^n\,\|L^0-L^\sharp\|_\gamma+\frac{\alpha\,\eta}{1-\rho_\gamma(\alpha)}\ \le\ \rho_\gamma(\alpha)^n\,\|L^0-L^\sharp\|_\gamma+\frac{\eta}{s_\gamma-\tfrac12\alpha c} .
$$

*Proof.* $\|L^{n+1}-L^\sharp\|_\gamma\le\|\Psi_\alpha(L^n)-\Psi_\alpha(L^\sharp)\|_\gamma+\alpha\|\eta_n\|_\gamma\le\rho\|L^n-L^\sharp\|_\gamma+\alpha\eta$ with $\rho:=\rho_\gamma(\alpha)$. Unrolling, $\|L^n-L^\sharp\|_\gamma\le\rho^n\|L^0-L^\sharp\|_\gamma+\alpha\eta\sum_{i<n}\rho^i\le\rho^n\|L^0-L^\sharp\|_\gamma+\alpha\eta/(1-\rho)$. The second form uses $1-\rho\ge\alpha(s_\gamma-\tfrac12\alpha c)$ from Theorem 2(a). $\square$

*Remark 3.1.* This is the "implementable theorem" of `notes_convergence.tex` §2.4 with explicit constants: the neighbourhood radius is $\alpha\eta/(1-\rho_\gamma(\alpha))$. Note that in the simplified bound the floor $\eta/(s_\gamma-\tfrac12\alpha c)$ *increases* with $\alpha$ while the rate $\rho_\gamma(\alpha)$ improves with $\alpha$ up to $\alpha^\star$; damping does not reduce the noise floor by more than the factor $s_\gamma/(s_\gamma-\tfrac12\alpha c)$, so its role is convergence, not variance reduction, in agreement with `notes_convergence.tex` §1 item 2.

Therefore the corrected claims follow. $\blacksquare$

## Corrections or Missing Assumptions

1. **The original claim is not provable for the continuum map** $\Phi$ of the notes. Well-posedness of the calibrated McKean–Vlasov SDE is open in the Heston-type continuous-state case; the law-dependence of the exact conditional expectation is not Lipschitz in Wasserstein distance [Djete 2024, chunk 0]. All statements above are for the time-discretised, capped, floored, regularised map.
2. **Assumption A (slice Lipschitz stability of the causal part)** is required for Theorem 1 and is an extra assumption. It is established for the RKHS ridge estimator [Bayer et al. 2024, chunk 4] and for mollified kernel estimators with Lipschitz-regularised coefficients [Reisinger–Tsianni 2023]; it is not established for the neural estimator used in the code. On the code's finite grid all maps are locally Lipschitz, but with constants that are neither known nor controlled.
3. **Assumption B, in particular $\kappa<1$,** is the load-bearing structural hypothesis for any statement in which $\alpha$ matters (Proposition 3). It is the nonlinear, numerical-range form of the notes' spectral condition, and it has no published counterpart: for the classical PDE inner iteration the contraction property "is stated but not proved" [Cozma et al. 2021, chunk 31]. It should be stated as a named hypothesis and measured in the frozen-body configuration (symmetric part of the head Jacobian, not its eigenvalues).
4. **The route "closeness to explicit + damping" is only half right.** Closeness to the explicit scheme (causality, weighted norm) removes the causal part of the Lipschitz constant and by itself gives Theorem 1 with $\alpha=1$ optimal. Damping is needed only against the same-slice feedback $D$ introduced by the shared network, and only under the one-sided condition. Neither ingredient can be dropped for the implemented scheme.
5. **The implemented iteration is not the iteration of a fixed map.** In `calibrate_implicit` the network and Adam state are created once and carried across outer iterations, and fresh random numbers are drawn each iteration. The theorems idealise $\Psi$ as deterministic (trained to a warm-start-independent minimiser) and push the rest into $\eta_n$ (Theorem 3). That idealisation is an assumption.
6. **Cap and floor are part of the map.** $L^*$ and $L^\sharp$ are fixed points of the clipped map. They coincide with the unclipped calibrated leverage only where the latter satisfies $0\le L\le L_{\max}$ and $\mathbb{E}[V\mid X]\ge f_{\min}$, which is the tail-degeneracy remark of `notes_convergence.tex` §2.3.

## Open Risks

- **Constants in the sup norm.** The weighted-norm contraction converts to the sup norm with the factor $\gamma^{K-1}$ (Theorem 1(b)); the slice-wise bound (Theorem 1(c)) avoids it but carries $n^k$ prefactors. With $\Lambda_{kj}$ not decaying in $k-j$ the bound $\Lambda/(\gamma-1)$ is not uniform in $\Delta$; uniformity needs the squared Volterra form of Remark L2, i.e. an $L^2$ stability estimate for the Euler chain, which is available for Lipschitz-regularised coefficients and not for the exact conditional expectation.
- **Transient growth.** $D\Phi(L^*)$ is nilpotent and highly non-normal; the entries $(N^m e^0)_k$ can be large before they vanish. Theorem 1(c) bounds the transient by $\sum_{m\le k}\|N^m\|$ for every $\alpha$, so damping does not reduce the transient bound in this analysis. The oscillatory overshoot reported for undamped classical iterations is, in this framework, attributable to the same-slice feedback $D$, not to the causal part.
- **Sign of $\kappa$.** Nothing here determines whether $\kappa<1$ holds for the implemented network. Heuristically, raising $L_k$ flattens $x\mapsto\mathbb{E}[V_{t_{k+1}}\mid X_{t_{k+1}}=x]$, so $\partial\Phi_k/\partial L_k$ has mixed sign across strikes; a symmetric-part measurement is the only honest way to settle it.
- **Anti-causal remainder.** $g_\gamma$ grows like $\gamma^p$ (Remark 2.1). If the network smooths over many slices ($p$ large) the hypothesis $s_\gamma>0$ may be unsatisfiable for any $\gamma$, in which case this proof gives nothing and a different norm (or a time-localised network) would be needed.
- **Estimator-level Lipschitz constant for the network.** Assumption A for the softplus network trained by Adam is not available; even the existence of a well-defined map $\mu\mapsto R_k(\mu)$ requires a uniqueness-of-minimiser assumption.

## Literature grounding (from the knowledge graph)

- Bayer, Belomestny, Butkovsky, Schoenmakers, *A Reproducing Kernel Hilbert Space approach to singular local stochastic volatility McKean–Vlasov models* (2024, arXiv:2203.01160), chunk 4: "we prove that under suitable conditions, $m^\lambda_A(x;\nu)$ is Lipschitz in both arguments, that is, w.r.t. the standard Euclidean norm in $x$ and the Wasserstein-1-norm in $\nu$"; main results Theorems 2.2 (well-posedness of the regularised MV system) and 2.3 (propagation of chaos). Chunk 3 also records why fixed-basis global regression fails: "Lipschitz constants of the resulting approximation to the conditional expectations in terms of the particle distribution explode as $L\to\infty$, unless the basis functions are carefully chosen."
- Djete, *Non-regular McKean–Vlasov equations and calibration problem in local stochastic volatility models* (2024, arXiv:2208.09986), chunk 0: the dependence on the distribution "is non-regular (i.e., non-Lipschitz/non-continue) for the Wasserstein distance"; chunk 1 on Abergel–Tachet: "by considering a discretized version (in time and space) of the Fokker–Planck equation associated to (1.1), the authors use a fixed point argument for establishing an existence result in short-time."
- Reisinger, Tsianni, *Convergence of the Euler–Maruyama particle scheme for a regularised McKean–Vlasov equation arising from the calibration of local-stochastic volatility models* (2023, arXiv:2302.00434): well-posedness of the regularised equation via Lipschitz coefficients; Proposition 3 (one-step estimate), Proposition 4 (moment stability), Theorem 2 (strong convergence, rate $1/2$).
- Cozma, Mariapragassam, Reisinger, *Calibration of a hybrid local-stochastic volatility stochastic rates model with a control variate particle method* (2021, arXiv:1701.06001), chunk 31: "It is stated but not proved in [38] that the map $\Phi$ is contracting ... Assuming this to be true, $f$ admits a unique fixed point."
- Lacker, Shkolnikov, Zhang, *Inverting the Markovian projection* (2019, doi:10.1214/19-aop1420), chunk 3: Abergel–Tachet "proves the existence in small time ... imposing a restrictive and somewhat implicit smallness assumption"; "very few rigorous results ... have been established so far."

Interpretation (not from the corpus): the weighted-norm argument of Lemma 2 and Remark L2 is the discrete form of the short-time contraction that these existence results use; the one-sided condition of Assumption B is the author's formulation and has no counterpart in the retrieved sources.
