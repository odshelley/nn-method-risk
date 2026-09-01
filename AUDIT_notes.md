# Humanize audit — paper/notes.tex (pass 2, 2026-09-01)

Baseline: commit d4c775b, branch `humanize-backup-2`, 14 pages, 0 reference warnings.
57 distinct findings after dedup (L 3, P 31, R 4 after merges, S 2 after merges, H 16, plus 1 merged must-fix).
Previous pass's flag-only items (theta overload, Z convention, N/phi/W/V/L overloads, Algorithm 1 forward refs) remain open and are NOT re-listed.

## Merged must-fix (S3 = R3 = H9, line 280)

- **M1** | must-fix | garbled clause: "The variance is not destroyed but relocated, since at any $x$ the tilt starves the same computation shows $w > 1$ and an inflated variance." → "The variance is not destroyed but relocated, since at every $x$ that the tilt starves, the same computation shows $w > 1$ and an inflated variance." (restores the relative clause, swaps any→every; satisfies structure, rhythm, hygiene simultaneously)

## Lexicon (3)

- **L1** | should-fix | 280 | "\Cref{eq:varexp} rewards a careful reading." → delete; fold ref into next sentence: "Every quantity in \cref{eq:varexp} lives under the original measure $\Pm$, ..."
- **L2** | should-fix | 231 | "the tilt moves where the samples live, the weights fix the estimand" → delete epigram, end sentence at "...with no further correction." (CONFLICTS with P8, which keeps the clauses; L2 wins if both categories approved)
- **L3** | judgment-call | 221 | "cleanest possible shape" → "best case"

## Punctuation (31; colon/semicolon thinning only — no em dashes or bold anywhere)

- **P1** | judgment | 79 | "projection identity: over all" → "projection identity. Over all"
- **P2** | should-fix | 118 | "Structural observation to state early in any write-up: the explicit scheme..." → "One structural observation belongs early in any write-up. The explicit scheme..." (SUPERSEDES R1: fixes fragment and colon at once)
- **P3** | should-fix | 118 | "explicit scheme; it bites" → "explicit scheme. It bites"
- **P4** | should-fix | 152 | "subtle reason: shared parameters" → "subtle reason. Shared parameters"
- **P5** | should-fix | 189 | "$= 0$: the pair" → "$= 0$, so the pair"
- **P6** | should-fix | 207 | "pathwise unchanged; the stock driver" → "pathwise unchanged. The stock driver"
- **P7** | should-fix | 221 | "$V$ path: the weights live" → "$V$ path, so the weights live"
- **P8** | should-fix | 231 | "correction: the tilt moves..., the weights fix..." → sentence break + "and" (see L2 conflict)
- **P9** | should-fix | 234 | "cancels; only the conditional" → "cancels, and only the conditional"
- **P10** | should-fix | 238 | "is needed: draw" → "is needed. Draw"
- **P11** | should-fix | 251 | "admissible family; this is" → "admissible family. This is"
- **P12** | should-fix | 280 | "is tiny; this is" → "is tiny. This is"
- **P13** | should-fix | 287 | "path-dependent weight; \cref{rem:optscope} records" → ", and \cref{rem:optscope} records"
- **P14** | should-fix | 328 | "refresh: monitor" → "refresh. Monitor"
- **P15** | should-fix | 332 | "feature extractor; per slice, solve" → "feature extractor. Per slice, solve"
- **P16** | should-fix | 332 | "theoretical payoff: a linear-in-features" → "theoretical payoff. A linear-in-features"
- **P17** | should-fix | 334 | "differ once regularised: penalising" → "They differ once regularised. Penalising"
- **P18** | should-fix | 334 | "right prior here: the target moves" → "right prior here, because the target moves"
- **P19** | should-fix | 334 | "right-hand side); under a positivity layer" → "right-hand side). Under a positivity layer"
- **P20** | should-fix | 350 | "Orthogonal to everything above: conditioning" → comma (SUPERSEDED by R4 fix if rhythm approved)
- **P21** | should-fix | 354 | "implicit one: distil" → "implicit one. Distil"
- **P22** | should-fix | 354 | "supplies one: its output" → "supplies one, since its output"
- **P23** | judgment | 364 | "combines the schemes: a full implicit solve" → ", with a full implicit solve"
- **P24** | should-fix | 379 | "regression problem; PURBF" → "regression problem. PURBF"
- **P25** | should-fix | 387 | "\citep{ren2007}; the inner" → ". The inner"
- **P26** | should-fix | 389 | "literature: it is asserted" → "literature. It is asserted"
- **P27** | should-fix | 393 | "literature; what has never" → "literature. What has never"
- **P28** | should-fix | 398 | "in print: our theory" → "in print. Our theory"
- **P29** | should-fix | 406 | "training; they call" → "training. They call"
- **P30** | should-fix | 406 | "axes entirely: the former" → "axes entirely. The former"
- **P31** | judgment | 422 | "weight recursion; applies unchanged... sample size; the central figure." → relative clause + comma

## Rhythm (4 after merges; R1 superseded by P2, R3 merged into M1)

- **R2** | should-fix | 158 | three-sentence metronome run "The classical particle method... The principal remedy... The cross-time..." → merge last two: "The remedy here is the Girsanov importance sampling of \cref{sec:girsanov}, helped by the cross-time strength borrowing of the implicit scheme."
- **R4** | judgment | 350 | fragment colon-label → "Orthogonal to everything above, conditioning estimators in the style of \citet{muguruza2019} and control variates in the style of \citet{cozma2019} reduce..."
- **R5** | should-fix | 354 | "The explicit scheme is thus the initialiser that plausibly places the damped iteration inside its convergence basin." → delete (punch line restating the paragraph)
- **R6** | judgment | 378 | "No bandwidth, guaranteed variance reduction by Rao--Blackwellisation." → "There is no bandwidth, and Rao--Blackwellisation guarantees the variance reduction."

## Structure (2 after merges; S3 merged into M1)

- **S1** | should-fix | 247 | "This subsection derives the criterion..." self-announcement → fold scoping into next sentence: "The tilt $\theta$ has so far been a free input. Fixing a single time slice $t$, it helps to first recall..."
- **S2** | judgment | 173 | split the single 200-word paragraph of subsection 6.1 at "The constraint \cref{eq:constraint} is one linear equation..." (no wording changes)

## Hygiene (16; notation items are FLAG-ONLY, author decision)

- **H1** | must-fix, FLAG-ONLY | 249 | $\Phi$ payoff collides with the self-consistency map $\Phimap = \Phi$ of \cref{eq:phimap} → rename payoff to $\Psi$
- **H2** | should-fix, FLAG-ONLY | 326 | "every $m$-th slice" collides with target $m(x)$ → e.g. "$\kappa$-th"
- **H3** | should-fix, FLAG-ONLY | 332 | "$O(Np + p^3)$ for $p$ features" collides with density $p$ → feature count $d$ (also in alg:ridgehead)
- **H4** | should-fix, FLAG-ONLY | 262 | kernel $K$/$K_b$/$R(K)$ collides with grid size $K$ (line 88) → $\mathcal{K}$
- **H5** | should-fix, FLAG-ONLY | 210/255 | $dW^Q_t$ italic $Q$ vs $\Qm$ (and $Q=2$ at line 387) → $dW^{\Qm}_t$
- **H6** | should-fix, FLAG-ONLY | 249 | $\sigma$ in $\sigma^{\top}\nabla_x \log h$ undefined → name it ("the diffusion coefficient of $X$")
- **H7** | should-fix, FLAG-ONLY | 303 | $q^*(t,x)$ vs single-slice convention → $q^*(x)$
- **H8** | judgment, FLAG-ONLY | 282 | $\omega$ orphan/near-identical to $w$ → drop it ("suppose $w$ depends on the path only through $X_t$")
- **H10** | should-fix | 289 | "at any $x$ with appreciable mass" → "at every $x$"
- **H11** | should-fix | 296 | "For any density $q$" → "For every density $q$"
- **H12** | should-fix | 358 | "from warm-start error $\delta_0$ of order $\log(\mathrm{tol}/\delta_0)/\log r$ corrections suffice" → "from a warm-start error $\delta_0$, a number of corrections of order $\log(\mathrm{tol}/\delta_0)/\log r$ suffices"
- **H13** | should-fix, FLAG-ONLY | 251 | Owen--Zhou variance-floor claim needs a precise locator
- **H14** | should-fix, FLAG-ONLY | 312 | balance-heuristic optimality needs the Veach--Guibas theorem number
- **H15** | should-fix, FLAG-ONLY | 332 | Bayer et al. stability theorem number missing
- **H16** | judgment, FLAG-ONLY | 285 | $\asymp$ in \cref{eq:designformula} vs $\approx$ in the chain it derives from → $\approx$
