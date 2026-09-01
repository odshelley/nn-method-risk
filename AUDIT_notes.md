# Humanize-math-prose audit: paper/notes.tex

Date: 2026-09-01. Baseline: 11 pages, 0 warnings, single file (414 lines), rollback branch `humanize-backup`.
Auditors: five parallel category agents. Severity scale as reported: high ~ must-fix, medium ~ should-fix, low ~ judgment-call.
FLAG-ONLY findings are reported for the author and are never applied by the rewrite.

## Lexicon (L) — 0 findings

Clean. Earlier manual scrubs (landscape, epigrams, fragments, tell-words) held up.

## Punctuation (P) — 12 findings

- P1 (low, ~157): "...or learn $\log \E[V \mid X]$; costs nothing in expressiveness..." — break at semicolon, supply subject ("This costs nothing...").
- P2 (low, ~158): "...of \cref{sec:girsanov}; the cross-time strength borrowing..." — full stop.
- P3 (medium, ~173): "...is forced once $\theta$ is chosen: in the $(W, W^\perp)$ coordinates..." — cut at colon into new sentence.
- P4 (low, ~180): "...material but benign; it should enter the effective-sample-size tuning..." — full stop.
- P5 (medium, ~218): "...agree step by step: the exactness of \cref{rem:discrete} is an identity..." — end sentence at "step by step."
- P6 (HIGH, ~253): Update-scheduling paragraph carries two semicolons + two colons across five sentences. Keep at most the "event-triggered refresh: monitor" colon; sentence breaks elsewhere.
- P7 (medium, ~257): "...function class grows; a short lemma ... should be written." — full stop before "A short lemma".
- P8 (HIGH, ~283): "For intraday recalibration, amortise: train offline ... ; online, evaluate ..." — split into three sentences; thin the paragraph's other joints.
- P9 (medium, ~301): colon-led fragments as a template across the six particle-method items ("Not regression at all:", "Caveat in print:", ...) — the colon form of a bold-lead list; convert most to sentences.
- P10 (HIGH, ~310): "Deterministic and accurate in two dimensions; hits the curse...; the thin-density wings ... need care." — three subjectless fragments on two semicolons; rewrite as sentences.
- P11 (low, ~327): "...confuse with ours; distinguish it early." — full stop.
- P12 (low, ~348): "House style forbids theorem environments; \cref{rem:cost,rem:selfnorm} become prose." — comma with "so".
- Clean: no em dashes anywhere; en dashes all correct; no bold misuse.

## Rhythm (R) — 5 findings

- R1 (medium, ~84): "No kernel, no bandwidth selection, no density estimation." — verbless fragment + triad without three distinct loads; fold into preceding sentence.
- R2 (low, ~275): "Stackable with the other levers." — subjectless fragment; make a sentence.
- R3 (medium, ~279): "The engineering trick and the hypothesis of the local convergence theorem are the same statement." — null restatement of the two preceding sentences; delete.
- R4 (medium, ~283): amortisation paragraph: four of five sentences run 35-48 words with the same colon/semicolon pivot shape; break at least two, drop one pivot. (Overlaps P8; apply together.)
- R5 (judgment-call, ~338): "Structure, one figure per claim." — telegraphic outline stem; exemption likely applies; leave unless outline is rewritten as prose.
- Clean: no connective runs, no hedging templates, no hedge clauses.

## Structure (S) — 8 findings

- S1 (low, ~249): "This is a large constant-factor reduction in training cost." — "large" unsupported pending experiments; drop quantifier or hedge. (Note: exp_d results now exist in PAPER_PLAN and could support it, but the notes text does not cite them.)
- S2 (low, ~253): Update-scheduling subsection is one ~175-word paragraph doing five moves; split at "A caution:". (Pairs with P6.)
- S3 (MEDIUM, ~257): "This answers the objection..." while the supporting lemma is admitted unwritten, and Bayer et al.'s constants are proved for a fixed hand-chosen RKHS, not learned features. Move to conditional; soften "recovers exactly the regularised structure".
- S4 (MEDIUM, ~283): "The convergence analysis quantifies the claim..." presupposes an analysis the document elsewhere says is unproved and reserved for the maths paper. Recast conditionally ("If the damped iteration is shown to contract locally at rate r...").
- S5 (low, ~283): break amortisation paragraph before "A riskier variant...".
- S6 (low, ~287): "the contraction factor bounds..." — same presupposition in the deployment claim; condition it ("once established, would bound").
- S7 (low, ~310): split PDE-methods paragraph at "Two details matter for us."
- S8 (MEDIUM, ~331): "Kernel methods genuinely fail, and networks do not, in multi-asset LSV" — asserts as fact what the same sentence concedes is conditional; make conditional and experiment-gated.
- Clean: headings, section openers, roadmaps, bullet usage.

## Hygiene (H) — 25 findings (most FLAG-ONLY)

Fixable by rewrite:
- H4 (medium, ~159): "$\sigDup$ requires an arbitrage-free..." — sentence starts with a symbol; prepend "The Dupire surface".
- H5 (medium, ~160): "$\Lev$ is defined on grid times" — prepend "The leverage".
- H11 (medium, ~221): "For deterministic $\eta$, $\ln Z$ is Gaussian..." — formula collision; interpose "the log-density".
- H13 (medium, ~226): "For any measurable $f$" — "any" ambiguous; use "every".

FLAG-ONLY (author decisions; notation and citations):
- H6 (HIGH, ~169): $\theta$ means both network parameters ($f_\theta$, $\theta_k$) and the Girsanov tilt ($\theta_t$); both time-subscripted. Rename one family (suggest tilt -> $\vartheta$ or $u$).
- H12 (HIGH, ~226): $Z$ contradiction. Line ~200 defines $Z_t = d\Qm/d\Pm$; the weighted-regression subsection redefines $Z = d\Pm/d\Qm$. Genuine error: pick one convention (e.g. weight $\zeta := Z^{-1}$) and rewrite eq:weightedreg accordingly.
- H3 (medium, ~112): Algorithm 1 uses $\Qm$, $\eta_k$, $\Delta\widetilde{B}^\perp$ before sec:girsanov defines them; add forward pointer or move tilt lines.
- H7 (medium, ~185): density derivation announces no ingredients (constraint, Levy characterisation (used silently), Girsanov, Novikov); add an opening toolbox sentence; confirm Levy characterisation is the intended justification.
- H8 (medium, ~210): $W^Q$ appears once, undefined; define or switch to tilde convention.
- H9 (low, ~218): $N$ = particle count and normal law on the same line; use $\mathcal{N}$ for the Gaussian.
- H10 (low, ~220): rem:indep is a formal claim used later as a fact; consider promoting to a lemma (house-style caveat noted).
- H15 (medium, ~263): $\varphi$ = Gaussian density and feature map; rename features (e.g. $\psi$).
- H16 (low, ~266): $W$ = Brownian driver and weight matrix; rename matrix.
- H17 (low, ~267): $V$ silently = response vector in the ridge system; introduce $\mathbf{v}$.
- H18 (low, ~298): $S_0$/$S_t$ vs $X$ notation switch in own prose; unify or state retention of cited papers' notation.
- H23 (medium, ~319): $L$ = leverage and number of basis functions ("as $L \to \infty$"); rename count.
- H1 (low, preamble): \Phal and \Id macros defined, never used; delete or use in eq:damped.
- H2 (low, ~66): Gyongy citation lacks theorem locator.
- H14 (medium, ~257): Bayer stability claim lacks theorem number.
- H19 (medium, ~300): four Bayer results on one bare citation; add theorem numbers to rate + propagation-of-chaos claims.
- H20 (medium, ~310): "stated but not proved" quotation lacks page locator (prior-art report has: Appendix E.3, p. 43).
- H21 (medium, ~310): "$Q = 2$" uncited in-sentence (source: wyns2017adjoint Sec. 7).
- H22 (low, ~318): "twice" without page numbers (prior-art report has: pp. 24 and 39).
- H24 (low, ~320): Hakala quotations lack page locators.
- H25 (low, ~327): Cuchiero quotation lacks page locator.
- Clean: display punctuation (all 15), no clearly/obviously, no symbols-for-words in prose, no theorem-env commentary.

## Totals

50 findings: 3 P-high + 3 S-medium-overclaims + 2 H-high notation errors are the load-bearing ones. 21 of 25 H findings are flag-only.
