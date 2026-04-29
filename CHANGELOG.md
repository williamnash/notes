# Changelog

## 2026-04-29

### Fixed
- **`references.bib`** — repaired four BibTeX syntax errors that prevented the bibliography from compiling, causing every `\cite` in `mathematics.pdf` to be undefined:
  - `zhang`: missing comma after `year` and missing closing `}` (entry was running into `banfi`).
  - `brilliant_backpropagation`: unclosed `"` on the `note` field.
  - `wiki_svm`: missing comma after `title`.
  - `milton`: `@unpublised` → `@unpublished`; `page` → `pages`.
- **`references.bib`** — spelling: `Biancoi` → `Bianconi`, `formulate` → `formulae`, `Micheal` → `Michael`, `Weinhert` → `Weinert`.
- **`mathematics.tex:40`** — `\input{mathematics/vectorCalculus}` → `\input{mathematics/vectorcalculus}` to match the on-disk filename. Worked on macOS (case-insensitive), would have broken on Linux/CI.
- **`mathematics/vectorcalculus.tex:414`** — removed redundant `\label{sphere}` on an `align` environment that duplicated the subsection label one line above.
- **`mathematics/vectorcalculus.tex:611`** — dropped broken `\ref{cylinderlaplace}` cross-reference (the label lives in `physics/electromagnetism.tex`, which compiles as a separate book). Also fixed adjacent "the the" typo.
- **`mathematics/machineLearning.tex:391`** — `\label{forcing}` on the confusion-matrix table was a copy-paste leftover that collided with the real `forcing` label in `vectorcalculus.tex`. Renamed to `confusionmatrix`.
- **`mathematics/algebra.tex:114, 119`** — `cos x` Taylor series: `x^4/41` → `x^4/4!` (factorial typo, in both the series and its derivative).
- **`mathematics/algebra.tex:138–204`** — Tensor product and direct sum 2x2 matrices used `a_{21}` in the upper-right position (and same for `b`); fixed to `a_{12}` / `b_{12}`. Fixed in both the input matrices and the expanded result.
- **`mathematics/algebra.tex:8–10`** — Geometric series text claimed the *finite* sum "converges if `|r|<1`"; that's the *infinite* series condition. Reworded so the finite-sum identity holds for any `r ≠ 1`, with the infinite-series convergence stated separately.
- **`mathematics/algebra.tex:338`** — "Obviously we should have `n>k`" → "We need `n ≥ k ≥ 0`" (the original wording excluded `n=k`, where `\binom{n}{k}=1` is valid).
- **`mathematics/algebra.tex:343, 357`** — "greatest common denominator" → "greatest common divisor" (×4).
- **`mathematics/algebra.tex:336`** — `${{n}\choose{k}}$` → `\binom{n}{k}` (silences the amsmath `\atopwithdelims` warning).
- **`mathematics/statistics.tex:41`** — Population covariance prefactor `1/n` → `1/N`, with `\bar{x}` → `\mu_x`, to match the population variance two equations earlier.
- **`mathematics/statistics.tex:134`** — Poisson derivation header `P(k; \mu)` → `P(k; \lambda)` to match the symbol used throughout the body and the result.
- **`mathematics/statistics.tex:438–465`** — Multiple `${N\choose A}$` → `\binom{N}{A}`.
- **`mathematics/groupTheory.tex:5–8, 60`** — Replaced `\exists` (existential quantifier) with `\in` (set membership) in 6 places where set membership was meant.
- **`mathematics/vectorcalculus.tex:465–624`** — Replaced 20 occurrences of `\over` with `\frac{}{}` across the cylindrical-coordinates and Laplace-equation sections (silences the amsmath `\over` warning).
- **`mathematics/computation.tex` and `mathematics/groupTheory.tex`** — chapters started with `\subsection` instead of `\section`, producing a "bookmark levels differ by more than one" warning. Promoted the appropriate top-level subsections to `\section`.
- **`mathematics/machineLearning.tex:797`** — Replaced missing `\includegraphics{seq2seq1.pdf}` with a placeholder framed box (TODO comment in source) so the build no longer references a missing file. The cross-reference `\ref{fig:encoder-decoder}` continues to resolve.
- **Prose typos**: `it's` → `its` (`algebra.tex:111`); `seperation` → `separation` (`vectorcalculus.tex:533`); `seperability` → `separability` (`vectorcalculus.tex:589`); `seperated` → `separated` (`vectorcalculus.tex:605`); `obey's` → `obeys` (`statistics.tex:128`); `edge's` → `edges` (`statistics.tex:354`); `previous eleent` → `previous element` (`machineLearning.tex` figure caption).

### Verified
- `mathematics.tex` rebuilds cleanly with `pdflatex → bibtex → pdflatex → pdflatex` and emits no `Warning` / `undefined` / `multiply defined` messages in `mathematics.log`.
- `physics.tex` build is currently blocked by a missing `tikz-feynman.sty` package in the local TeX install. That's environmental (pre-existing), not a source-level issue.

### Added
- `CLAUDE.md` with build instructions, layout conventions, and case-sensitivity gotcha.
- `CHANGELOG.md` (this file).
