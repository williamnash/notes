# Repo

Personal LaTeX notes (since 2017). Two books compile from the root:

- `mathematics.tex` → `mathematics.pdf` (algebra, statistics, computation, machine learning, vector calculus, group theory)
- `physics.tex` → `physics.pdf` (classical mechanics, statistical mechanics, quantum mechanics, electromagnetism, particle physics theory + practice)

Each book `\input`s chapter files from `mathematics/` or `physics/`. Shared preamble lives in `includes.tex`. Bibliography is `references.bib` (BibTeX, `\bibliographystyle{alpha}`).

# Build

```bash
pdflatex mathematics.tex && bibtex mathematics && pdflatex mathematics.tex && pdflatex mathematics.tex
pdflatex physics.tex     && bibtex physics     && pdflatex physics.tex     && pdflatex physics.tex
```

The bibtex pass + two extra pdflatex passes are required to resolve `\cite` / `\ref` cross-references — a single `pdflatex` run will leave "undefined references" warnings.

# Layout & conventions

- Filenames in `mathematics/` and `physics/` are mixed case (`machineLearning.tex`, `vectorcalculus.tex`, `groupTheory.tex`). macOS APFS is case-insensitive by default, so mismatches between `\input{...}` and the on-disk name silently work locally but **break on Linux/CI**. When adding or renaming inputs, match casing exactly.
- Custom macros defined in `includes.tex`: `\ffrac`, `\E`, `\Var`, `\Cov`, `\B{...}`, `\argmax`, `\argmin`. Prefer these to ad-hoc `\mathrm{...}`.
- Figures live in `mathematics/fig/` and `physics/images/`. Reference them with the full path from the repo root (e.g. `\includegraphics{mathematics/fig/svm.png}`) since `pdflatex` is run from the root.
- Code listings use the `mystyle` `lstdefinestyle` from `includes.tex`.
- `physics.tex` adds `tikz`, `tikz-feynman`, and `enumitem` on top of `includes.tex` (mathematics doesn't need them).

# Don't commit

`.gitignore` currently lists only `*.pdf` and `*.out`, but pdflatex also produces `.aux`, `.log`, `.toc`, `.bbl`, `.blg`, `.synctex.gz`, plus macOS `.DS_Store`. Don't stage those — extend `.gitignore` if it comes up.

# Working on prose

- Notes are personal: terse, equation-heavy, occasionally informal. Don't rewrite for a textbook tone unless asked.
- When fixing math, preserve the surrounding pedagogical flow — the goal is correctness, not restructuring.
- Cross-file label collisions are easy to introduce; check `\label{...}` is unique across all chapters when adding one.
