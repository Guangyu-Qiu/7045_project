# Leave-one-out validation for generalized linear models

This midterm project provides reusable R functions for leave-one-out prediction with generalized linear models.

## Files

- `R/main.R` contains the implementation, simulated examples, S3 methods and verification checks. `run_loo()` uses one worker by default. Set `workers` to a larger positive integer to enable parallel execution.
- `R/simulation.Rmd` contains my original pseudocode.
- `report/7045_midterm_loo.pdf` and `report/presentation.html` are the materials used for the project presentation. Their sources are `report/loo_background.tex` and `report/presentation.qmd`, respectively.
- `report/report.qmd` is the midterm report. It sources `R/main.R` and renders to `report/report.pdf`.