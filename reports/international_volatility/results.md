# International volatility: second fixed wave

Six new comparisons of regional summaries and training-only latent factors against the strong SPX baseline.
Positive gain means lower QLIKE loss. All 72 enumerated past and current contrasts remain in the cumulative family.

| Candidate | Horizon | Development gain | Evaluation gain | Wave Holm p | Cumulative Holm p | Gate |
|---|---:|---:|---:|---:|---:|---|
| regional | 1 | +0.029% | +0.320% | 1.00000 | 1.00000 | DOES_NOT_QUALIFY |
| latent | 1 | -0.234% | +0.632% | 1.00000 | 1.00000 | DOES_NOT_QUALIFY |
| regional | 5 | -0.141% | -0.049% | 1.00000 | 1.00000 | DOES_NOT_QUALIFY |
| latent | 5 | -0.908% | +1.743% | 1.00000 | 1.00000 | DOES_NOT_QUALIFY |
| regional | 21 | -2.425% | -0.177% | 1.00000 | 1.00000 | DOES_NOT_QUALIFY |
| latent | 21 | +0.277% | +7.360% | 1.00000 | 1.00000 | DOES_NOT_QUALIFY |

Passing exploratory leads: []

Foreign features use each market's archived observation calendar, then a backward date join with an explicit freshness bound.
The prior-US-session cutoff limits timing leakage; it does not establish historical publication vintages or remove revisions.
The six comparisons test added international information. They do not test whether latent factors beat regional summaries.
All estimates, nominal uncertainty, MDE, annual slices and nonoverlapping phases are in metrics.json.
Historical reuse remains exploratory. No revised threshold or post-score model selection is permitted.
