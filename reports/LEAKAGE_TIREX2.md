# Leakage assessment — TiRex-2 (arXiv 2607.01204v1, `NX-AI/TiRex-2`)

Written 2026-08-11, before running TiRex-2 on the clean window.

## Why this needs its own assessment

The project's leakage rule is date-based: *the checkpoint was published on date
X, so anything before X may be in or adjacent to its training corpus.* That rule
exists because Chronos-2's training corpus is not fully disclosed — with no
corpus listing, publication date is the only conservative boundary available.

TiRex-2 is published **2026-07-01**, which is *after* `clean_start`
(2025-11-03). Applied literally, the date rule would shrink the usable window to
**28 scored origins**. That is not a usable sample.

TiRex-2 differs in one respect that matters: **its corpus is fully enumerated**
(Appendix E, Tables 4–7). We can replace a date heuristic with a direct check.

## What is actually in the corpus

Three univariate components, inherited from TiRex / TiRex-1.1:

1. **Chronos training collection, ~30M series** — the public collection from
   Ansari et al. (Chronos-1), i.e. `autogluon/chronos_datasets` and
   `chronos_datasets_extra`. Note this is Chronos-**1**'s disclosed training
   set, not Chronos-2's undisclosed corpus.
2. **Synthetic Gaussian-process series, ~15M** — KernelSynth-style, wholly
   synthetic.
3. **GIFT-Eval pretraining subset, ~2.5M.**

Plus `Salesforce/lotsa_data`, `dysts` chaotic trajectories (1%), `boom`
(observability metrics, leakage-filtered), and `hydrology`. Multivariate
training samples are synthesised on the fly by coupling univariate series — no
additional real multivariate corpus is introduced.

### Financial content

Everything financial in the enumerated tables:

| Dataset | What it is |
|---|---|
| `chronos_datasets/exchange_rate` | daily FX rates, 8 currencies |
| `lotsa_data/bitcoin_with_missing` | Monash bitcoin series |
| `chronos_datasets/monash_fred_md` | FRED macro monthly (GIFT-Eval ckpt only) |
| `lotsa_data/largest_2017…2021` | Spanish electricity market prices |
| `chronos_datasets/dominick` | retail scanner sales |

**Absent: any equity index, QQQ or NDX series, any realized-variance series, any
implied-volatility index (VIX/VXN), and any options data.** The target of this
experiment and its market benchmark are both unrepresented.

### Time coverage

Every real component is a frozen public benchmark archive with a hard end date
well before the clean window — `cmip6` 1850–2010, `era5` 1991–2018, `largest`
2017–2021, the M1/M3/M4 and Monash collections mostly pre-2020, with LOTSA and
`chronos_datasets` themselves assembled in 2024. The **effective data horizon of
the real corpus is roughly 2023**, not the 2026-07-01 publication date.

## Conclusion and how results are reported

For NDX realized variance over 2025-11 → 2026-08, the corpus contains neither
the asset class nor the period. The leakage risk is materially lower than
Chronos-2's, and for a better reason: it is checked rather than assumed.

Two caveats, both stated plainly rather than resolved:

1. The paper documents two *benchmark* checkpoints (`TiRex-2-fev`,
   `TiRex-2-GIFT-Eval`) built by removing benchmark-overlapping datasets. It
   does not state which corpus the public `NX-AI/TiRex-2` checkpoint was trained
   on. That it is the full union is an inference, not a stated fact.
2. "No equity data in the corpus" is not "no equity information in the weights."
   Cross-domain transfer from an FX or bitcoin series to an index-vol series
   cannot be ruled out by a dataset listing. It is, however, a far weaker
   channel than direct memorisation.

**Therefore:** TiRex-2 results are reported on the **full clean window (192
origins)**, with the strict post-publication subwindow (2026-07-01 →, **28
origins**) as a robustness check. If the two disagree in sign, believe neither
and keep accruing. Nothing about `clean_start` in `config.yaml` changes.

**2026-08-12: the sign-disagreement rule fired.** Full window, `har_iv` (0.3167)
beats `tirex_cov_ivf` (0.3296); post-publication subwindow, `tirex_cov_ivf`
(0.1882) beats `har_iv` (0.1991). Per the rule above, neither ranking is
believed; the comparison stays open and accrues. (The across-the-board QLIKE
drop on those 28 days is a calm-regime level effect — only the ranking is
readable there, and it flips.) Note also `tirex_cov_ivf` fails the 80% interval
gate on the full window (coverage 0.740, p_uc = 0.043, uniformly overconfident) —
see FINDINGS.

## Unrelated constraint: the quantile grid

TiRex-2 emits exactly nine deciles (0.1 … 0.9; `model.quantiles`) and accepts no
quantile-level argument. It cannot produce the 0.05/0.95 levels the
pre-registered grid specifies, and `mean_var` — the truncated-mean estimator
behind QLIKE — is only comparable across models sharing a grid.

So the whole model set is recomputed on the decile grid into a parallel report
(`results_clean_dec.md`, via `--quantile-grid deciles`). `config.yaml` is not
amended and the pre-registered results stand untouched. On that grid the widest
available interval is **80%, not 90%**, so the interval gate there is the
decile-grid analogue of the registered gate, not the registered gate itself.
