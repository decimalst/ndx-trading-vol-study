# Neural context warm-up clarification before the first fit

Recorded 2026-09-06. The first empirical neural attempt stopped in training-row
eligibility validation, before a model was instantiated or fitted. At the
requested initial origin 2013-02-01, only 496 eligible 22-session windows had
both target horizons completed. The fixed requirement is at least 500.

Checking chronology and availability only gave:

| Initial diagnostic origin | Complete training windows | Status |
|---|---:|---|
| 2013-02-01 | 496 | Insufficient warm-up |
| 2013-02-04 | 497 | Insufficient warm-up |
| 2013-02-05 | 498 | Insufficient warm-up |
| 2013-02-06 | 499 | Insufficient warm-up |
| 2013-02-07 | 500 | First eligible neural fit |

Interpret the initial partial-2013 fit as the first origin satisfying the
complete neural context and minimum-training requirement: **2013-02-07**.
The four earlier diagnostic origins receive explicit insufficient-warm-up
records. Starting in 2014, refit at the first eligible origin of each year as
specified. Retain the 500-window minimum, 1,500-window cap, every architectural
and optimization setting, and all 2016-01-04 through 2025-10-10 scored origins.

No neural empirical training loss, forecast or evaluation score existed when
this clarification was made. No model outcome was used to choose the date.
The shared reference YAML remains byte-for-byte unchanged; this separately
hashed clarification records its initial warm-up interpretation. The initial
prefit manifest is preserved as `neural_manifest_initial_insufficient.json`,
and the excluded diagnostic rows are saved in `neural_warmup_exclusions.json`.
