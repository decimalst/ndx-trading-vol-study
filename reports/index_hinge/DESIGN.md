# Registered medium-horizon SPX return study

This sixth wave asks whether a single asymmetric relationship between implied and trailing realized risk adds predictive value for 21- and 63-session SPX price returns. The design was specified before fitting or examining any new predictive relationship. Earlier experiments, their thresholds, and their results remain frozen.

## Identifiable comparison

The raw implied-minus-realized gap adds no predictor span to a linear model already containing its components. Its addition to ridge could change penalty geometry without adding information. This experiment therefore adds a fixed hinge of the gap, centered on the precise admitted training rows at each monthly horizon-specific fit. The baseline contains both component levels and their separately centered squares, four return histories, daily and weekly realized-risk histories, VIX9D/VIX, VVIX, and weekday indicators. The same-row historical mean is a second required control.

This is an interaction within existing source information. It is not a new data source or a claim that implied and daily OHLC variance measure the same contract. The monthly/quarterly return question is motivated by research on implied and realized variation, which explicitly emphasizes accurate high-frequency realized measures. The daily OHLC proxy here differs materially. [Bollerslev, Tauchen and Zhou, 2009](https://public.econ.duke.edu/~get/wpapers/btz.pdf).

## Calendar and information

All numerical features end at the preceding observed SPX session. The target is the raw-close cumulative log price return from entry close to the close 21 or 63 observed sessions later. A training label is admitted only after that ending close, no later than the previous session of the monthly fit. The first complete feature row of the month determines the fit, independently of its subsequent outcome availability. Training and scoring rows are common across all three models within a horizon; the horizons need not have the same rows.

Development uses 2016–2019 origins whose target ends by 2019-12-31. Evaluation starts 2020-01-02 and requires complete targets by 2025-10-20. The protected period beginning 2025-11-03 remains excluded. Source histories are current archival vintages, including back-calculated early VIX9D data; these dates and lags do not certify historical availability or revisions. The SPX calendar contains observed vendor rows and is not a certified complete historical exchange calendar.

## Four comparisons and strict gates

At each horizon the hinge model must improve on both the component baseline and historical mean. All four comparisons remain in wave Holm correction at 1/840 and cumulative correction over 103 enumerated contrasts at 0.05. This enumerated family is not a claim to count every historical experiment in the repository.

Each comparison needs at least 0.25% lower MSE in both fixed periods, a negative paired MSE gap in both evaluation slices, and improvement in every nonoverlapping offset in both periods. Offsets are anchored to position modulo horizon on the original bounded SPX calendar, before complete-row filtering. None can be omitted or selected for a favorable result.

Inference takes the maximum two-sided probability from centered-null circular bootstrap blocks of 126, 252, and 504 retained observations and Bartlett HAC with 504 lags, then the maximum across development and evaluation. The 99,999 bootstrap draws resolve below one tenth of the strictest wave correction threshold. At least 505 observations are required in each phase, preserving the literal HAC bandwidth. These choices acknowledge dependence; four years still provide few independent quarters and low power.

## Validation and failure accounting

Tests were written before their new implementation modules existed. Synthetic checks cover source cutoffs, strict missing windows, raw price targets, target maturity, future mutation, train-only transforms, normalized ridge, independent numerical solutions, overlapping outcomes, all-offset gates, and publication faults. An independent verifier reconstructs inputs, labels, monthly fits, every forecast, both controls, every uncertainty calculation, cumulative corrections, and ledger events.

The pre-fit review corrected two bookkeeping issues in new code: inherited measurement contrasts now retain their measurement label and source-row identity; and a phase of exactly 504 observations is rejected instead of allowing a helper to truncate the declared HAC lag count. A fault fixture also identified a fragile source-path fallback, repaired before fitting. No threshold or empirical result prompted these changes.

A manifest pins this protocol, all source and test Python files, the exact input bytes, and previous protocol/report artifacts before fitting. Registered failures retain all four hypotheses with probability one and no lead. Partial scored artifacts remain diagnostic only. Results require independent reconstruction before interpretation.

## Scope of any finding

SPX price returns exclude reinvested dividends and do not subtract a risk-free rate. No executable fund return, trade sizing, overlapping-position capital, costs, or strategy profitability is asserted. Any passing result would remain exploratory because this history has been repeatedly studied and source-vintage limitations remain.
