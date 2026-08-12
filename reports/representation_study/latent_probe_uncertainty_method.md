# Latent-probe control clustering amendment

Frozen before any aggregate latent-probe result was produced. A first definitive
run was interrupted after annual fold 2016 when review identified that the
planned selectivity interval deleted actual-label episodes from the surrogate
scoreboards as well. Those fold files were preserved as provisional artifacts;
no aggregate metric or verdict existed and no score was inspected to choose this
repair.

The point estimator remains exactly the registered quantity: actual-label
five-phase mean AUC (or lift) minus the median of ten identical-probe,
first-order-Markov control scores. The uncertainty calculation is clarified as
follows:

1. Actual positive origins retain the registered episode identifier: the first
   future threshold-exceedance trigger, with triggers no more than five market
   sessions apart grouped together.
2. A surrogate has no observed future trigger. Its structural proxy attaches a
   positive control label to that origin session and groups positive origins no
   more than five source sessions apart. Episodes are constructed separately for
   every seed and annual fold. They never bridge annual folds because each
   training-estimated Markov path resets at the fold boundary.
3. Actual and control episode uncertainty are separate block-jackknife
   components. The actual component deletes each actual episode while holding
   all control scores fixed. For each of the ten seeds, its control component
   deletes each of that seed's surrogate episodes, recomputes that control
   metric and the ten-control median, and holds the actual score and other nine
   controls fixed.
4. The component jackknife variances are summed and the reported 95% interval is
   the full-sample selectivity plus or minus 1.96 times the square root of that
   sum. The report includes actual and per-seed control episode counts and the
   component variances.

This treats the effective sample as transition episodes on both sides of the
selectivity contrast. It does not change labels, representations, probes,
seeds, annual fits, point scores, evidence thresholds, or the fixed five-phase
scoreboard.
