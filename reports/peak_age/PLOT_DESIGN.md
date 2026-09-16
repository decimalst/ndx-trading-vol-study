# Verified peak-age comparison figure

The new generated test suite precedes the plot implementation. It reads no real
experiment metrics. The figure contains all six registered control/phase
effects: peak_age against baseline, depth and training mean, in development
and evaluation, for the sole21-session return target.

Horizontal coordinates are100 times relative MSE improvement. Positive values
indicate smaller candidate MSE; the dashed effect threshold is +0.25percent.
Each nominal paired-loss confidence envelope[lo,hi] becomes
[-100*hi/control_MSE,-100*lo/control_MSE]. Both phases use their own control MSE.
The title remains descriptive; all statistical gates and reused-history limits
remain explicit. No partial-control or single-period headline is rendered.

Only a VERIFIED proof binding exact metric bytes and exact new-manifest bytes
can supply a figure. The new manifest must bind the still-present regular
old wave20 failure/metrics/verification trio. Current and successful
index-hinge/wave18/wave19/wave21 failure markers, including dangling links,
block the figure. Absence and hashes are rechecked after source-record reading.
Missing/duplicate/foreign contrasts, incomplete phases, invalid losses,
inconsistent effect units, unordered/nonfinite intervals and failed trials
raise instead of silently hiding a comparison.

`load_verified(root)` returns checked metrics; `draw(metrics)` returns a
Matplotlib figure; `main(root=None)` writes comparison_intervals.png and .pdf.
Generated tests verify all six plotted effects, interval conversion, the
positive0.25percent threshold, order independence, output usability, exact
input preservation and late-marker detection. The figure is rendered and
visually inspected on generated data before any real verified result is used.

Generated validation completed:11 plot contracts pass after missing-module RED;
the separate22 publication contracts pass including the new existing-sentinel
symlink test. The six-effect synthetic PNG was visually inspected: all three
control labels, both period colors, six markers, interval segments, percent
axis, positive0.25percent threshold, legend and limitations are visible without
clipping. This is layout QA on generated values, not an experiment figure or
historical result. PNG and PDF export usability is covered by the tests.
