"""Post-result presentation of authenticated, saved replay outputs; no fits/tests."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = ROOT / "data/model_memory_study/copula_spread_replay"
terminal = json.loads((OUT / "terminal.json").read_text())
for name, expected in terminal["output_hashes"].items():
    actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"Saved output changed: {name}")

summary = pd.read_parquet(DATA / "summary.parquet")
safety = pd.read_parquet(DATA / "safety_summary.parquet")
positions = pd.read_parquet(DATA / "positions.parquet")
density = pd.DataFrame(json.loads((OUT / "density_descriptions.json").read_text())["rows"])
labels = {
    "always_sell": "Always sell", "orig_gaussian": "Original Gaussian",
    "orig_t8": "Original t8", "cal_gaussian": "Affine Gaussian",
    "cal_t8": "Affine t8", "shape_gaussian": "Shape Gaussian", "shape_t8": "Shape t8",
}
models = list(labels)
structures = ["put", "call", "condor"]

def write_new(name, value):
    with (OUT / name).open("x") as stream:
        stream.write(value)

def table(headers, rows):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *("| " + " | ".join(map(str, row)) + " |" for row in rows),
    ])

def cases(phase, distance=.02):
    return summary[(summary.phase == phase) & (summary.distance == distance)
                   & (summary.credit_fraction == .10)].set_index(["model", "structure"])

def policy_table(phase, distance=.02):
    selected = cases(phase, distance)
    rows = []
    for model in models:
        values = []
        for structure in structures:
            row = selected.loc[(model, structure)]
            values.append(f"{100*row.coverage:.2f} / {100*row.sold_any_breach_rate:.2f} / {100*row.sold_mean_debit:.3f}")
        rows.append([labels[model], *values])
    return table(["Policy", "Put", "Call", "Iron condor"], rows)

primary = cases("evaluation")
quick = []
full = []
credit = []
for structure in structures:
    always = primary.loc[("always_sell", structure)]
    old = primary.loc[("orig_t8", structure)]
    new = primary.loc[("shape_t8", structure)]
    quick.append([
        structure.title(), f"{int(new.sold_sessions):,} / {int(new.sessions):,}",
        f"{100*always.sold_any_breach_rate:.2f}%", f"{100*old.sold_any_breach_rate:.2f}%",
        f"{100*new.sold_any_breach_rate:.2f}%", f"{100*new.coverage:.1f}%",
    ])
    full.append([
        structure.title(),
        *[f"{round(primary.loc[(model,structure)].sold_any_full_loss_rate*primary.loc[(model,structure)].sold_sessions)} / {round(primary.loc[(model,structure)].sold_both_full_loss_rate*primary.loc[(model,structure)].sold_sessions)}"
          for model in ["always_sell", "orig_t8", "shape_t8"]],
    ])
    credit.append([
        structure.title(), f"{100*(always.sold_mean_debit+.02):.3f}%",
        f"{100*(old.sold_mean_debit+.02):.3f}%", f"{100*(new.sold_mean_debit+.02):.3f}%",
    ])

main = """# Same-day spread safety: completed descriptive replay

Prepared 2026-09-13. Status: **COMPLETED_VERIFIED_DESCRIPTIVE_PROBABILITY_REPLAY**.

At the primary 2% strike distance, the fixed probability filters selected sessions with fewer expiration strike breaches and lower average expiration liabilities than selling every day. **The new shape calibration did not consistently improve on the older filters.** This is evidence worth carrying into a prospective options benchmark, with no new model promoted and no claim of executable option profit.

The shape correction improved descriptive marginal density scores, but the registered wave29 attempt failed a numerical tail-quantile check. Its **UNEVALUABLE verdict and six comparisons recorded as p=1 remain unchanged**. The separately registered replay uses the already saved breach probabilities and the unchanged trading rules. It does not recover statistical significance from the failed attempt.

## What was tested

- QQQ and SPX, equally allocated, using their saved next-session open-to-close log returns. These are two underlying instruments; SPX is not SPY. Features retain the inherited extra lag, ending on the reference session preceding the forecast origin.
- Hypothetical same-day put credit spreads, call credit spreads and iron condors. Primary short strikes are **2% from the session open**, with **1% of open as wing width**. Fixed 1% and 3% strike distances are reported as sensitivities, without selecting a winner.
- Each model sells both asset spreads only if its forecast probability that **at least one underlying breaches a short strike at expiration is at most 10%**. Six models and an always-sell control are retained.
- Aggregate gross-width reserve is **2% of current equity**, split 1% per asset. Position size does not increase with assumed premium. Each structure/distance/credit case is a separate account.
- Development has **733 sessions, target dates 2017-02-02 through 2019-12-31** after warmup. Evaluation has **1,457 sessions, target dates 2020-01-03 through 2025-10-20**. The history was previously inspected and uses current-vintage source data; neither phase is a new untouched prospective test.

The payoff is terminal intrinsic liability at the observed underlying close. Actual contract availability, entry quotes, fills, settlement conventions and intraday exits were not reconstructed. Daily 0DTE availability also varied historically; these hypothetical cases do not assert that every tested contract existed. [Cboe's history of same-day trading](https://www.cboe.com/insights/posts/the-evolution-of-same-day-options-trading).

## Main result: fewer breaches, with less participation

Evaluation, primary 2% distance. Breach means at least one underlying finishes beyond a short strike. **It is not the net losing-trade rate, an intraday touch, or a full-width loss.** Each filtered rate uses that policy's own accepted days.

"""
main += table(["Structure", "Shape t8 sold days", "Always-sell breach", "Original t8 breach", "Shape t8 breach", "Shape t8 participation"], quick)
main += """

![Accepted-day breach rates and participation](spread_safety.png)

Every cell below is **participation / accepted-day breach rate / mean expiration liability**, all in percent. Liability is a percentage of wing width, averaged equally across the two assets. Different accepted-day sets mean this is a policy comparison, not a controlled attribution of forecast improvement.

""" + policy_table("evaluation") + """

- **Puts:** shape calibration rejects more days, while accepted-day breach rates and liabilities are slightly worse than the older filters.
- **Calls:** shape calibration accepts more days, with higher breach rates and liabilities than the older filters.
- **Iron condors:** shape calibration lowers average liability substantially versus always selling. Comparable protection already appears with affine Gaussian: 1.019% of width versus 1.021% for shape t8, at 63.42% versus 62.94% participation.

The copula contributes only to joint risks in this setup. Within a fixed marginal system, each asset's individual breach probability and the portfolio's mean liability are identical for Gaussian and t8 dependence. Across all 19,710 session/structure/distance cases, Gaussian versus t8 changes the sell decision in only **76 original, 52 affine and 60 shape cases**. A better joint density score alone does not establish a useful spread-selling edge.

Full-width losses remain. The following are **counts of any asset / both assets reaching full-width liability** among accepted days, not probabilities inferred from a large sample:

""" + table(["Structure", "Always sell", "Original t8", "Shape t8"], full) + """

All six evaluation put filters retain five simultaneous full-width losses. In development, every model filter retains the same simultaneous full-width counts: three puts, one call and three condors. These scarce events cannot certify safety.

## Sensitivities and development

The 1% and 3% distance results are retained for all models in [ALL_CASES.md](ALL_CASES.md). The headline is not uniform across distances. At 1%, the shape-t8 condor filter accepts only **32 of 1,457 evaluation sessions** and records **6 breaches (18.75%)**, despite a forecast gate of at most 10%. Original t8 accepts 49 sessions and records 6 breaches (12.24%). The samples are small; these are concerning descriptive rates, not a new hypothesis test. At 3%, shape t8 accepts about 91% of condor sessions, with a 1.82% accepted-day breach rate, close to original t8's 1.80% at about 92% participation.

Development at the primary distance, using the same units as above:

""" + policy_table("development") + """

## Premiums, sizing and avoided losses

The account engine uses three fixed credits: **5%, 10% and 20% of wing width**, with assumed round-trip costs of **2% of width**. For a condor, credit is the combined credit for both wings. These credits and costs are scenarios, not observed prices. On a sold day, equity return is `0.02 × (credit − cost − portfolio liability)`; otherwise it is zero. Phase accounts reset to one.

A low breach probability is only part of the spread-selling decision. The available premium must compensate for expected liability, execution costs and the risk being borne. Quiet days at distant strikes may offer very little premium. Applying the same credit to every historical day can create large compounded returns with no evidence that those trades were available.

For scale, historical average liability plus the assumed cost gives the following **average-cost credit thresholds**. These are percentages of wing width, not percentages of underlying price or equity. They are retrospective arithmetic, not live minimum-credit recommendations or compensation for model uncertainty and tail risk.

""" + table(["Structure", "Always sell", "Original t8 accepted days", "Shape t8 accepted days"], credit) + """

Under the **10%-of-width credit scenario only**, evaluation maximum account drawdown changes from 9.52% to 3.18% for puts, 12.13% to 2.18% for calls, and 23.98% to 1.84% for condors when comparing always selling with shape t8. These are expiration-only hypothetical account drawdowns. Actual intraday drawdowns, margin calls, assignment and executable exits are outside the data.

Foregone premium matters. At the 5%-credit assumption, shape t8 has higher terminal wealth than always selling for all three primary structures; at 20%, it has lower terminal wealth for all three. At 10%, only the condor improves terminal wealth versus always selling. This change in ranking is a reason to collect actual quotes, not choose a favorable assumed premium. All **378 phase/case/policy/credit rows** and their compounded returns, drawdowns and attribution are retained in [ALL_CASES.md](ALL_CASES.md).

The saved path decomposes each day's return difference from always selling into avoided liability, missed credit and saved fees. This identity applies to daily equity fractions; their sums are **not** a decomposition of the difference between two compounded terminal wealth paths.

## What the new calibration changed

After the frozen affine correction, a bounded-parameter sinh-arcsinh transformation adjusts shape using only matured historical forecast errors at each monthly refit. Scores use full normalized densities, including the recalibration Jacobian. The Gaussian and t8 dependence arms are crossed with original, affine and shape marginals; single-asset effects are therefore distinguishable from dependence effects in the design. The transformation is motivated by [Jones and Pewsey, Sinh-arcsinh distributions](https://doi.org/10.1093/biomet/asp053); this implementation fixes the affine parameters separately rather than fitting all four parameters jointly.

Descriptive shape-minus-affine mean negative log-density differences (lower is better):

"""
diffs = []
for phase in ["development", "evaluation"]:
    vals = []
    for asset in ["qqq", "spx"]:
        d = density[(density.phase == phase) & (density.asset == asset)].set_index("margin")
        vals.append(f"{d.loc['shape','mean_loss']-d.loc['calibrated','mean_loss']:.6f}")
    diffs.append([phase.title(), *vals])
main += table(["Phase", "QQQ", "SPX"], diffs) + """

Evaluation lower/upper probability-transform tail rates for shape marginals are **2.81% / 1.65% for QQQ** and **2.88% / 2.20% for SPX**, against nominal 2.5% in each tail. They are generally closer than affine calibration, but positive normal-score mean bias increases and negative skew remains. At the primary 2% distance, all-session breach Brier scores improve slightly for puts/calls and worsen for condors in both phases, compared with affine calibration. There is no consistent improvement across every calibration or payoff metric.

These are descriptions of independently verified saved density rows. No p-values have been reconstructed and no attribution mechanism has been established. The old simulation remains an assumed-model benchmark, not a ceiling on possible score gains.

## Verification and the failed original attempt

The original shape study passed **173 selected pre-run tests**, including the forecast transformations, spread accounting and prospective ledger. Independent forecast reconstruction checked 105 monthly fits, 2,191 issued applications, 210 marginal shape fits and 210 new dependence fits, normalized density factors, maturity clocks and optimization certificates. The density score verification completed before the later numerical failure.

The spread sampler then exceeded its fixed numerical consistency limit for a sampled tail quantile: **0.03620 difference versus a 0.03 limit**. Some rare liabilities were also missed by finite sampling, yielding 429 sampled tail averages below the separately integrated mean. The original study remains UNEVALUABLE with all six planned comparisons recorded as p=1. No frozen result, threshold or forecast was changed to obtain a pass.

The separate replay passed **31 additional pre-run checks**. It authenticates and reuses the exact saved probabilities, strikes, dates and decisions, with no refit, resampling, new threshold, inference or promotion. The largest two-scramble breach-probability difference was 0.00354, below the original 0.005 diagnostic limit. All 69 model/case/date rows straddling the 10% threshold between scrambles retain the originally specified average-probability decision. Agreement between two scrambles is not a rigorous error bound.

The replay excludes sampled VaR and expected-shortfall estimates. Its adapter uses the known maximum terminal debit of one wing width to satisfy an unused legacy accounting field; that field is removed from published tables. Decisions and cash flows never depend on that field. The independent accounting verifier reconstructed all saved account outputs and reported VERIFIED.

The resulting tables contain 137,970 positions, 413,910 path rows, 19,710 shared realized outcomes, 378 scenario summaries and 324 probability/calibration summaries. Hash receipts bind them to the failed original attempt and the separately frozen replay protocol. See [terminal.json](terminal.json), [accounting_verification.json](accounting_verification.json), and [the preserved original terminal](../copula_shape/predictive/terminal.json).

## New-data system and next benchmark

The implemented, tested ledger preserves raw source bytes, model versions, immutable issued forecasts, revisions, exact receipt times and missing/late forecasts. Its current scorer supports scalar normal/Student-t densities and variance forecasts. The option decision adapter, calibrated joint-density scorer and live feed integration remain to be built; **collection is not running**.

The next benchmark should retain one frozen candidate and its control, exact option contracts and synchronized bid/ask quotes at the decision, selected legs and size, actual settlement and all fees. Retain rejected trades as well as accepted ones. A stop or early-exit rule needs intraday option quotes. Bind each forecast and decision to data received before its actual target starts.

Start with a fixed collection and evaluation plan, not a rule to stop when a result looks favorable. Five hundred new sessions can assess broad calibration but contain only 12.5 expected observations in a nominal 2.5% tail; a rare-loss safety claim requires more evidence and a planned precision/power analysis.

The complete operating plan, supported interfaces, provider documentation and remaining integration work are in [PROSPECTIVE_BENCHMARK.md](../../docs/PROSPECTIVE_BENCHMARK.md). No market data beyond 2025-10-20 was opened, and no subscription, live order or scheduled collector was created for this study.
"""
write_new("FINDINGS.md", main)

appendix = """# All retained spread cases

Post-result rendering of frozen replay summaries; no new model selection or tests.
All rates below refer to terminal outcomes. Liability is normalized by wing width.
All accounts start at one per phase. Hypothetical premiums are constant through time;
large compounded returns are not evidence of attainable option returns.

## Safety and coverage

Cells are participation / accepted-day breach / accepted-day mean liability, all percent.
Undefined means remain missing. The .02 distance was primary before this run.

"""
for phase in ["development", "evaluation"]:
    for distance in [.01, .02, .03]:
        appendix += f"### {phase.title()}, {100*distance:.0f}% distance\n\n" + policy_table(phase,distance) + "\n\n"
appendix += "## All probability and mean-liability diagnostics\n\n"
appendix += "All / sell / skip partitions are retained. Predictions are means across the named partition. Observed/predicted debit and average-cost credit are percentages of width. Brier and payout MSE use fraction-scale inputs. Threshold ambiguities count model/case/date rows.\n\n"
rows=[]
for r in safety.itertuples(index=False):
    rows.append([r.phase,r.structure,f"{r.distance:.2f}",labels[r.model],r.selection,r.sessions,
        f"{100*r.coverage:.2f}",f"{100*r.p_any_breach:.3f}",f"{100*r.any_breach:.3f}",
        f"{100*r.p_both_breach:.3f}",f"{100*r.both_breach:.3f}",
        f"{100*r.p_both_full:.3f}",f"{100*r.both_full_loss:.3f}",
        f"{100*r.mean_debit:.4f}",f"{100*r.portfolio_debit:.4f}",
        f"{r.brier_any:.6f}",f"{r.brier_both:.6f}",f"{r.payout_mse:.6f}",
        f"{100*r.break_even_credit:.4f}",r.numerical_threshold_ambiguities])
appendix += table(["Phase","Structure","Distance","Model","Partition","N","Coverage %","Pred any %","Obs any %","Pred both %","Obs both %","Pred both full %","Obs both full %","Pred debit %","Obs debit %","Brier any","Brier both","Payout MSE","Average-cost credit %","Ambiguities"],rows)+"\n\n"
appendix += "## All hypothetical-credit account results\n\n"
appendix += "Credit is a percentage of width; costs are 2% of width. Returns/drawdowns are percentages of account equity. Attribution sums are daily equity-fraction contributions, shown in percentage points; they do not decompose compounded wealth. Every policy/case/phase is retained for credits 5%, 10%, 20%.\n\n"
rows=[]
for r in summary.itertuples(index=False):
    rows.append([r.phase,r.structure,f"{r.distance:.2f}",labels[r.model],f"{100*r.credit_fraction:.0f}",
        r.sold_sessions,f"{100*r.total_return:.3f}",f"{100*r.max_drawdown:.3f}",
        f"{100*r.worst_net_return:.3f}",f"{100*r.avoided_liability_return_sum:.3f}",
        f"{100*r.missed_credit_return_sum:.3f}",f"{100*r.saved_fee_return_sum:.3f}",r.ruined])
appendix += table(["Phase","Structure","Distance","Model","Credit %","Sold N","Total return %","Max drawdown %","Worst day %","Avoided liability sum pp","Missed credit sum pp","Saved fee sum pp","Ruined"],rows)+"\n\n"
appendix += "## All marginal density descriptions\n\nLower/upper PIT tails have a nominal rate of 2.5% each. Lower negative log-density loss is better; these means carry no recovered significance.\n\n"
rows=[]
for r in density.itertuples(index=False):
    rows.append([r.phase,r.asset,r.margin,r.n,f"{r.mean_loss:.6f}",f"{r.normal_mean:.6f}",f"{r.normal_variance:.6f}",f"{r.normal_skew:.6f}",f"{100*r.pit_lower:.3f}",f"{100*r.pit_upper:.3f}"])
appendix += table(["Phase","Asset","Margin","N","Mean loss","Normal mean","Normal variance","Normal skew","Lower tail %","Upper tail %"],rows)+"\n"
write_new("ALL_CASES.md",appendix)

plot_rows=[]
for structure in structures:
    for model in ["always_sell","orig_t8","shape_t8"]:
        r=primary.loc[(model,structure)]
        plot_rows.append({"structure":structure,"model":model,"sold_sessions":int(r.sold_sessions),
                          "coverage":float(r.coverage),"breach_rate":float(r.sold_any_breach_rate)})
write_new("FIGURE_DATA.json",json.dumps(plot_rows,indent=2)+"\n")
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"axes.spines.top":False,
                     "axes.spines.right":False,"axes.spines.left":False,"axes.spines.bottom":False})
fig, axes=plt.subplots(1,2,figsize=(12,5.9),gridspec_kw={"width_ratios":[1.25,1]})
fig.patch.set_facecolor("#fbfaf7")
palette={"always_sell":"#9da4ad","orig_t8":"#517d93","shape_t8":"#c46d47"}
y=np.arange(3); offsets=[-.23,0,.23]
for ax in axes:
    ax.set_facecolor("#fbfaf7");ax.set_axisbelow(True);ax.xaxis.grid(True,color="#dfdfdb",linewidth=.8)
    ax.set_ylim(-.62,2.65);ax.invert_yaxis();ax.set_yticks(y,["Put spread","Call spread","Iron condor"])
    ax.tick_params(axis="both",length=0,pad=7)
for offset,model in zip(offsets,["always_sell","orig_t8","shape_t8"]):
    values=[100*primary.loc[(model,s)].sold_any_breach_rate for s in structures]
    coverage=[100*primary.loc[(model,s)].coverage for s in structures]
    axes[0].barh(y+offset,values,height=.19,color=palette[model],label=labels[model])
    axes[1].barh(y+offset,coverage,height=.19,color=palette[model])
    for yy,v in zip(y+offset,values):axes[0].text(v+.15,yy,f"{v:.1f}%",va="center",fontsize=10)
    for yy,v in zip(y+offset,coverage):axes[1].text(v+1.1,yy,f"{v:.1f}%",va="center",fontsize=10)
axes[0].set_xlim(0,12);axes[1].set_xlim(0,116)
axes[0].set_title("Breach rate on accepted days",loc="left",fontsize=13,pad=14)
axes[1].set_title("Share of days sold",loc="left",fontsize=13,pad=14)
axes[0].set_xlabel("At least one underlying beyond a short strike (%)",fontsize=10)
axes[1].set_xlabel("Participation (%)",fontsize=10)
axes[1].set_yticklabels([])
fig.suptitle("0DTE proxy: fewer breaches, with less participation",x=.05,y=.98,ha="left",fontsize=18,fontweight="bold")
fig.text(.05,.905,"2020–2025 · 1,457 sessions · 2% short-strike distance · 1% wing width",fontsize=11,color="#555d64")
handles,legend_labels=axes[0].get_legend_handles_labels()
fig.legend(handles,legend_labels,ncol=3,loc="lower left",bbox_to_anchor=(.04,.063),frameon=False)
fig.text(.05,.035,"Historical underlying-close outcomes; no actual option quotes. Breach is not net trade loss. Reused history, descriptive results.",fontsize=9,color="#555d64")
fig.subplots_adjust(left=.13,right=.97,top=.81,bottom=.24,wspace=.25)
for suffix in ["png","pdf"]:
    target=OUT/f"spread_safety.{suffix}"
    if target.exists():raise FileExistsError(target)
    fig.savefig(target,dpi=180,facecolor=fig.get_facecolor())
plt.close(fig)
receipt={"scope":"Presentation only; authenticated saved outputs; no fits, sampling or hypothesis tests.",
         "input_terminal_sha256":hashlib.sha256((OUT/"terminal.json").read_bytes()).hexdigest(),
         "artifacts":{name:hashlib.sha256((OUT/name).read_bytes()).hexdigest()
                      for name in ["FINDINGS.md","ALL_CASES.md","FIGURE_DATA.json","spread_safety.png","spread_safety.pdf","render_findings.py"]}}
write_new("FIGURE_RECEIPT.json",json.dumps(receipt,indent=2)+"\n")
print(json.dumps({"report":"FINDINGS.md","appendix":"ALL_CASES.md","plot":"spread_safety.png",
                  "summary_rows":len(summary),"safety_rows":len(safety)},indent=2))
