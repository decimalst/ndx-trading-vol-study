"""Present authenticated hindsight outputs; no search, fit or hypothesis test."""
from pathlib import Path
import hashlib
import json
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FuncFormatter, NullFormatter
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
DATA = ROOT / "data/model_memory_study/spread_geometry_hindsight"
terminal = json.loads((OUT / "terminal.json").read_text())
for name, expected in {**terminal["outputs"], **terminal["anchors"]}.items():
    if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
        raise ValueError(f"Saved output changed: {name}")
g = pd.read_parquet(DATA/"geometry.parquet")
a = pd.read_parquet(DATA/"accounts.parquet")
w = pd.read_parquet(DATA/"winners.parquet")
POLICIES = ["always_sell", "fixed_2pct_orig_t8", "fixed_2pct_shape_t8"]
LABELS = {"always_sell": "Always sell", "fixed_2pct_orig_t8": "Original t8 day filter",
          "fixed_2pct_shape_t8": "Shape t8 day filter"}
PHASES = ["development", "evaluation", "pooled"]

def label(name):
    return LABELS.get(name, name.replace("fixed_2pct_", "").replace("_", " "))

def write(name, content):
    if name.endswith(".md"):
        content = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", content)
        content = content.replace("t 8", "t8").replace("%,", "%, ")
        content = re.sub(r"(?<=\d)(?=bps\b)", " ", content)
        content = re.sub(r"(?<=[A-Za-z])[,:;](?=\d)", lambda match: match.group(0)+" ", content)
        content = content.replace("of**", "of **")
    with (OUT/name).open("w" if "--refresh-presentation" in sys.argv else "x") as stream:
        stream.write(content)

def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"]*len(headers)) + " |",
                      *("| " + " | ".join(map(str,row)) + " |" for row in rows)])

zero_rows = []
for policy in POLICIES:
    for structure in ["put", "call", "condor"]:
        cells = []
        for phase in PHASES:
            z = g[(g.policy == policy)&(g.structure == structure)&(g.phase == phase)]
            found = z.groupby("distance_bps").mean_debit.apply(lambda s: s.eq(0).all())
            hits = found[found].index.tolist()
            n = int(z.sold_sessions.iloc[0])
            cells.append(f"{hits[0]/100:.2f}% ({n:,} days)" if hits else f"None ≤5% ({n:,} days)")
        zero_rows.append([label(policy), structure.title(), *cells])

chosen = w.merge(g, on=["phase","structure","policy","distance_bps","width_bps"], suffixes=("","_geometry"))
top_rows = []
for r in chosen[(chosen.phase == "evaluation") & chosen.policy.isin(POLICIES)
                & chosen.scenario.isin(["width_10", "open_5bp"])].itertuples(index=False):
    top_rows.append([label(r.policy), r.structure.title(), r.scenario,
                     f"{r.distance_bps/100:.2f}%",f"{r.width_bps/100:.2f}%",r.tie_count,
                     f"{r.any_breach_count}/{r.sold_sessions}",f"{100*r.mean_debit:.4f}%",
                     f"{r.average_cost_credit_open_bps:.4f}"])

report = """# Hindsight spread geometry: the attractive cases and the pricing trap

**Completed 2026-09-13. Explicitly selected with hindsight; no predictive promotion.**

The most striking evaluation case is the **shape-t8 day filter with iron-condor short strikes 3.75% from the open**: neither underlying breached a short strike on any of its **917 accepted sessions** in 2020–2025. Every tested wing width, **0.10% through 3.00% of opening price**, therefore had zero expiration liability. There are 48 tied geometries at distances3.75–5.00%; the displayed0.10% wing is just the declared tie-breaking choice.

Over the full 2017–2025 history, the first zero-liability condor distance is **4.25%**, across **1,542 shape-filtered days**. The3.75% geometry had **two development breaches in625 days**, so the later-period result does not hold across the entire sample. The original t8 filter also reaches zero liability at4.25% over its1,608 pooled accepted days.

This search found attractive hindsight **strike distances**. It did not identify a uniquely superior **wing width**. The premium convention determines the width winner, and actual quotes remain missing.

## What was searched

- **20 distances:** 0.25%–5.00% from each underlying's opening price, every0.25%.
- **Eight wing widths:** 0.10%,0.25%,0.50%,0.75%,1.00%,1.50%,2.00%,3.00% of opening price.
- Puts, calls and symmetric iron condors: **480 geometries**. QQQ and SPX are equally allocated.
- Always selling plus all six unchanged model day-selection masks from the earlier2%-distance/1%-width test. **These masks are fixed across the new geometries. They do not forecast a10% breach probability for the new strikes.** No model was refit, no joint probabilities were recomputed and no day was selected using its realized return.
- The inherited development733/evaluation1,457 sessions, plus their pooled2,190 sessions. Origins begin2017-02-01; target dates run2017-02-02 through2025-10-20. Each phase account resets to one; pooled accounting compounds all saved sessions in chronological order.
- Three constant credits of5%,10%,20% of width, plus a deliberately different constant credit of**5 basis points of opening price**. Assumed round-trip costs are2% of width throughout. A condor's credit covers both wings together.
- Each separate account reserves **2% of equity in gross maximum expiration liability**, split equally across assets. Narrower widths imply more underlying notional at the same reserve:0.10% wings imply20× combined notional/equity;3% wings imply about0.67×. This is fractional proxy sizing, with no actual contracts, integer lots, broker margin or assignment model.

The search retains10,080 geometry summaries and40,320 account scenarios. It selects252 within-period/structure/policy/premium winners. All choices use the full named period in retrospect. There are no new p-values, untouched out-of-sample claims or live trading recommendations.

## First distance with zero observed expiration liability

Each entry is the nearest **tested** short-strike distance with zero liability across all eight widths, followed by the number of accepted days. A missing result means none within this finite grid. This is not an estimate of zero future risk.

""" + table(["Day filter","Structure","Development","Evaluation","Pooled"],zero_rows) + """

The original-t8 and shape-t8 filtered call spreads still have one breach at5% distance in evaluation and pooled history. In the saved data, an accepted call day ending **2022-02-24** includes a6.79% open-to-close rise in one underlying. A5%-away,3%-wide wing reduces that event's averaged portfolio debit to29.84% of width; a0.10%-wide wing has50% averaged portfolio debit. These are expiration liabilities before premium and fees.

Always selling does not become loss-free at5%: pooled history retains3 put,6 call and9 condor breach days. Zero account drawdown can still coexist with intrinsic payouts when the assumed premium covers them; zero drawdown is not the same as zero liability.

## Why the premium assumption chooses the width

For a vertical, normalized expiration liability is the distance beyond the short strike divided by wing width, capped at one. Moving the short strike farther away lowers this liability; widening the wing lowers it per unit of maximum liability. With unchanged trading days, fixed capital reserve and the same premium **fraction of width**, every daily return weakly improves along those directions. The farthest/widest corner is therefore guaranteed to be a maximum before looking at any returns. Zero-liability regions create ties.

Across the63 period/structure/policy groups, each of the5%,10%,20%-of-width credit scenarios has **47 groups with zero-liability ties across all widths**, and the remaining **16 select the widest3% wing**. Under the alternative fixed5bps-of-opening-price premium, **all63 groups select the narrowest0.10% wing**. Distance is still mechanically rewarded because both premium conventions hold credit constant as distance changes.

A5bps premium on a0.10% wing pays50% of width. At the2% reserve, a zero-liability day then earns an assumed0.96% of account equity after fees. Its enormous compounded returns are consequences of these assumptions, not evidence that such premiums were quoted. No premium surface, executable spread fill, changing quote availability or price/volatility relationship has been reconstructed.

For comparison, the standard spread relationship links maximum loss to strike width minus collected credit; the required credit cannot be inferred from width alone. [Options Industry Council: bull put spreads](https://prd-web.optionseducation.org/strategies/all-strategies/bull-put-spread-credit-put-spread). Our sizing conservatively reserves gross width before credit so that varying the premium does not itself increase the liability budget.

Evaluation selections under two premium conventions are below. “Ties” counts full distance/width pairs within1e-10 of maximum total log growth. The representative is the nearest distance, then narrowest width within that set. Mean liability is a percentage of width; cost credit is in opening-price basis points.

""" + table(["Day filter","Structure","Credit convention","Distance","Width","Ties","Breach days","Mean liability","Average-cost credit (bps open)"],top_rows) + """

## The useful output: what credit would have covered historical liability?

The **average-cost credit** is accepted-day average expiration liability plus the assumed cost. It is presented both as a percentage of width and in basis points of opening price. It is an arithmetic historical daily break-even calculation, not the credit that guarantees nonnegative compounded wealth or compensates future risk, tail uncertainty and financing.

![Historical average-cost credit by geometry](credit_surface.png)

At zero historical liability, this threshold collapses to the assumed fee:2% of width. That is0.2bps of opening price for a0.10% wing, or6bps for a3% wing. The right-hand chart makes that width-dependent fee assumption visible.

For the shape-filtered evaluation condor cohort, the table below holds short strikes at2% and varies the width. **All widths retain the same29 short-breach days out of917**, while the liability and full-loss counts change.

"""
width_rows=[]
for r in g[(g.phase=="evaluation")&(g.policy=="fixed_2pct_shape_t8")&(g.structure=="condor")&(g.distance_bps==200)].itertuples(index=False):
    width_rows.append([f"{r.width_bps/100:.2f}%",r.both_full_count,f"{100*r.mean_debit:.3f}%",f"{100*r.average_cost_credit_fraction:.3f}%",f"{r.average_cost_credit_open_bps:.3f}"])
report += table(["Width","Both assets full-width loss days","Mean liability (% width)","Average-cost credit (% width)","Average-cost credit (bps open)"],width_rows)
report += """

Wider wings reduce normalized liability but can increase liability in opening-price units. A wider spread also uses fewer units of underlying exposure under the fixed reserve. Comparing widths without specifying both premium and position sizing would obscure these differences.

## Verification, preservation and next evidence

**38 pre-run tests passed** before the source/input freeze. They cover independent intrinsic payoffs, exact boundaries, condor maximum liability, extreme returns, unit conversions, fixed masks, phase/pooled compounding, geometry monotonicity, deterministic ties, empty groups and tamper rejection.

An independent verifier reconstructed every saved geometry, account and winner without importing the producer or prior accounting engine. The unchanged2%-distance/1%-width anchor reproduces **126 prior accounts and42 prior liability/event summaries**. The18 pinned source/code/receipt files and snapshot members, all saved outputs, private permissions and exact parent-derived cohorts also passed a separate artifact audit.

The original failed shape study remains UNEVALUABLE with its six comparisons recorded as p=1; the earlier descriptive replay is unchanged. This new hindsight search adds no formal hypothesis or trading lead. All inputs stop at the same historical ceiling. See [verification.json](verification.json), [INDEPENDENT_REVIEW.json](INDEPENDENT_REVIEW.json) and [terminal.json](terminal.json).

Full selected-case tables, including every premium scenario and every policy, are in [ALL_WINNERS.md](ALL_WINNERS.md). The complete geometry/account/winner tables remain in the private study directory. Hypothetical return numbers there are deliberately labeled and are not actual option returns.

The next useful check is the premium actually available at these strikes and widths. The existing [collection plan](../../docs/PROSPECTIVE_BENCHMARK.md) specifies exact contracts, synchronized quotes, decision timestamps, costs and settlement outcomes. Historical daily closes cannot show intraday touches, executable exits, assignment or live margin survival. A hindsight zero-breach geometry is a quote-checking candidate, not a safety certificate.
"""
write("FINDINGS.md",report)

appendix = """# All252 hindsight selections

Every period/structure/policy/credit scenario is retained. These are retrospectively selected from the full finite grid. Total return and drawdown are hypothetical percentages of account equity; they are not option-market observations. Width_05/10/20 assume those percentages of width as credit; open_5bp assumes5bps of opening price. Costs are2% of width. Cost-credit values are arithmetic historical thresholds only.

"""
for phase in PHASES:
    rows=[]
    for r in chosen[chosen.phase==phase].itertuples(index=False):
        rows.append([label(r.policy),r.structure,r.scenario,f"{r.distance_bps/100:.2f}%",f"{r.width_bps/100:.2f}%",r.tie_count,
                     f"{r.tie_distance_min_bps/100:.2f}–{r.tie_distance_max_bps/100:.2f}%",f"{r.tie_width_min_bps/100:.2f}–{r.tie_width_max_bps/100:.2f}%",
                     r.sold_sessions,r.any_breach_count,r.both_full_count,f"{r.average_cost_credit_open_bps:.4f}",
                     f"{100*r.total_return:.5g}",f"{100*r.max_drawdown:.4f}"])
    appendix += f"## {phase.title()}\n\n" + table(["Policy","Structure","Credit","Distance","Width","Ties","Tie distance range","Tie width range","Sold days","Breach days","Both full loss days","Average-cost credit (bps open)","Hypothetical total return %","Hypothetical max drawdown %"],rows)+"\n\n"
write("ALL_WINNERS.md",appendix)

surface=g[(g.phase=="evaluation")&(g.policy=="fixed_2pct_shape_t8")&(g.structure=="condor")].copy()
write("FIGURE_DATA.json",surface.to_json(orient="records",indent=2)+"\n")
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10})
fig,axes=plt.subplots(1,2,figsize=(13,6.9))
fig.patch.set_facecolor("#fbfaf7")
for ax,(column,title,multiplier) in zip(axes,[("average_cost_credit_fraction","Credit as % of wing width",100),("average_cost_credit_open_bps","Credit in basis points of opening price",1)]):
    frame=surface.pivot(index="distance_bps",columns="width_bps",values=column).sort_index()*multiplier
    matrix=frame.to_numpy()
    im=ax.imshow(matrix,aspect="auto",cmap="cividis",norm=LogNorm(vmin=float(matrix.min()),vmax=float(matrix.max())))
    ax.set_xticks(range(len(frame.columns)),[f"{x/100:g}" for x in frame.columns]);ax.set_xlabel("Wing width (% of opening price)")
    ticks=list(range(1,len(frame.index),2))
    ax.set_yticks(ticks,[f"{frame.index[i]/100:g}" for i in ticks]);ax.set_ylabel("Short-strike distance (% of open)")
    ax.set_title(title,loc="left",pad=12,fontsize=12)
    # Highlight the previous fixed geometry, not a selected winner.
    ax.scatter([list(frame.columns).index(100)],[list(frame.index).index(200)],marker="s",s=130,facecolors="none",edgecolors="white",linewidths=1.6)
    ax.axhline(list(frame.index).index(375)-.5,color="white",linestyle="--",alpha=.7,linewidth=1)
    bar=fig.colorbar(im,ax=ax,fraction=.047,pad=.03)
    ticks=[2,3,5,10,20,40,60] if multiplier==100 else [.2,.5,1,2,5,10,20,50]
    bar.set_ticks([v for v in ticks if matrix.min() <= v <= matrix.max()])
    bar.ax.yaxis.set_major_formatter(FuncFormatter(lambda value,_:f"{value:g}"))
    bar.ax.yaxis.set_minor_formatter(NullFormatter())
    bar.ax.tick_params(labelsize=9)
fig.suptitle("Hindsight credit thresholds change with the unit of comparison",x=.045,y=.975,ha="left",fontweight="bold",fontsize=17)
fig.text(.045,.914,"Shape t8 day filter · Iron condors · 917 accepted sessions, 2020–2025",fontsize=11,color="#50565b")
fig.text(.045,.071,"Threshold = observed mean expiration liability + assumed fees. White square: original 2% distance / 1% width.",fontsize=9,color="#50565b")
fig.text(.045,.041,"Below dashed line: zero historical liability; thresholds equal assumed fees. No actual quotes or future-risk premium included.",fontsize=9,color="#50565b")
fig.subplots_adjust(left=.08,right=.96,top=.82,bottom=.19,wspace=.27)
for suffix in ["png","pdf"]:
    path=OUT/f"credit_surface.{suffix}"
    if path.exists() and "--refresh-presentation" not in sys.argv:raise FileExistsError(path)
    fig.savefig(path,dpi=180,facecolor=fig.get_facecolor())
plt.close(fig)
print(json.dumps({"report":"FINDINGS.md","winner_rows_presented":len(chosen),"surface_rows":len(surface)}))
