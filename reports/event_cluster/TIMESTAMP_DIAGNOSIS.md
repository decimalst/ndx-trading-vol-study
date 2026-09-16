# Event-cluster timestamp failure: bounded transport review

The saved application cutoff and the independently reconstructed cutoff represent the same dates, but have different datetime units. The saved state column is `datetime64[us]`; the source feature column and independent reconstruction are `datetime64[ms]`. The frozen exact-frame check rejects that dtype difference. This review reproduces the mismatch without fitting any model or inspecting predictive values. It does not verify the generated forecasts or rescue the failed wave20 trial.

## Scope and observed source schemas

Read-only inspection used source code, test definitions, the canonical FAILED verification record, Parquet schema metadata, file hashes, and only datetime columns from the original features and new states/memory. No probabilities, losses, event indicators, memory measurements, fitted coefficients or scored diagnostic payloads were decoded. The only new computation used actual date columns to compare cutoff instants and unknown masks. The synthetic demonstration below used invented 2001 dates and no outcomes. No repository files were written, and no verifier entrypoint or optimizer was run.

The installed runtime reports pandas **3.0.5** and PyArrow **25.0.1**. Observed Parquet timestamp fields are:

| Artifact | Date-field units |
|---|---|
| `data/range_alert/features.parquet` | index `date` and `feature_cutoff_date`: ms |
| `data/range_alert/targets.parquet` | index `date`, `target_end`, `available_date`: ms |
| `data/range_alert/states.parquet` | `origin`, `feature_cutoff_date`: ms; source/seed/latest-consumed audit dates: us |
| `data/event_cluster/states.parquet` | `origin` and all four window-date fields: ms; `feature_cutoff_date`, `source_fit_origin`: us |
| `data/event_cluster/memory.parquet` | index `date`, `feature_cutoff_date`, all four window-date fields: ms |
| `data/event_cluster/forecasts.parquet` | origin/cutoff/target/availability: ms; monthly-fit/training metadata dates: us |

Every saved state cutoff equals the corresponding original feature cutoff after lossless conversion to nanosecond representation, and their unknown masks agree. A direct exact DataFrame comparison nevertheless fails with:

```text
Attributes of DataFrame.iloc[:, 0] (column name="feature_cutoff_date") are different
Attribute "dtype" are different
[left]:  datetime64[us]
[right]: datetime64[ms]
```

This confirms the specific date-schema failure. It does not establish equality of other state values or successful independent numerical reconstruction.

## Exact construction and serialization path

The frozen [issued replay](/Users/byrons/code/trading-vol/ndx-vol-experiment/src/issued_calibration_models.py:302) writes the feature cutoff into an in-memory record as `str(source_timestamp.date())`, a `YYYY-MM-DD` string. The new frozen [producer pipeline](/Users/byrons/code/trading-vol/ndx-vol-experiment/src/event_cluster_pipeline.py:181) reparses that string with `pd.Timestamp(...)` and constructs its state DataFrame. In this runtime that string route produces microsecond timestamps.

The [independent helper](/Users/byrons/code/trading-vol/ndx-vol-experiment/src/event_cluster_verification.py:171) instead retrieves the cutoff directly from the original feature column, preserving its millisecond scalar resolution. Its [_frame check](/Users/byrons/code/trading-vol/ndx-vol-experiment/src/event_cluster_verification.py:35) calls `assert_frame_equal(..., check_exact=True)` with the default dtype check enabled. The all-state check occurs at [the pre-solve boundary](/Users/byrons/code/trading-vol/ndx-vol-experiment/src/event_cluster_verification.py:218), before its loop of independent stage solves.

The [runner](/Users/byrons/code/trading-vol/ndx-vol-experiment/src/event_cluster_search.py:765) writes the producer DataFrame to Parquet without an explicit timestamp-unit normalization. The [verifier entry](/Users/byrons/code/trading-vol/ndx-vol-experiment/src/verify_event_cluster.py:1469) reads the pinned Parquet bytes. PyArrow preserves the distinct ms/us units; it does not create the mismatch. The divergence is already present in memory and survives the roundtrip.

## Synthetic roundtrip falsifier

The following date-only construction was executed with source units ms, us and ns. It mirrors the two actual cutoff construction routes and invokes only the frozen comparison helper; it calls no producer pipeline, replay or fitter.

```python
import io
import pandas as pd
from src.event_cluster_verification import _frame

def roundtrip(frame):
    buffer = io.BytesIO()
    frame.to_parquet(buffer)
    return pd.read_parquet(io.BytesIO(buffer.getvalue()))

dates = pd.bdate_range("2001-02-01", periods=4, name="date").as_unit("ms")
source = roundtrip(pd.DataFrame({
    "feature_cutoff_date": pd.Series(dates, index=dates).shift(1)
}, index=dates))
selected = source.index[1:]
issued_text = [str(source.loc[d, "feature_cutoff_date"].date()) for d in selected]
produced = pd.DataFrame({
    "feature_cutoff_date": [pd.Timestamp(text) for text in issued_text]
})
expected = pd.DataFrame({
    "feature_cutoff_date": [source.loc[d, "feature_cutoff_date"] for d in selected]
})
restored = roundtrip(produced)
assert all(a == b for a, b in zip(
    restored.feature_cutoff_date, expected.feature_cutoff_date
))
_frame(restored, expected, "synthetic application cutoff")  # Expected failure.
```

| Source unit | Produced / roundtrip unit | Scalar instants | Frozen check before / after Parquet |
|---|---|---|---|
| ms | us / us | All equal | FAIL / FAIL: us versus ms |
| us | us / us | All equal | PASS / PASS |
| ns | us / us | All equal | FAIL / FAIL: us versus ns |

The default synthetic `pd.bdate_range(...)` in this runtime has us resolution. The [full mathematical pipeline fixture](/Users/byrons/code/trading-vol/ndx-vol-experiment/tests/test_range_alert_models.py:19) uses that default, so its source and reparsed timestamp resolutions coincide. The [supplied-table integration test](/Users/byrons/code/trading-vol/ndx-vol-experiment/tests/test_event_cluster_verification.py:23) checks its generated outputs directly. The separate [temporary-tree publication fixture](/Users/byrons/code/trading-vol/ndx-vol-experiment/tests/test_verify_event_cluster.py:810) mocks the numerical pipeline helper while testing publication behavior. Those tests covered useful separate contracts but did not exercise an unmocked full pipeline across the actual mixed-unit transport boundary. The passed suite therefore did not establish this compatibility.

## Conditions for a future separately registered replay

Before any future replay, prewrite a synthetic full-pipeline transport test whose original source calendar and cutoff columns explicitly use ms, then serialize and reload the generated states/memory/panel before calling the unmocked independent helper. Include us and ns representations, a wholly unscored application, NaT-mask tampering, a one-day cutoff mutation and a subday/timezone mutation. The ms fixture must fail the present frozen implementation as shown above. A future implementation must declare one exact date representation or a lossless date-only comparison rule before registration, then demonstrate that equal instants pass while altered instants, unknown masks and chronology fail. It must preserve exact numeric, boolean, integer, row-order and column-schema checks; a blanket `check_dtype=False` on the entire state frame would weaken unrelated verification and is not justified.

This note does not choose or implement that future contract. The frozen wave20 verifier and trial accounting remain unchanged. Its three hypotheses remain unevaluable at p=1; generated rows and schedules are not independently verified forecasts or evidence for or against the predictive mechanism. A corrected transport contract may expose further failures once independent stage solves are reached. Neither successful future numerical verification nor any predictive result follows from this diagnosis.

## Inspected artifact identities

| Repository-relative path | SHA-256 |
|---|---|
| `data/range_alert/features.parquet` | `b6096cc95f64819158ce79fb65303cc47b0633c4b313d4dad9c5b4206c4cd8ab` |
| `data/range_alert/targets.parquet` | `478569c32021dd698d9b7504fa7835ecdaedb040463ce217f6b441db0bbfab5e` |
| `data/range_alert/states.parquet` | `f633eaafd4e7b7bd77e4ab6fff11a474602e0f6bed44b9fd3c737a58c9e5dc9f` |
| `data/event_cluster/states.parquet` | `ebd3dc31210efebdd392ad91989ae64d412cff1b24a04e59c7cb06003fe81e73` |
| `data/event_cluster/memory.parquet` | `f601936f5c0ca726be099f370bd9995ead99f15b9fcd71933b5d0d8dbb6bd8d0` |
| `data/event_cluster/forecasts.parquet` | `c90bb1ad21c6da84e91b77ce71cbb2ac4033816cd0431163c8334fdea87924c0` |

These are whole-file identity hashes. The date-schema inspection of target/forecast files did not decode their numerical target or score columns.
