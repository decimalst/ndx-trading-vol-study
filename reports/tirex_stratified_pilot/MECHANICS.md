# Architecture and mechanics checked against the pinned installation

A patched **unit** is one of 1,408 intermediate coordinates of the time-mixer gated feed-forward network, immediately before `stack[layer].time_mixer.ffn.proj_down`. Its value is `SiLU(W_gate h)[i] * (W_up h)[i]`; the 1,408-vector is projected to the 512-dimensional residual stream. It is neither a residual-stream coordinate nor a recurrent sLSTM/mLSTM cell state.

Pinned checkpoint `05e5b26db52bfb256f1ae1bdf785589850482de3` uses 12 alternating mLSTM and sLSTM blocks; zero-index layers 0/5/11 have mLSTM/sLSTM/sLSTM recurrence before the hooked FFN. The first-block unit is early in depth, but has already passed through nonlinear patch embedding, RMS normalization, recurrent mixing, another normalization, and a SiLU product gate. A near-linear recent-mean mechanism is a hypothesis, not a demonstrated property of layer 0.

The coordinate-wise gate supplies an architectural native basis; gated MLPs in other architectures can have such a basis too. This pilot does not compare native coordinates with rotated coordinates and cannot resolve a general native-basis hypothesis.

Each history has 2,048 observations. Non-overlapping input patches contain 32 observations, hence 64 observed tokens. The wrapper pads a 320-step future window (10 tokens); the hook sees 74 tokens in total. Token 63 is zero-indexed: it is the final observed patch (input observations 2,016–2,047), not a particular observed day. TiRex's shift/slicing maps the first forecast output to that token's output patch. The public call asks for prediction_length=1: a single next observation, not 63 steps ahead. Synthetic data do not themselves define a calendar horizon.

The native scaler computes full-history mean and population SD, standardizes, then applies arcsinh (enabled in this checkpoint). Missing future placeholders do not enter its NaN-aware statistics. Outputs go through inverse sinh and the retained location/scale. The public forecast wrapper also applies its configured uncertainty postprocessing. TTA sign flips/differencing and training dropout are disabled; we retain q10, q50, q90 in original input units (synthetic log variance).

Median change is the primary point-forecast sensitivity metric. It is not a proper forecast score and does not establish accuracy. q10/q90 and q90−q10 effects must be retained to expose distribution-shape movement; no future outcomes exist in this synthetic experiment for calibration or pinball-loss assessment.

Sources inspected: installed `tirex2/model/component/bi_xlstm.py`, `tirex2/model/tirex2.py`, `tirex2/model/component/scaler.py`, `tirex2/model/component/patch_tokenizer.py`, `xlstm/xlstm_large/model.py`, pinned model-config.yaml, and frozen repository `src/tirex_intervention_adapter.py`. Installed package/checkpoint hashes are captured by the runner.
