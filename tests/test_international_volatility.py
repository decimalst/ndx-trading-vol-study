"""Synthetic wave-two contracts written before its producer implementation."""
import hashlib
import io
import unittest
import zipfile

import numpy as np
import pandas as pd

from src import iterative_hf
from src.international_volatility import (
    ALL_FEATURES,
    FOREIGN_COLUMNS,
    REGIONAL_COLUMNS,
    SYMBOLS,
    build_foreign_features,
    fit_log_ols,
    fit_pca,
    fit_predict,
    forecast_panel,
    parse_international_archive,
    regional_features,
    training_mask,
)
from tests.test_iterative_hf import fixture as spx_fixture


def foreign_fixture(dates):
    rng = np.random.default_rng(205)
    return {s: pd.Series(np.exp(rng.normal(-8 - i / 6, .6, len(dates))), index=dates)
            for i, s in enumerate(SYMBOLS)}


def complete_fixture(n=1400):
    daily, hf, vix = spx_fixture(n)
    base = iterative_hf.build_features(daily, hf, vix).loc[:, iterative_hf.BASE]
    foreign, _ = build_foreign_features(foreign_fixture(daily.index), daily.index)
    features = pd.concat([base, foreign], axis=1)
    targets = {h: iterative_hf.make_targets(hf.rv5, daily.index, h) for h in [1, 5, 21]}
    return features, targets


class InternationalArchiveContracts(unittest.TestCase):
    def archive(self):
        rows = []
        for symbol in SYMBOLS:
            for date, value in [("2017-06-01 00:00:00+01:00", .001),
                                ("2017-06-02 00:00:00+01:00", np.nan),
                                ("2017-06-05 00:00:00+01:00", .002)]:
                rows.append({"date": date, "Symbol": symbol, "rv5": value})
        content = io.BytesIO()
        with zipfile.ZipFile(content, "w") as z:
            z.writestr("archive.csv", pd.DataFrame(rows).to_csv(index=False))
        return content.getvalue()

    def test_fixed_six_symbols_hash_dates_and_source_end(self):
        raw = self.archive()
        series = parse_international_archive(raw, expected_sha256=hashlib.sha256(raw).hexdigest(), end="2017-06-02")
        self.assertEqual(tuple(series), SYMBOLS)
        for value in series.values():
            self.assertEqual(value.index.tolist(), [pd.Timestamp("2017-06-01"), pd.Timestamp("2017-06-02")])
            self.assertTrue(np.isnan(value.iloc[1]))
        with self.assertRaises(ValueError):
            parse_international_archive(raw, expected_sha256="0" * 64)

    def test_missing_registered_market_is_not_replaced(self):
        raw = self.archive()
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            frame = pd.read_csv(z.open(z.namelist()[0]))
        frame = frame[frame.Symbol != SYMBOLS[0]]
        content = io.BytesIO()
        with zipfile.ZipFile(content, "w") as z:
            z.writestr("archive.csv", frame.to_csv(index=False))
        with self.assertRaises(ValueError):
            parse_international_archive(content.getvalue())


class InternationalCalendarContracts(unittest.TestCase):
    def test_local_observation_rolling_precedes_us_asof(self):
        us = pd.bdate_range("2017-01-02", periods=50)
        local = pd.bdate_range("2016-12-01", periods=70).difference(pd.to_datetime(["2017-01-05", "2017-01-16"]))
        sources = {s: pd.Series(np.arange(1, len(local) + 1) / 1e5, index=local) for s in SYMBOLS}
        features, audit = build_foreign_features(sources, us)
        origin = pd.Timestamp("2017-01-17")
        source = pd.Timestamp("2017-01-13")
        expected = np.log(sources[SYMBOLS[0]].loc[:source].iloc[-5:].mean())
        self.assertAlmostEqual(features.loc[origin, "n225_w"], expected)
        row = audit[(audit.origin == origin) & (audit.symbol == ".N225")].iloc[0]
        self.assertEqual(row.source_date, source)
        self.assertEqual(row.cutoff_date, pd.Timestamp("2017-01-16"))
        self.assertEqual(row.extra_us_sessions, 1)

    def test_no_same_us_day_or_future_source_is_available(self):
        us = pd.bdate_range("2010-01-01", periods=60)
        sources = foreign_fixture(us)
        before, audit = build_foreign_features(sources, us)
        origin = us[40]
        for source in sources.values():
            source.loc[origin:] *= 100
        after, _ = build_foreign_features(sources, us)
        pd.testing.assert_frame_equal(before.loc[:origin], after.loc[:origin])
        used = audit.source_date.notna()
        self.assertTrue((audit.loc[used, "source_date"] <= audit.loc[used, "cutoff_date"]).all())

    def test_age_zero_exact_cutoff_three_valid_four_invalid(self):
        us = pd.bdate_range("2011-01-03", periods=50)
        sources = foreign_fixture(us[:35])
        features, audit = build_foreign_features(sources, us)
        for age, pos, valid in [(0, 35, True), (3, 38, True), (4, 39, False)]:
            row = audit[(audit.origin == us[pos]) & (audit.symbol == ".N225")].iloc[0]
            self.assertEqual(row.extra_us_sessions, age)
            self.assertEqual(bool(row.fresh), valid)
            self.assertEqual(bool(np.isfinite(features.loc[us[pos], "n225_m"])), valid)

    def test_foreign_session_on_us_holiday_waits_another_us_session(self):
        us = pd.to_datetime(["2017-07-03", "2017-07-05", "2017-07-06", "2017-07-07"])
        local = pd.bdate_range("2017-05-01", "2017-07-07")
        features, audit = build_foreign_features(foreign_fixture(local), us)
        july5 = audit[(audit.origin == pd.Timestamp("2017-07-05")) & (audit.symbol == ".N225")].iloc[0]
        self.assertEqual(july5.source_date, pd.Timestamp("2017-07-03"))
        self.assertTrue(np.isfinite(features.loc[pd.Timestamp("2017-07-05"), "n225_d"]))

    def test_missing_and_invalid_value_rows_are_retained_inside_rolling_windows(self):
        us = pd.bdate_range("2008-01-01", periods=65)
        for bad in [np.nan, np.inf, 0, -1]:
            sources = foreign_fixture(us)
            sources[".N225"].iloc[30] = bad
            features, _ = build_foreign_features(sources, us)
            self.assertTrue(np.isnan(features.loc[us[31], "n225_d"]))
            self.assertTrue(np.isnan(features.loc[us[35], "n225_w"]))
            self.assertTrue(np.isfinite(features.loc[us[36], "n225_w"]))

    def test_entire_unavailable_market_stays_missing_without_regional_reweighting(self):
        us = pd.bdate_range("2014-01-01", periods=50)
        sources = foreign_fixture(us)
        sources[".HSI"].iloc[30] = np.nan
        features, _ = build_foreign_features(sources, us)
        regional = regional_features(features)
        self.assertTrue(np.isnan(regional.loc[us[31], "asia_d"]))
        self.assertTrue(np.isfinite(regional.loc[us[31], "europe_d"]))

    def test_equal_weight_regional_means_use_raw_log_features(self):
        f = pd.DataFrame(np.arange(36).reshape(2, 18) / 10, columns=FOREIGN_COLUMNS)
        r = regional_features(f)
        self.assertEqual(tuple(r.columns), REGIONAL_COLUMNS)
        for k, suffix in enumerate(["d", "w", "m"]):
            np.testing.assert_allclose(r["asia_" + suffix], f.iloc[:, [k, k + 3, k + 6]].mean(axis=1))
            np.testing.assert_allclose(r["europe_" + suffix], f.iloc[:, [k + 9, k + 12, k + 15]].mean(axis=1))


class InternationalPCAContracts(unittest.TestCase):
    def matrix(self):
        rng = np.random.default_rng(940)
        return pd.DataFrame(rng.normal(size=(850, 18)), columns=FOREIGN_COLUMNS)

    def test_pca_geometry_is_training_standardized_and_unwhitened(self):
        raw = self.matrix()
        train, apply = raw.iloc[:750], raw.iloc[750:]
        result = fit_pca(train, apply)
        z = (train - train.mean()) / train.std(ddof=0)
        _, vectors = np.linalg.eigh(z.to_numpy().T @ z.to_numpy())
        projector = vectors[:, -3:] @ vectors[:, -3:].T
        components = np.asarray(result["audit"]["components"])
        np.testing.assert_allclose(components.T @ components, projector, atol=1e-12)
        np.testing.assert_allclose(result["train_scores"], z.to_numpy() @ components.T, atol=1e-12)
        np.testing.assert_allclose(result["apply_scores"], ((apply - train.mean()) / train.std(ddof=0)).to_numpy() @ components.T, atol=1e-12)
        self.assertTrue(np.all(components[np.arange(3), np.abs(components).argmax(axis=1)] > 0))

    def test_future_application_matrix_cannot_change_training_geometry(self):
        raw = self.matrix()
        one = fit_pca(raw.iloc[:750], raw.iloc[750:])
        two = fit_pca(raw.iloc[:750], raw.iloc[750:] * 100)
        self.assertEqual(one["audit"], two["audit"])

    def test_zero_scale_and_retained_boundary_tie_fail(self):
        raw = self.matrix()
        raw.loc[:, FOREIGN_COLUMNS[0]] = 1
        with self.assertRaises(ValueError):
            fit_pca(raw.iloc[:750], raw.iloc[750:])
        # Centered orthogonal columns of exactly equal variance tie all PCs.
        matrix = np.vstack([np.eye(18), -np.eye(18)])
        tied = pd.DataFrame(matrix, columns=FOREIGN_COLUMNS)
        with self.assertRaises(ValueError):
            fit_pca(tied, tied.iloc[:2])

    def test_component_sign_and_rotation_do_not_change_ols_predictions(self):
        raw = self.matrix()
        scores = fit_pca(raw.iloc[:750], raw.iloc[750:])
        rng = np.random.default_rng(401)
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        y = np.exp(rng.normal(-8, .3, 750))
        predictions = []
        for matrix in [np.eye(3), -np.eye(3), rotation]:
            a = pd.DataFrame(scores["train_scores"] @ matrix, columns=["pc1", "pc2", "pc3"])
            b = pd.DataFrame(scores["apply_scores"] @ matrix, columns=a.columns)
            a.insert(0, "const", 1.0)
            b.insert(0, "const", 1.0)
            predictions.append(fit_log_ols(a, y, b)[0])
        for prediction in predictions[1:]:
            np.testing.assert_allclose(prediction, predictions[0], rtol=1e-12)


class InternationalForecastContracts(unittest.TestCase):
    def test_training_requires_common_all_foreign_features_and_delayed_target(self):
        f, t = complete_fixture()
        f.loc[f.index[400], FOREIGN_COLUMNS[-1]] = np.nan
        origin = f.index[1000]
        mask = training_mask(f, t[21], origin, 750)
        self.assertFalse(mask.loc[f.index[400]])
        self.assertFalse(mask.loc[f.index[979]])
        self.assertTrue(mask.loc[f.index[978]])
        self.assertTrue((t[21].loc[mask, "available_date"] <= origin).all())
        with self.assertRaises(ValueError):
            training_mask(f, t[21], f.index[400], 750)

    def test_all_three_models_have_exact_log_ols_mean_forecasts(self):
        f, t = complete_fixture()
        f = f.dropna(subset=ALL_FEATURES)
        train, apply = f.iloc[:850], f.iloc[850:860]
        y = t[1].loc[train.index, "y"].to_numpy()
        result = fit_predict(train, y, apply)
        pca = result["pca_audit"]
        mean, scale, comp = map(np.asarray, [pca["means"], pca["scales"], pca["components"]])
        latent_a = (train.loc[:, FOREIGN_COLUMNS].to_numpy() - mean) / scale @ comp.T
        latent_b = (apply.loc[:, FOREIGN_COLUMNS].to_numpy() - mean) / scale @ comp.T
        for model, a_extra, b_extra in [
            ("baseline", np.empty((len(train), 0)), np.empty((len(apply), 0))),
            ("regional", regional_features(train).to_numpy(), regional_features(apply).to_numpy()),
            ("latent", latent_a, latent_b),
        ]:
            a = np.column_stack([train.loc[:, iterative_hf.BASE], a_extra])
            b = np.column_stack([apply.loc[:, iterative_hf.BASE], b_extra])
            beta = np.linalg.lstsq(a, np.log(y), rcond=None)[0]
            expected = np.exp(b @ beta) * np.exp(np.log(y) - a @ beta).mean()
            np.testing.assert_allclose(result["predictions"][model], expected, rtol=1e-10)

    def test_panel_common_origins_publication_fence_and_monthly_fit(self):
        f, t = complete_fixture()
        cfg = {"horizons": [1, 5, 21], "baseline": list(iterative_hf.BASE),
               "development": [str(f.index[900].date()), str(f.index[1000].date())],
               "evaluation": [str(f.index[1001].date()), str(f.index[1150].date())],
               "development_target_available_by": str(f.index[1000].date()),
               "latest_target": str(f.index[1150].date()), "minimum_train": 750}
        forecasts, fits = forecast_panel(f, t, cfg)
        self.assertEqual(len(forecasts.groupby(["model", "horizon"])), 9)
        self.assertEqual(forecasts.groupby(["model", "horizon"]).size().nunique(), 1)
        dev = forecasts[forecasts.phase == "development"]
        self.assertTrue((dev.available_date <= pd.Timestamp(cfg["development_target_available_by"])).all())
        self.assertTrue((forecasts.target_end <= pd.Timestamp(cfg["latest_target"])).all())
        self.assertTrue((forecasts.train_last_available <= forecasts.fit_origin).all())
        self.assertEqual(forecasts.groupby(["horizon", forecasts.origin.dt.to_period("M")]).fit_origin.nunique().max(), 1)
        self.assertEqual(len(fits), forecasts.groupby(["horizon", "fit_origin"]).ngroups)
        self.assertTrue(all("pca_audit" in fit and "model_audit" in fit for fit in fits))


if __name__ == "__main__":
    unittest.main()
