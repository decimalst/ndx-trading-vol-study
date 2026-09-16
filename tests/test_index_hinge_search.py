"""Prewritten calendar, maturity, inference and decision tests for wave six."""
import copy
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from src import index_hinge as model
from src import index_hinge_search as study


def synthetic(n=1700):
    dates = pd.bdate_range('2010-01-01', periods=n)
    rng = np.random.default_rng(919)
    f = pd.DataFrame(rng.normal(size=(n, len(model.RAW))), index=dates, columns=model.RAW)
    f['const'] = 1.
    f['feature_cutoff_date'] = pd.Series(dates, index=dates).shift(1)
    targets = {}
    for h in (21, 63):
        ends = pd.Series(dates, index=dates).shift(-h)
        targets[h] = pd.DataFrame({'y': rng.normal(size=n)*.01, 'target_end': ends, 'available_date': ends}, index=dates)
        targets[h].loc[ends.isna(), 'y'] = np.nan
    return f, targets


class TestIndexHingeSearch(unittest.TestCase):
    def protocol(self):
        return yaml.safe_load(study.PROTOCOL.read_text())

    def test_fixed_family_and_resolution(self):
        p = self.protocol()
        study.validate(p)
        self.assertEqual(len(study.CONTRASTS), 4)
        self.assertLess(1/(p['inference']['bootstrap_draws']+1), p['wave_alpha']/40)
        for group, key, value in [('index', 'penalty', .02), ('inference', 'hac_lags', 126),
                                  ('inference', 'blocks', [21, 63, 126]), ('index', 'horizons', [21]),
                                  ('comparisons', 'cumulative_hypotheses', 102)]:
            bad = copy.deepcopy(p)
            bad[group][key] = value
            with self.assertRaises(ValueError):
                study.validate(bad)

    def test_training_excludes_overlapping_unmatured_labels(self):
        f, t = synthetic()
        entry = f.index[1200]
        for h in (21, 63):
            mask = study.training_mask(f, t[h], entry, h)
            self.assertEqual(int(mask.sum()), 1200-h)
            self.assertEqual(f.index[mask][-1], f.index[1199-h])
            changed = t[h].copy()
            changed.loc[changed.available_date > f.loc[entry, 'feature_cutoff_date'], 'y'] = 1e50
            pd.testing.assert_series_equal(mask, study.training_mask(f, changed, entry, h))

    def test_maturity_dates_must_match_calendar(self):
        f, t = synthetic()
        t[63].loc[f.index[20], 'available_date'] = f.index[82]
        with self.assertRaises(ValueError):
            study.training_mask(f, t[63], f.index[1200], 63)

    def test_zero_signed_targets_remain_admitted(self):
        f, t = synthetic()
        t[21]['y'] = 0.
        self.assertEqual(study.training_mask(f, t[21], f.index[1200], 21).sum(), 1179)

    def test_month_fit_does_not_follow_future_label_missingness(self):
        f, t = synthetic(1900)
        start, end = f.index[1200], f.index[1530]
        config = self.protocol()['index']
        config.update(origin_start=str(start.date()), origin_end=str(end.date()),
                      development=[str(start.date()), str(f.index[1360].date())],
                      development_target_available_by=str(f.index[1360].date()),
                      evaluation=[str(f.index[1361].date()), str(end.date())], latest_target=str(f.index[-1].date()))
        p1, fits1 = study.forecast_panel(f, t, config)
        month = f.index[1200:1531].to_period('M')
        first = f.index[1200:1531][month != np.roll(month, 1)]
        first = pd.DatetimeIndex(sorted(set(first) | {start}))
        changed = {h: table.copy() for h, table in t.items()}
        for table in changed.values():
            table.loc[first, 'y'] = np.nan
        p2, fits2 = study.forecast_panel(f, changed, config)
        self.assertEqual([(x['horizon'], x['fit_origin']) for x in fits1], [(x['horizon'], x['fit_origin']) for x in fits2])
        for panel in (p1, p2):
            self.assertTrue((panel.train_last_available <= panel.fit_cutoff_date).all())
            self.assertTrue((panel.loc[panel.phase == 'development', 'target_end'] <= f.index[1360]).all())
            self.assertTrue((panel.groupby(['origin', 'horizon']).model.nunique() == 3).all())

    def test_inference_uses_all_long_blocks_and_hac504(self):
        rng = np.random.default_rng(772)
        a, b = rng.uniform(.1, .8, 600), rng.uniform(.2, .9, 600)
        p = self.protocol()
        p['inference']['bootstrap_draws'] = 99
        result = study.paired_inference(a, b, p, 33)
        expected = study.inference.hac_summary(a-b, lags=504)
        self.assertEqual(result['hac504'], expected)
        self.assertEqual(set(result['block_inference']), {'126', '252', '504'})
        self.assertAlmostEqual(result['gain_relative'], (b.mean()-a.mean())/b.mean())
        self.assertEqual(result['p_conservative'], max(expected['p'], *(v['p'] for v in result['block_inference'].values())))
        with self.assertRaisesRegex(ValueError, 'INSUFFICIENT_DATA'):
            study.paired_inference(a[:503], b[:503], p, 33)
        with self.assertRaisesRegex(ValueError, 'INSUFFICIENT_DATA'):
            study.paired_inference(a[:504], b[:504], p, 33)

    def test_decision_requires_both_controls_and_every_offset(self):
        rows = []
        for candidate, control, h in study.CONTRASTS:
            phases = [{'name': phase, 'n': 600, 'gain_relative': .01, 'delta': -.001,
                       'nonoverlap_phases': [{'phase': i, 'n': 10, 'delta': -.001} for i in range(h)],
                       'stability': [{'delta': -.001}, {'delta': -.001}] if phase == 'evaluation' else []}
                      for phase in ['development', 'evaluation']]
            rows.append({'candidate': candidate, 'control': control, 'horizon': h, 'phases': phases,
                         'p_holm_wave': 1e-6, 'p_holm_cumulative': 1e-6})
        self.assertEqual(study.candidate_leads(rows), [21, 63])
        rows[0]['phases'][0]['nonoverlap_phases'][0]['delta'] = 0.
        self.assertEqual(study.candidate_leads(rows), [63])
        with self.assertRaises(ValueError):
            study.candidate_leads(rows[:-1])

    def test_failure_retains_all_four(self):
        result = study.failure_metrics(ValueError('INSUFFICIENT_DATA: fixture'), 'hash')
        self.assertEqual(result['hypothesis_count'], 4)
        self.assertEqual(result['cumulative_hypothesis_count'], 103)
        self.assertEqual(len(result['rows']), 4)
        self.assertTrue(all(row['p_conservative'] == 1 for row in result['rows']))
        self.assertEqual(study.candidate_leads(result['rows']), [])

    def test_inherited_measurement_contrasts_keep_unique_source_identity(self):
        rows = study.inherited(self.protocol())
        self.assertEqual(len(rows), 99)
        self.assertEqual(len({(r['source'], r['source_row_index']) for r in rows}), 99)
        measurement = [r for r in rows if r['source'] == 'reports/measurement_memory/metrics.json']
        self.assertEqual(len(measurement), 15)
        self.assertEqual({r['measure'] for r in measurement}, {'qmle', 'rv5', 'rv15'})

    def test_hac504_matches_explicit_bartlett_covariances(self):
        rng = np.random.default_rng(849)
        d = rng.normal(size=610)
        x = d-d.mean()
        covariance = np.array([np.sum(x[k:]*x[:len(x)-k])/len(x) for k in range(505)])
        variance = covariance[0]+2*np.sum((1-np.arange(1, 505)/505)*covariance[1:])
        self.assertAlmostEqual(study.inference.hac_summary(d, lags=504)['se'], np.sqrt(variance/len(x)), places=13)

    def test_evaluate_pairs_both_controls_and_rejects_label_mutation(self):
        dates = pd.bdate_range('2016-01-04', periods=600).append(pd.bdate_range('2020-01-02', periods=600))
        rows = []
        for h in (21, 63):
            for name, prediction in [('mean', .01), ('baseline', .012), ('hinge', .015)]:
                rows.append(pd.DataFrame({'origin': dates, 'horizon': h, 'model': name, 'prediction': prediction,
                                          'y': .02, 'target_end': dates+pd.Timedelta(days=1),
                                          'available_date': dates+pd.Timedelta(days=1), 'fit_origin': dates,
                                          'feature_cutoff_date': dates-pd.Timedelta(days=1),
                                          'fit_cutoff_date': dates-pd.Timedelta(days=1), 'train_n': 1100,
                                          'train_last_available': dates-pd.Timedelta(days=1),
                                          'phase': np.where(dates.year < 2020, 'development', 'evaluation')}))
        panel = pd.concat(rows, ignore_index=True)
        p = self.protocol()
        p['inference']['bootstrap_draws'] = 99
        p['index']['evaluation_stability'] = [['2020-01-02', '2020-12-31'], ['2021-01-01', '2023-01-01']]
        calendar = pd.bdate_range('2009-01-02', '2025-10-20')
        with patch.object(study, 'inherited', return_value=[{'p_conservative': 1.}]*99):
            result = study.evaluate(panel, calendar, p)
            self.assertEqual(len(result['rows']), 4)
            for row in result['rows']:
                expected = .000025-(.000064 if row['control'] == 'baseline' else .0001)
                for phase in row['phases']:
                    self.assertAlmostEqual(phase['delta'], expected)
                    self.assertEqual(sum(item['n'] for item in phase['nonoverlap_phases']), phase['n'])
            panel.loc[0, 'y'] = .03
            with self.assertRaisesRegex(ValueError, 'Paired'):
                study.evaluate(panel, calendar, p)


if __name__ == '__main__':
    unittest.main()
