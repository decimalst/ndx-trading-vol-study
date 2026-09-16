"""Wave-two scoring contracts declared before international fits or scores."""
import unittest
from copy import deepcopy

import numpy as np
import pandas as pd
import yaml

from src import international_search as study


class InternationalSearchTests(unittest.TestCase):
    def test_fixed_protocol_family_and_wave_budget(self):
        protocol = yaml.safe_load(study.PROTOCOL.read_text())
        study.validate(protocol)
        for section, key, value in [('comparisons', 'cumulative_hypotheses', 6),
                                    ('hf', 'pca_components', 4),
                                    ('international', 'max_extra_us_sessions', 5)]:
            altered = deepcopy(protocol)
            altered[section][key] = value
            with self.assertRaises(ValueError):
                study.validate(altered)

    def test_no_previous_or_failed_contrasts_disappear(self):
        prior = [{'p_conservative': 1.}] * 66
        rows = [{'p_conservative': .0001}] + [{'p_conservative': 1.}] * 5
        wave, cumulative = study.adjust(prior, rows)
        self.assertAlmostEqual(wave[0], .0006)
        self.assertAlmostEqual(cumulative[0], .0072)
        with self.assertRaises(ValueError):
            study.adjust(prior, rows[:-1])

    def test_lead_gate_uses_second_wave_alpha_and_both_phases(self):
        row = {'phases': [{'improvement_pct': 2., 'name': 'development'},
                          {'improvement_pct': 2., 'name': 'evaluation',
                           'stability': [{'delta': -.1}, {'delta': -.2}]}],
               'p_holm_wave': .005, 'p_holm_cumulative': .01}
        self.assertTrue(study.passes(row))
        self.assertFalse(study.passes(dict(row, p_holm_wave=.01)))
        changed = deepcopy(row)
        changed['phases'][0]['improvement_pct'] = -.1
        self.assertFalse(study.passes(changed))

    def test_failure_bookkeeping_has_no_null_claim(self):
        rows = study.failure_rows(ValueError('INSUFFICIENT_DATA: rank'))
        self.assertEqual(len(rows), 6)
        self.assertEqual({row['p_conservative'] for row in rows}, {1.})
        self.assertEqual({row['status'] for row in rows}, {'INSUFFICIENT_DATA'})
        self.assertTrue(all(row['phases'] == [] for row in rows))

    def test_diagnostics_use_original_session_positions(self):
        calendar = pd.bdate_range('2014-01-01', periods=30)
        origins = calendar[[2, 4, 8, 11, 17, 26]]
        differences = np.array([-1., 2., 3., -4., 5., -6.])
        got = study.diagnostics(origins, differences, calendar, 3, [])
        self.assertEqual([r['n'] for r in got['nonoverlap_phases']], [0, 1, 5])
        self.assertIsNone(got['nonoverlap_phases'][0]['delta'])
        self.assertEqual(got['nonoverlap_phases'][1]['delta'], 2.)
        self.assertEqual(got['nonoverlap_phases'][2]['delta'], -.6)

    def test_inference_rejects_horizons_with_different_origins(self):
        p = yaml.safe_load(study.PROTOCOL.read_text())
        records = []
        for date in pd.to_datetime(['2010-01-04', '2010-01-05', '2014-01-03']):
            for horizon in [1, 5, 21]:
                for model in ['baseline', 'regional', 'latent']:
                    records.append({'origin': date, 'horizon': horizon, 'model': model,
                                    'target_end': date + pd.Timedelta(days=1),
                                    'available_date': date + pd.Timedelta(days=2)})
        panel = pd.DataFrame(records)
        study.validate_panel(panel, p)
        bad = panel.loc[~((panel.horizon == 5) & (panel.origin == '2010-01-05'))]
        with self.assertRaises(ValueError):
            study.validate_panel(bad, p)
        with self.assertRaisesRegex(ValueError, 'INSUFFICIENT_DATA'):
            study.evaluate(panel.assign(y=.001, prediction=.001), pd.DatetimeIndex(panel.origin.unique()), p)


if __name__ == '__main__':
    unittest.main()
