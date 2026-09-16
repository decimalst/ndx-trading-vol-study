"""Prewritten scoring contracts for the third search wave."""
import unittest
from copy import deepcopy

import numpy as np
import yaml

from src import overnight_search as study


class OvernightSearchContracts(unittest.TestCase):
    def test_frozen_family_budget_and_measurement_threshold(self):
        p = yaml.safe_load(study.PROTOCOL.read_text())
        study.validate(p)
        for section, key, value in [('comparisons', 'cumulative_hypotheses', 6),
                                    ('index', 'effect_threshold_pct', .1),
                                    ('measurement', 'event_threshold', .001),
                                    ('inference', 'bootstrap_draws', 4999)]:
            bad = deepcopy(p)
            bad[section][key] = value
            with self.assertRaises(ValueError):
                study.validate(bad)

    def test_cumulative_family_retains_all_previous_trials(self):
        prior = [{'p_conservative': 1.}] * 72
        rows = [{'p_conservative': .0001}] + [{'p_conservative': 1.}] * 5
        wave, total = study.adjust(prior, rows)
        self.assertAlmostEqual(wave[0], .0006)
        self.assertAlmostEqual(total[0], .0078)
        with self.assertRaises(ValueError):
            study.adjust(prior[:-1], rows)

    def test_primary_gain_cannot_override_measurement_failure(self):
        phases = [{'name': name, 'improvement_pct': 1.,
                   'action_sensitivity': {'n': 200, 'delta': -.01},
                   'stability': [{'delta': -.01}, {'delta': -.02}]} for name in ['development', 'evaluation']]
        row = {'phases': phases, 'p_holm_wave': .001, 'p_holm_cumulative': .002}
        self.assertTrue(study.passes(row))
        for invalid in [dict(row, p_holm_wave=.005), dict(row, p_holm_cumulative=.06)]:
            self.assertFalse(study.passes(invalid))
        bad = deepcopy(row)
        bad['phases'][0]['action_sensitivity']['delta'] = .01
        self.assertFalse(study.passes(bad))

    def test_unflagged_sensitivity_keeps_original_predictions(self):
        result = study.action_sensitivity(np.array([.2, .4, .6]), np.array([.3, .1, .9]), np.array([False, True, False]))
        self.assertEqual(result['n'], 2)
        self.assertAlmostEqual(result['delta'], -.2)
        self.assertAlmostEqual(result['improvement_pct'], 100/3)
        with self.assertRaisesRegex(ValueError, 'INSUFFICIENT_DATA'):
            study.action_sensitivity(np.ones(3), np.ones(3), np.ones(3, bool))

    def test_failure_retains_six_entries_without_null_claim(self):
        result = study.failure_rows(ValueError('INSUFFICIENT_DATA: labels'))
        self.assertEqual(len(result), 6)
        self.assertEqual({r['p_conservative'] for r in result}, {1.})
        self.assertEqual({r['status'] for r in result}, {'INSUFFICIENT_DATA'})
        self.assertTrue(all(r['phases'] == [] for r in result))


if __name__ == '__main__':
    unittest.main()
