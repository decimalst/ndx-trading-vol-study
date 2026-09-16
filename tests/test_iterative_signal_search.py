"""Synthetic search-ledger and inference contracts written before new scores."""
import unittest
from copy import deepcopy

import numpy as np
import pandas as pd
import yaml

from src import iterative_signal_search as study


class SearchContracts(unittest.TestCase):
    def test_protocol_preserves_family_fences_and_practical_thresholds(self):
        p = yaml.safe_load(study.PROTOCOL.read_text())
        study.validate_protocol(p)
        for section, key, value in [('comparisons', 'new_hypotheses', 6),
                                     ('index', 'effect_threshold_pct', 0.01),
                                     ('hf', 'horizons', [1, 5]),
                                     ('inference', 'bootstrap_draws', 4999)]:
            bad = deepcopy(p)
            bad[section][key] = value
            with self.assertRaises(ValueError):
                study.validate_protocol(bad)

    def test_return_loss_accepts_zero_and_negative_actuals(self):
        got = study.loss(np.array([-0.02, 0., .03]), np.array([0., .01, .02]), 'index')
        np.testing.assert_allclose(got, [.0004, .0001, .0001])
        with self.assertRaises(ValueError):
            study.loss(np.array([0.]), np.array([.001]), 'hf')

    def test_cumulative_family_cannot_drop_old_or_failed_hypotheses(self):
        prior = [{'p_conservative': 1.}] * 54
        new = [{'p_conservative': .0001}] + [{'p_conservative': 1.}] * 11
        wave, cumulative = study.adjust(prior, new)
        self.assertAlmostEqual(wave[0], .0012)
        self.assertAlmostEqual(cumulative[0], .0066)
        with self.assertRaises(ValueError):
            study.adjust(prior, new[:-1])

    def test_success_requires_both_phases_effect_size_and_stability(self):
        phases = [{'name': s, 'improvement_pct': 2., 'stability': [{'delta': -.1}, {'delta': -.1}]} for s in ['development', 'evaluation']]
        row = {'study': 'hf', 'phases': phases, 'p_holm_wave': .01, 'p_holm_cumulative': .02}
        self.assertTrue(study.passes(row))
        for bad in [dict(row, p_holm_wave=.03), dict(row, p_holm_cumulative=.06)]:
            self.assertFalse(study.passes(bad))
        bad = deepcopy(row)
        bad['phases'][0]['improvement_pct'] = .5
        self.assertFalse(study.passes(bad))
        bad = deepcopy(row)
        bad['phases'][1]['stability'][1]['delta'] = .01
        self.assertFalse(study.passes(bad))

    def test_equal_losses_are_zero_gap_with_p_one(self):
        p = yaml.safe_load(study.PROTOCOL.read_text())
        r = study.paired_inference(np.ones(200), np.ones(200), p, 101)
        self.assertEqual(r['delta'], 0.)
        self.assertEqual(r['p_conservative'], 1.)

    def test_costs_charge_round_trip_on_each_day_and_zero_when_flat(self):
        got = study.execution_returns(np.array([.01, -.02, .005]), np.array([1, 1, 0]), .0002)
        np.testing.assert_allclose(got, [.0096, -.0204, 0.])
        with self.assertRaises(ValueError):
            study.execution_returns(np.zeros(2), np.ones(3), .0002)

    def test_hf_development_fence_uses_publication_not_target_end(self):
        section = {'development': ['2013-12-01', '2013-12-31'],
                   'development_target_available_by': '2013-12-31'}
        frame = pd.DataFrame({'origin': pd.to_datetime(['2013-12-27', '2013-12-30']),
                              'target_end': pd.to_datetime(['2013-12-30', '2013-12-31']),
                              'available_date': pd.to_datetime(['2013-12-31', '2014-01-02'])})
        got = study.phase_frame(frame, section, 'development')
        self.assertEqual(len(got), 1)
        self.assertEqual(got.origin.iloc[0], pd.Timestamp('2013-12-27'))

    def test_failed_wave_retains_every_hypothesis_without_a_null_claim(self):
        for error, status in [(ValueError('INSUFFICIENT_DATA: too few rows'), 'INSUFFICIENT_DATA'),
                              (KeyError('bad schema'), 'INVALID_RUN')]:
            rows = study.failure_rows(error)
            self.assertEqual(len(rows), 12)
            self.assertEqual({r['p_conservative'] for r in rows}, {1.})
            self.assertEqual({r['status'] for r in rows}, {status})
            self.assertTrue(all(r['phases'] == [] for r in rows))


if __name__ == '__main__':
    unittest.main()
