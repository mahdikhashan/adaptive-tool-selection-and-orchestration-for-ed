"""Synthetic fixtures exercise protocol boundaries; these are not experiments."""
import json
from pathlib import Path
import tempfile
import unittest

from fedorch.ppo_v2_env import ACTIONS, FEATURES, CLASSES, Bundle, InspectionEnv


class EnvTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.items = [dict(item_id=str(i), split='train', tower='tower', source_id=f'image{i}',
                           image=f'image{i}', client=0, task='T3') for i in range(3)]
        self.audits = {str(i): dict(audited=i == 0, propensity=.05) for i in range(3)}
        self.labels = [dict(item_id=str(i), label='rust') for i in range(3)]
        self.observations = [self.record(str(i), 'INITIAL', duration=.2) for i in range(3)]
        self.manifest = dict(schema='fedorch-ppo-v2', classes=list(CLASSES), audit_rate=.05,
                             synthetic_fixture=True, available_tools=['INITIAL', 'YOLO_CLS',
                             'SMOL_EXPLAIN', 'ZOOM_ROI', 'ROI_PROPOSE', 'IMAGE_QUALITY',
                             'CLOUD_QWEN8B', 'ENHANCE'])

    @staticmethod
    def record(item_id, tool, view=0, duration=.1, prediction=0, **kwargs):
        probs = [0.]*6
        probs[prediction] = 1.
        return dict(item_id=item_id, tool=tool, view_id=view, duration_s=duration,
                    valid=True, probs=probs, laya_probs=probs, laya_ms=50., **kwargs)

    def bundle(self, feedback='audit'):
        for name, records in [('items', self.items), ('labels', self.labels),
                               ('observations', self.observations)]:
            (self.path / (name+'.jsonl')).write_text(''.join(json.dumps(x)+'\n' for x in records))
        (self.path/'manifest.json').write_text(json.dumps(self.manifest))
        (self.path/'audits.json').write_text(json.dumps(self.audits))
        return Bundle(self.path, feedback)

    def env(self, indices=(0,), **kwargs):
        bundle = self.bundle()
        return InspectionEnv(bundle, [bundle.items[i] for i in indices], **kwargs)

    def test_audit_error_weighting_known_costs_and_initial_duration(self):
        env = self.env(reward={'lambda_time': 2.})
        obs, reward, done, info = env.step(ACTIONS.index('ACCEPT'), policy_duration_s=.1)
        self.assertTrue(done)
        self.assertEqual(len(obs), len(FEATURES))
        self.assertAlmostEqual(reward, -200.-2.*.3/6.)
        self.assertAlmostEqual(info['duration_s'], .3)
        self.assertEqual(info['observed_error'], 10.)
        self.assertEqual(env.results[0]['prediction'], 'good')
        self.assertEqual(env.results[0]['policy_version'], 0)
        env = self.env()
        self.assertAlmostEqual(env.step(1)[1], -.5-.2/6.)

    def test_unaudited_truth_cannot_affect_rollout(self):
        env = self.env((1,))
        before = env.reset(), env.step(0)
        self.labels[1]['label'] = 'good'
        env2 = self.env((1,))
        after = env2.reset(), env2.step(0)
        self.assertEqual(before, after)
        self.assertIsNone(after[1][3]['observed_error'])
        self.assertFalse(after[1][3]['error_observed'])
        self.assertEqual(set(env.bundle.training_labels), {'0'})

    def test_heldout_labels_are_not_training_feedback_even_full(self):
        self.items[0]['split'] = 'test'
        self.items[0]['client'] = -1
        self.items[0]['tower'] = 'heldout'
        bundle = self.bundle('full')
        env = InspectionEnv(bundle, [bundle.items[0]], feedback='full')
        self.assertIsNone(bundle._evaluation_labels)
        obs, reward, done, info = env.step(0)
        self.assertIsNone(info['observed_error'])
        self.assertAlmostEqual(reward, -.2/6.)
        self.assertIsNone(bundle._evaluation_labels)
        self.assertEqual(bundle.evaluation_labels, {'0': 'rust'})

    def test_initial_overtime_all_components_abstain_once_late_cost(self):
        self.observations[0]['duration_s'] = 2.1
        env = self.env((0, 1, 2), budget_s=2., policy_version=4)
        self.assertEqual([i for i, valid in enumerate(env.mask()) if valid], [1])
        _, reward, done, info = env.step(1)
        self.assertTrue(done)
        self.assertAlmostEqual(reward, -1.05-10.-1.5)
        self.assertAlmostEqual(info['duration_s'], 2.1)
        self.assertEqual(len(env.results), 3)
        self.assertTrue(all(r['abstained'] for r in env.results))
        self.assertEqual(info['policy_version'], 4)
        self.assertEqual(len(env.trace), 1)

    def test_group_advances_and_charges_next_initial(self):
        env = self.env((1, 2))
        _, reward, done, info = env.step(0)
        self.assertFalse(done)
        self.assertEqual(env.component, 1)
        self.assertAlmostEqual(info['duration_s'], .4)
        self.assertAlmostEqual(reward, -.4/6.)
        _, reward, done, info = env.step(1)
        self.assertTrue(done)
        self.assertAlmostEqual(reward, -.5)
        self.assertEqual([r['item_id'] for r in env.results], ['1', '2'])

    def test_masks_missing_duplicate_inputs_roi_limits_and_new_view(self):
        self.observations += [self.record('0', 'SMOL_EXPLAIN'), self.record('0', 'ROI_PROPOSE')]
        self.observations += [self.record('0', 'ZOOM_ROI', i) for i in range(1, 5)]
        self.observations += [self.record('0', 'YOLO_CLS', 1)]
        env = self.env()
        self.assertFalse(env.mask()[ACTIONS.index('ZOOM_ROI_1')])
        self.assertFalse(env.mask()[ACTIONS.index('YOLO_CLS')])
        env.step(ACTIONS.index('SMOL_EXPLAIN'))
        self.assertFalse(env.mask()[ACTIONS.index('SMOL_EXPLAIN')])
        env.step(ACTIONS.index('ROI_PROPOSE'))
        env.step(ACTIONS.index('ZOOM_ROI_1'))
        self.assertTrue(env.mask()[ACTIONS.index('YOLO_CLS')])
        self.assertFalse(env.mask()[ACTIONS.index('ZOOM_ROI_1')])
        env.step(ACTIONS.index('ZOOM_ROI_2'))
        self.assertFalse(any(env.mask()[ACTIONS.index('ZOOM_ROI_1'):ACTIONS.index('ZOOM_ROI_4')+1]))

    def test_no_p95_mask_in_primary_but_explicit_minimum_masks(self):
        self.observations += [self.record('0', 'SMOL_EXPLAIN', duration=10.)]
        env = self.env(budget_s=2.)
        self.assertTrue(env.mask()[ACTIONS.index('SMOL_EXPLAIN')])
        self.manifest['tool_latency_profiles'] = {'SMOL_EXPLAIN': {'p50_s': 5., 'p95_s': 10.}}
        strict = self.env(budget_s=2., mask_mode='p95')
        self.assertFalse(strict.mask()[ACTIONS.index('SMOL_EXPLAIN')])
        self.observations[-1]['min_duration_s'] = 3.
        self.assertFalse(self.env(budget_s=2.).mask()[ACTIONS.index('SMOL_EXPLAIN')])

    def test_deadline_crossing_abstains_instead_of_accepting_tool_result(self):
        self.observations += [self.record('0', 'SMOL_EXPLAIN', duration=2., prediction=1)]
        env = self.env((0, 1), budget_s=2.)
        _, reward, done, info = env.step(ACTIONS.index('SMOL_EXPLAIN'))
        self.assertTrue(done)
        self.assertTrue(all(x['prediction'] is None for x in env.results))
        self.assertAlmostEqual(reward, -1.1-10.-1.)
        self.assertIsNone(info['observed_error'])

    def test_policy_time_can_exhaust_deadline_without_executing_tool(self):
        self.observations += [self.record('0', 'SMOL_EXPLAIN')]
        env = self.env(budget_s=2.)
        _, _, done, info = env.step(ACTIONS.index('SMOL_EXPLAIN'), policy_duration_s=2.)
        self.assertTrue(done)
        self.assertEqual(len(env.trace), 1)
        self.assertTrue(env.results[0]['abstained'])
        self.assertAlmostEqual(info['duration_s'], 2.2)

    def test_network_explicit_and_bytes_and_time_known_unweighted(self):
        self.observations += [self.record('0', 'CLOUD_QWEN8B', bytes_up=1000, bytes_down=0)]
        self.assertFalse(self.env().mask()[ACTIONS.index('CLOUD_QWEN8B')])
        env = self.env((1,))
        self.assertFalse(env.mask()[ACTIONS.index('CLOUD_QWEN8B')])
        env = self.env(network={'available': True, 'rtt_s': .1, 'bandwidth_mbps': 1.},
                       reward={'lambda_bytes': 1., 'bytes_ref': 1000.})
        _, reward, _, info = env.step(ACTIONS.index('CLOUD_QWEN8B'))
        self.assertAlmostEqual(info['duration_s'], .408)
        self.assertAlmostEqual(reward, -1.-.408/6.)

    def test_full_feedback_explicit_and_subtype_cost(self):
        self.observations[1]['laya_probs'] = [0., 0., 1., 0., 0., 0.]
        bundle = self.bundle('full')
        env = InspectionEnv(bundle, [bundle.items[1]], feedback='full')
        self.assertAlmostEqual(env.step(0)[1], -1.-.2/6.)
        with self.assertRaises(ValueError):
            InspectionEnv(self.bundle(), [self.items[1]], feedback='full')

    def test_enhancement_opt_in_one_per_component_original_grids(self):
        self.observations += [self.record('0', 'ENHANCE', i) for i in (5, 6, 7)]
        self.observations += [self.record('0', 'ROI_PROPOSE', 0), self.record('0', 'ZOOM_ROI', 1)]
        index = ACTIONS.index('ENHANCE_DENOISE')
        self.assertFalse(self.env().mask()[index])
        env = self.env(allow_enhance=True)
        env.step(index)
        self.assertEqual(env.view, 5)
        self.assertFalse(any(env.mask()[11:]))
        env.step(ACTIONS.index('ROI_PROPOSE'))
        env.step(ACTIONS.index('ZOOM_ROI_1'))
        self.assertEqual(env.view, 1)
        self.assertFalse(any(env.mask()[11:]))

    def test_rejects_label_leak_missing_initial_and_invalid_group(self):
        self.observations[0]['label'] = 'rust'
        with self.assertRaises(ValueError):
            self.bundle()
        self.observations[0].pop('label')
        self.observations.pop(0)
        with self.assertRaisesRegex(ValueError, 'INITIAL'):
            self.env()
        self.items[2]['tower'] = 'different'
        with self.assertRaisesRegex(ValueError, 'tower'):
            self.env((1, 2))

    def test_uncalled_future_duration_and_bytes_do_not_change_state_or_mask(self):
        self.observations += [self.record('0', 'CLOUD_QWEN8B', bytes_down=1000)]
        network = {'available': True, 'rtt_s': .1, 'bandwidth_mbps': 1.}
        env = self.env(network=network)
        before = env.reset(), env.mask()
        self.observations[-1]['duration_s'] = 10000.
        self.observations[-1]['bytes_down'] = 1_000_000_000
        other = self.env(network=network)
        self.assertEqual(before, (other.reset(), other.mask()))

    def test_measured_cloud_http_transport_not_double_counted(self):
        self.manifest['cloud_transport_in_duration'] = True
        self.observations += [self.record('0', 'CLOUD_QWEN8B', bytes_up=1000)]
        env = self.env(network={'available': True, 'rtt_s': 1., 'bandwidth_mbps': 1., 'delay_s': .2})
        _, _, _, info = env.step(ACTIONS.index('CLOUD_QWEN8B'))
        self.assertAlmostEqual(info['duration_s'], .5)
        self.assertAlmostEqual(env.trace[-1]['transport_s'], .2)

    def test_hidden_training_label_field_never_inspected(self):
        self.labels[1].pop('label')
        bundle = self.bundle()
        env = InspectionEnv(bundle, [bundle.items[1]])
        self.assertIsNone(env.step(0)[3]['observed_error'])

    def test_failed_optional_output_has_explicit_missing_laya_timing(self):
        self.observations += [dict(item_id='0', view_id=0, tool='SMOL_EXPLAIN',
                                   duration_s=.1, valid=False, laya_ms=None,
                                   error='tool failed before downstream Laya')]
        env = self.env()
        prior = env.laya
        env.step(ACTIONS.index('SMOL_EXPLAIN'))
        self.assertTrue(env.failed)
        self.assertEqual(env.laya, prior)
        self.assertFalse(env.mask()[ACTIONS.index('SMOL_EXPLAIN')])

    def test_failed_cloud_traffic_unknown_preserved_and_not_priced_as_zero(self):
        self.manifest['cloud_transport_in_duration'] = True
        self.observations += [dict(item_id='0', view_id=0, tool='CLOUD_QWEN8B',
                                   duration_s=.1, valid=False, laya_ms=0., bytes_up=0,
                                   bytes_down=0, bytes_measurement_valid=False,
                                   error='connection failed after request sent')]
        env = self.env(network={'available': True})
        env.step(ACTIONS.index('CLOUD_QWEN8B'))
        self.assertFalse(env.trace[-1]['bytes_measurement_valid'])
        priced = self.env(network={'available': True}, reward={'lambda_bytes': 1.})
        with self.assertRaisesRegex(ValueError, 'traffic is unknown'):
            priced.step(ACTIONS.index('CLOUD_QWEN8B'))

    def test_frozen_audit_repeated_resets(self):
        env = self.env((1,))
        for _ in range(5):
            env.reset()
            self.assertFalse(env.step(0)[3]['audited'])
            self.assertEqual(env.bundle.audits['1'], self.audits['1'])


if __name__ == '__main__':
    unittest.main()
