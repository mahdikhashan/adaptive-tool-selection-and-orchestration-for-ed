"""Measured-cache S3/A2 replay. This module never invokes perception models.

Training rewards read only the fixed audit-visible training-label mapping.
Held-out labels are loaded lazily, solely through ``evaluation_labels``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

CLASSES = ('good', 'rust', 'corrosion', 'missing_cap', 'bird_nest', 'other')
ACTIONS = ('ACCEPT', 'ABSTAIN', 'YOLO_CLS', 'SMOL_EXPLAIN', 'CLOUD_QWEN8B',
           'ROI_PROPOSE', 'ZOOM_ROI_1', 'ZOOM_ROI_2', 'ZOOM_ROI_3', 'ZOOM_ROI_4',
           'IMAGE_QUALITY', 'ENHANCE_DENOISE', 'ENHANCE_SHARPEN', 'ENHANCE_DEBLUR')
FEATURES = tuple(
    ['laya_' + c for c in CLASSES] +
    ['laya_entropy', 'laya_margin', 'laya_defect', 'laya_valid', 'laya_duration_s'] +
    ['yolo_' + c for c in CLASSES] +
    ['yolo_entropy', 'yolo_margin', 'yolo_valid', 'yolo_age_s'] +
    ['smol_ran', 'explanation_length', 'smol_valid'] +
    ['smol_laya_' + c for c in CLASSES] +
    ['last_' + a for a in ACTIONS] +
    ['action_count', 'tool_latency_s'] + ['visited_roi_' + str(i) for i in range(1, 5)] +
    ['roi_coverage', 'repeated_action', 'tool_failed', 'zoom_count', 'roi_proposed',
     'view_id', 'enhanced', 'budget_s', 'remaining_s', 'remaining_fraction',
     'uninspected_components', 'network_rtt_s', 'network_bandwidth_mbps',
     'network_available', 'free_memory_mb'] +
    ['ready_' + t for t in ('laya', 'yolo', 'smol', 'cloud')] +
    [t + '_' + p for t in ('YOLO_CLS', 'SMOL_EXPLAIN', 'CLOUD_QWEN8B',
                           'ROI_PROPOSE', 'ZOOM_ROI', 'IMAGE_QUALITY', 'ENHANCE')
     for p in ('p50_s', 'p95_s', 'profile_valid')] +
    ['quality_blur', 'quality_contrast', 'quality_brightness', 'quality_compression'])


def _records(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _finite(value, name, minimum=0):
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        raise ValueError(f'{name} must be finite and >= {minimum}')
    return number


def _probabilities(value):
    if value is None:
        return None
    if len(value) != 6:
        raise ValueError('probabilities must have six canonical classes')
    result = tuple(_finite(x, 'probability') for x in value)
    if any(x > 1 for x in result) or not math.isclose(sum(result), 1., abs_tol=1e-4):
        raise ValueError('probabilities must sum to one')
    return result


def _reject_labels(record):
    forbidden = {'label', 'labels', 'ground_truth', 'truth', 'correct', 'correctness', 'target'}
    if isinstance(record, dict):
        if forbidden.intersection(record):
            raise ValueError('labels/correctness are forbidden in items and observations')
        for value in record.values():
            _reject_labels(value)
    elif isinstance(record, list):
        for value in record:
            _reject_labels(value)


class Bundle:
    def __init__(self, path, feedback='audit'):
        if feedback not in ('audit', 'full'):
            raise ValueError('feedback must be audit or full')
        self.path, self.feedback = Path(path), feedback
        self.manifest = json.loads((self.path / 'manifest.json').read_text())
        if self.manifest.get('schema') != 'fedorch-ppo-v2':
            raise ValueError('unsupported cache schema')
        if tuple(self.manifest.get('classes', ())) != CLASSES:
            raise ValueError('cache classes must use canonical six-class order')
        if self.manifest.get('audit_rate') != .05:
            raise ValueError('principal audit manifest must declare audit_rate=0.05')
        self.items = list(_records(self.path / 'items.jsonl'))
        self.audits = json.loads((self.path / 'audits.json').read_text())
        self.observations = {}
        by_id = {}
        tower_splits, source_splits = {}, {}
        for item in self.items:
            _reject_labels(item)
            if item['item_id'] in by_id or item['split'] not in ('train', 'val', 'test'):
                raise ValueError('duplicate item or invalid split')
            by_id[item['item_id']] = item
            for split_map, identifier in ((tower_splits, item['tower']),
                                           (source_splits, item['source_id'])):
                if identifier in split_map and split_map[identifier] != item['split']:
                    raise ValueError('tower/source contamination across splits')
                split_map[identifier] = item['split']
            if item['split'] == 'train':
                audit = self.audits.get(item['item_id'])
                if not audit or not isinstance(audit.get('audited'), bool):
                    raise ValueError('every train item needs a fixed audit record')
                p = _finite(audit['propensity'], 'audit propensity')
                if not 0 < p <= 1:
                    raise ValueError('audit propensity must be in (0,1]')
        for record in _records(self.path / 'observations.jsonl'):
            _reject_labels(record)
            key = (record['item_id'], record['view_id'], record['tool'])
            if key in self.observations or key[0] not in by_id:
                raise ValueError('duplicate or unknown cache observation')
            if not isinstance(key[1], int) or not 0 <= key[1] <= 7:
                raise ValueError('view_id outside bounded original/grid/enhancement views')
            record['duration_s'] = _finite(record['duration_s'], 'duration_s')
            for name in ('probs', 'laya_probs'):
                if name in record:
                    record[name] = _probabilities(record[name])
            for name in ('bytes_up', 'bytes_down', 'laya_ms', 'tool_ms'):
                if record.get(name) is not None:
                    _finite(record[name], name)
            self.observations[key] = record
        self.training_labels = {}
        seen = set()
        for record in _records(self.path / 'labels.jsonl'):
            item_id = record['item_id']
            if item_id not in by_id or item_id in seen:
                raise ValueError('unknown or duplicate label item_id')
            seen.add(item_id)
            if by_id[item_id]['split'] == 'train' and (
                feedback == 'full' or self.audits[item_id]['audited']
            ):
                label = record['label']
                if label not in CLASSES:
                    raise ValueError('noncanonical verified training label')
                self.training_labels[item_id] = label
        for item in self.items:
            if item['split'] == 'train' and (feedback == 'full' or self.audits[item['item_id']]['audited']):
                if item['item_id'] not in self.training_labels:
                    raise ValueError('verified training label missing')
        self._evaluation_labels = None

    @property
    def evaluation_labels(self):
        """Evaluator-only access; no environment method calls this property."""
        if self._evaluation_labels is None:
            ids = {x['item_id'] for x in self.items if x['split'] != 'train'}
            self._evaluation_labels = {}
            for record in _records(self.path / 'labels.jsonl'):
                if record['item_id'] in ids:
                    if record['label'] not in CLASSES:
                        raise ValueError('noncanonical evaluation label')
                    self._evaluation_labels[record['item_id']] = record['label']
            if self._evaluation_labels.keys() != ids:
                raise ValueError('held-out evaluation labels missing')
        return self._evaluation_labels


class InspectionEnv:
    action_count = len(ACTIONS)
    feature_count = len(FEATURES)

    def __init__(self, bundle, items, budget_s=6., feedback='audit', seed=0,
                 network=None, reward=None, mask_mode='hard', policy_version=0,
                 allow_enhance=False):
        if feedback not in ('audit', 'full') or mask_mode not in ('hard', 'p95'):
            raise ValueError('invalid feedback or mask mode')
        if feedback == 'full' and bundle.feedback != 'full':
            raise ValueError('full feedback requires explicitly full-loaded bundle')
        self.bundle, self.items = bundle, list(items)
        if not self.items or len(self.items) > 5:
            raise ValueError('missions contain one to five components')
        if len({x['tower'] for x in self.items}) != 1:
            raise ValueError('grouped mission must share a tower')
        if len({x['source_id'] for x in self.items}) != len(self.items):
            raise ValueError('grouped mission needs unique original source images')
        if len({x['split'] for x in self.items}) != 1:
            raise ValueError('mission cannot mix data splits')
        known = {x['item_id']: x for x in bundle.items}
        if any(known.get(x['item_id']) != x for x in self.items):
            raise ValueError('mission items must match bundle metadata')
        self.budget_s = _finite(budget_s, 'budget_s')
        if not self.budget_s:
            raise ValueError('budget must be positive')
        self.feedback, self.seed = feedback, seed
        self.network = dict(network or {})
        for name in ('rtt_s', 'bandwidth_mbps', 'delay_s'):
            if name in self.network:
                _finite(self.network[name], name)
        if 'bandwidth_mbps' in self.network and not self.network['bandwidth_mbps']:
            raise ValueError('network bandwidth must be positive')
        self.reward_config = dict(lambda_time=1., lambda_bytes=0., lambda_late=10.,
                                  lambda_energy=0., bytes_ref=1_000_000., energy_ref=1.,
                                  subtype_cost=1.)
        self.reward_config.update(reward or {})
        for name, value in self.reward_config.items():
            _finite(value, name)
        if not self.reward_config['bytes_ref'] or not self.reward_config['energy_ref']:
            raise ValueError('reference normalizers must be positive')
        self.mask_mode, self.policy_version = mask_mode, policy_version
        self.allow_enhance = allow_enhance
        # These are fixed training/profiling summaries, never future per-item outcomes.
        profiles = bundle.manifest.get('tool_latency_profiles', {})
        self._latencies = {}
        for tool in ('YOLO_CLS', 'SMOL_EXPLAIN', 'CLOUD_QWEN8B', 'ROI_PROPOSE',
                     'ZOOM_ROI', 'IMAGE_QUALITY', 'ENHANCE'):
            profile = profiles.get(tool)
            if profile and 'p50_s' in profile and 'p95_s' in profile:
                p50 = _finite(profile['p50_s'], 'profile p50_s')
                p95 = _finite(profile['p95_s'], 'profile p95_s')
                if p95 < p50:
                    raise ValueError('profile P95 must be >= P50')
                self._latencies[tool] = (p50, p95, 1.)
            else:
                self._latencies[tool] = (0., 0., 0.)
        for item in self.items:
            if (item['item_id'], 0, 'INITIAL') not in bundle.observations:
                raise ValueError(f"mandatory INITIAL missing for {item['item_id']}")
        self.reset()

    def reset(self):
        self.trace, self.results = [], []
        self.elapsed_s = self.tool_latency_s = 0.
        self.component = 0
        self.total_actions = 0
        self.done = False
        self._pending_duration = self._pending_cost = 0.
        self._late_charged = False
        self.last_action = None
        self._begin_component()
        return self._observation()

    @property
    def item(self):
        return self.items[min(self.component, len(self.items)-1)]

    def _begin_component(self):
        self.view = 0
        self.visited, self.called = set(), set()
        self.zoom_count = self.steps = 0
        self.roi_proposed = self.enhanced = self.smol_ran = self.smol_valid = False
        self.repeated = self.failed = False
        self.laya = self.yolo = self.smol_laya = None
        self.laya_duration_s = self.yolo_at = self.explanation_length = 0.
        self.quality = (0.,)*4
        self._execute('INITIAL', 0)
        # INITIAL already executed this exact original-image compound operation.
        self.called.add((0, 'YOLO_CLS'))

    def _cloud_available(self):
        return bool(self.network.get('available', False)) and (
            self.bundle.manifest.get('cloud_transport_in_duration', False) or
            'delay_s' in self.network or
            ('rtt_s' in self.network and 'bandwidth_mbps' in self.network))

    def _transport(self, record):
        if record['tool'] != 'CLOUD_QWEN8B':
            return 0.
        if self.bundle.manifest.get('cloud_transport_in_duration', False) or (record.get('timing') or {}).get('remote_wall_includes_transport', False):
            return float(self.network.get('delay_s', 0.))
        delay = float(self.network.get('delay_s', self.network.get('rtt_s', 0.)))
        bandwidth = self.network.get('bandwidth_mbps')
        if bandwidth:
            delay += 8*(record.get('bytes_up', 0)+record.get('bytes_down', 0))/(bandwidth*1e6)
        return delay

    def _charge(self, duration, record=None):
        if record and self.reward_config['lambda_bytes'] and record.get('bytes_measurement_valid') is False:
            raise ValueError('byte penalty requires measured traffic; failed cloud traffic is unknown')
        self.elapsed_s += duration
        self._pending_duration += duration
        self._pending_cost += self.reward_config['lambda_time']*duration/self.budget_s
        if record:
            byte_count = record.get('bytes_up', 0)+record.get('bytes_down', 0)
            self._pending_cost += self.reward_config['lambda_bytes']*byte_count/self.reward_config['bytes_ref']
            if self.reward_config['lambda_energy']:
                if record.get('energy_j') is None:
                    raise ValueError('energy penalty requires measured energy_j for every call')
                self._pending_cost += self.reward_config['lambda_energy']*_finite(record['energy_j'], 'energy_j')/self.reward_config['energy_ref']
        if self.elapsed_s > self.budget_s and not self._late_charged:
            self._pending_cost += self.reward_config['lambda_late']
            self._late_charged = True

    def _execute(self, tool, view):
        record = self.bundle.observations[(self.item['item_id'], view, tool)]
        start = self.elapsed_s
        duration = record['duration_s'] + self._transport(record)
        self._charge(duration, record)
        self.tool_latency_s += duration
        self.called.add((view, tool))
        valid = bool(record.get('valid', False))
        self.failed = not valid
        # An unsuccessful call cannot erase the last valid Laya assessment.
        if valid and record.get('laya_probs') is not None:
            self.laya = record['laya_probs']
            self.laya_duration_s = (record.get('laya_ms') or 0.)/1000.
        if tool in ('INITIAL', 'YOLO_CLS') and valid and record.get('probs') is not None:
            self.yolo, self.yolo_at = record['probs'], self.elapsed_s
        if tool == 'SMOL_EXPLAIN':
            self.smol_ran, self.smol_valid = True, valid
            self.explanation_length = len(record.get('explanation', ''))
            self.smol_laya = record.get('laya_probs') if valid else None
        if tool == 'IMAGE_QUALITY' and valid:
            quality = record.get('quality')
            if quality is not None:
                if len(quality) != 4:
                    raise ValueError('quality needs four measured metrics')
                self.quality = tuple(float(x) for x in quality)
        self.trace.append(dict(item_id=self.item['item_id'], tool=tool, view_id=view,
                               start_s=start, end_s=self.elapsed_s, duration_s=duration,
                               cache_duration_s=record['duration_s'], transport_s=self._transport(record),
                               valid=valid, error=record.get('error'), timing=record.get('timing'),
                               bytes_up=record.get('bytes_up', 0), bytes_down=record.get('bytes_down', 0),
                               bytes_measurement_valid=record.get('bytes_measurement_valid', True),
                               byte_measurement=record.get('byte_measurement'),
                               actual_network_bytes=record.get('actual_network_bytes'),
                               energy_j=record.get('energy_j'), policy_version=self.policy_version))
        return valid

    def _target(self, action):
        if action.startswith('ZOOM_ROI_'):
            return 'ZOOM_ROI', int(action.rsplit('_', 1)[1])
        if action.startswith('ENHANCE_'):
            return 'ENHANCE', 5+ACTIONS[11:].index(action)
        return action, 0 if action == 'ROI_PROPOSE' else self.view

    def mask(self):
        result = [False]*len(ACTIONS)
        if self.done:
            return result
        result[1] = True
        if self.elapsed_s >= self.budget_s:
            return result
        result[0] = self.laya is not None
        if self.steps >= 16:
            return result
        portfolio = set(self.bundle.manifest.get('available_tools', ()))
        remaining = self.budget_s-self.elapsed_s
        for i, action in enumerate(ACTIONS[2:], 2):
            tool, view = self._target(action)
            record = self.bundle.observations.get((self.item['item_id'], view, tool))
            if tool not in portfolio or record is None or (view, tool) in self.called:
                continue
            if tool == 'CLOUD_QWEN8B' and not self._cloud_available():
                continue
            if tool == 'ZOOM_ROI' and (not self.roi_proposed or self.zoom_count >= 2 or view in self.visited):
                continue
            if tool == 'ENHANCE' and (not self.allow_enhance or self.enhanced or self.view != 0):
                continue
            if tool == 'ROI_PROPOSE' and self.roi_proposed:
                continue
            # Minimum feasibility must be explicitly measured; no P95 hard mask by default.
            minimum = record.get('min_duration_s', record.get('duration_min_s', 0.))
            threshold = (self._latencies[tool][1] if self.mask_mode == 'p95' else minimum)
            if _finite(threshold, 'minimum/profile duration') > remaining:
                continue
            result[i] = True
        return result

    def _finalize(self, abstained):
        prediction = None if abstained or self.laya is None else CLASSES[max(range(6), key=self.laya.__getitem__)]
        abstained = prediction is None
        item_id = self.item['item_id']
        audit = self.bundle.audits.get(item_id, {'audited': False, 'propensity': 0.})
        verified = self.item['split'] == 'train' and (
            self.feedback == 'full' or audit['audited'])
        error = None
        if abstained:
            self._pending_cost += .5
        elif verified:
            truth = self.bundle.training_labels[item_id]
            error = (10. if truth != 'good' and prediction == 'good' else
                     1. if truth == 'good' and prediction != 'good' else
                     self.reward_config['subtype_cost'] if truth != prediction else 0.)
            self._pending_cost += error/(audit['propensity'] if self.feedback == 'audit' else 1.)
        self.results.append(dict(item_id=item_id, prediction=prediction, abstained=abstained,
                                 elapsed_s=self.elapsed_s, within_deadline=self.elapsed_s <= self.budget_s,
                                 audited=bool(audit['audited']) if self.item['split'] == 'train' else False,
                                 audit_propensity=audit['propensity'] if self.item['split'] == 'train' else None,
                                 observed_error=error, error_observed=error is not None,
                                 policy_version=self.policy_version))
        self.component += 1

    def step(self, action_index, policy_duration_s=0.):
        if self.done:
            raise RuntimeError('episode is complete; reset before step')
        if not isinstance(action_index, int) or not 0 <= action_index < len(ACTIONS):
            raise ValueError('invalid action index')
        stored_mask = self.mask()
        if not stored_mask[action_index]:
            raise ValueError('action is masked')
        action = ACTIONS[action_index]
        previous_results = len(self.results)
        item_id = self.item['item_id']
        self._charge(_finite(policy_duration_s, 'policy_duration_s'))
        self.repeated = self.last_action == action
        self.last_action = action
        self.total_actions += 1
        if self.elapsed_s < self.budget_s:
            if action in ('ACCEPT', 'ABSTAIN'):
                self._finalize(action == 'ABSTAIN')
                if self.component < len(self.items):
                    self._begin_component()
            else:
                tool, view = self._target(action)
                valid = self._execute(tool, view)
                self.steps += 1
                if tool == 'ROI_PROPOSE' and valid:
                    self.roi_proposed = True
                elif tool == 'ZOOM_ROI':
                    self.zoom_count += 1
                    self.visited.add(view)
                    self.view = view
                elif tool == 'ENHANCE':
                    self.enhanced, self.view = True, view
        # Include every unresolved component in terminal cost and coverage accounting.
        if self.elapsed_s >= self.budget_s:
            while self.component < len(self.items):
                self._finalize(True)
        self.done = self.component == len(self.items)
        newly = self.results[previous_results:]
        observed = [r['observed_error'] for r in newly if r['observed_error'] is not None]
        audit = self.bundle.audits.get(item_id, {}) if self.items[0]['split'] == 'train' else {}
        info = dict(duration_s=self._pending_duration, terminal=self.done,
                    policy_version=self.policy_version, observed_error=sum(observed) if observed else None,
                    error_observed=bool(observed), audited=audit.get('audited', False),
                    audit_propensity=audit.get('propensity'), action_mask=stored_mask,
                    action=action, item_id=item_id, elapsed_s=self.elapsed_s,
                    results=newly, trace=list(self.trace))
        reward = -self._pending_cost
        self._pending_cost = self._pending_duration = 0.
        return self._observation(), reward, self.done, info

    @staticmethod
    def _stats(probabilities):
        if probabilities is None:
            return [0.]*6, 0., 0.
        ordered = sorted(probabilities, reverse=True)
        entropy = -sum(p*math.log(p) for p in probabilities if p)
        return list(probabilities), entropy, ordered[0]-ordered[1]

    def _observation(self):
        z, ze, zm = self._stats(self.laya)
        y, ye, ym = self._stats(self.yolo)
        remaining = max(0., self.budget_s-self.elapsed_s)
        resource = self.bundle.manifest.get('resource_state', {})
        ready = resource.get('model_ready', {})
        values = (z + [ze, zm, 1.-z[0] if self.laya else 0., float(self.laya is not None), self.laya_duration_s] +
                  y + [ye, ym, float(self.yolo is not None), self.elapsed_s-self.yolo_at if self.yolo else 0.] +
                  [float(self.smol_ran), self.explanation_length, float(self.smol_valid)] +
                  list(self.smol_laya or (0.,)*6) +
                  [float(self.last_action == a) for a in ACTIONS] +
                  [self.total_actions, self.tool_latency_s] + [float(i in self.visited) for i in range(1, 5)] +
                  [len(self.visited)/4., float(self.repeated), float(self.failed), self.zoom_count,
                   float(self.roi_proposed), self.view, float(self.enhanced), self.budget_s,
                   remaining, remaining/self.budget_s, len(self.items)-self.component,
                   self.network.get('rtt_s', 0.), self.network.get('bandwidth_mbps', 0.),
                   float(self._cloud_available()), resource.get('free_memory_mb', 0.)] +
                  [float(ready.get(t, False)) for t in ('laya', 'yolo', 'smol', 'cloud')] +
                  [x for tool in self._latencies for x in self._latencies[tool]] + list(self.quality))
        assert len(values) == self.feature_count
        return [float(x) for x in values]
