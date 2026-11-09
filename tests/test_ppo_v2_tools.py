"""No perception model downloads or GPU are needed for these contract tests."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from PIL import Image
from fedorch.ppo_v2_tools import (CLASSES, export_cache, grid_boxes, letterbox,
                                 normalized, prepare_yolo_dataset, read_jsonl, reorder_probs)


def make_bundle(root, duplicate=True):
    root = Path(root)
    root.mkdir(exist_ok=True)
    images = root / 'images'
    images.mkdir(exist_ok=True)
    rows, labels = [], []
    for index, split in enumerate(('train', 'val', 'test')):
        name = split + '.png'
        Image.new('RGB', (24, 16), color=(80 + index * 30, 100, 120)).save(images / name)
        for question in range(2 if duplicate else 1):
            item_id = split + str(question)
            rows.append({'item_id': item_id, 'source_id': name, 'image': name,
                         'split': split, 'tower': split + 'tower', 'task': 'T3',
                         'client': 0 if split == 'train' else -1})
            labels.append({'item_id': item_id, 'label': 'rust'})
    (root / 'manifest.json').write_text(json.dumps({'schema': 'fedorch-ppo-v2', 'classes': CLASSES,
                                                  'provenance': {}, 'image_root': str(images), 'fraction': 1.}))
    (root / 'items.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    (root / 'labels.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in labels))
    for name in ('yolo.pt', 'laya.pt', 'smol.pt', 'adapter.pt'):
        (root / name).write_text(name)
    return root


def args_for(root, **updates):
    values = dict(bundle=str(root), yolo_checkpoint=str(root / 'yolo.pt'),
                  laya_checkpoint=str(root / 'laya.pt'), smol_base=str(root / 'smol.pt'),
                  smol_adapter=str(root / 'adapter.pt'), cloud_url=None,
                  cloud_model='Qwen/Qwen3-VL-8B-Instruct', cloud_revision=None, cloud_timeout=60.,
                  tools='YOLO_CLS,ZOOM_ROI,ROI_PROPOSE,IMAGE_QUALITY,SMOL_EXPLAIN',
                  enhance=False, device='cpu', imgsz=224, smol_max_tokens=96,
                  profile_limit=0, synthetic_fixture=True)
    values.update(updates)
    return argparse.Namespace(**values)


class Adapters:
    def __init__(self):
        self.calls = {'yolo': 0, 'smol': 0, 'laya': 0}
        self.evidence = []

    def factory(self, name):
        def adapter(image, evidence=None):
            self.calls[name] += 1
            if name == 'smol':
                return {'explanation': 'A visible brown patch; uncertain material condition.'}
            if name == 'laya':
                self.evidence.append(evidence)
            return {'probs': [.1, .5, .1, .1, .1, .1]}
        return lambda: adapter

    def factories(self):
        return {name: self.factory(name) for name in self.calls}


class ToolTests(unittest.TestCase):
    def test_canonical_class_remap_and_reject_invalid(self):
        names = sorted(CLASSES)
        values = [1 if c == 'rust' else 0 for c in names]
        self.assertEqual(reorder_probs(values, dict(enumerate(names))), [0, 1, 0, 0, 0, 0])
        with self.assertRaises(ValueError):
            reorder_probs([1] * 6, ['a'] * 6)
        with self.assertRaises(ValueError):
            normalized([float('nan')] * 6)

    def test_original_only_yolo_dataset_and_split_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_bundle(Path(tmp) / 'bundle')
            output, counts = prepare_yolo_dataset(root, Path(tmp) / 'dataset')
            self.assertEqual(counts, {'train': 1, 'val': 1})
            self.assertFalse((output / 'test').exists())
            self.assertEqual(len(list((output / 'train' / 'rust').glob('*.png'))), 1)
            rows = read_jsonl(root / 'items.jsonl')
            rows[-1]['source_id'] = rows[0]['source_id']
            (root / 'items.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
            with self.assertRaisesRegex(ValueError, 'crosses'):
                prepare_yolo_dataset(root, Path(tmp) / 'other')

    def test_export_is_label_free_deduplicated_resumable_and_compound(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_bundle(Path(tmp) / 'bundle')
            (root / 'labels.jsonl').unlink()  # Cache must never even need ground truth.
            mock = Adapters()
            report = export_cache(args_for(root), mock.factories())
            self.assertEqual(report['unique_images_exported'], 3)
            self.assertEqual(mock.calls['yolo'], 3 * 5 + 1)
            self.assertEqual(mock.calls['smol'], 3 * 5 + 1)
            self.assertEqual(mock.calls['laya'], 3 * 5 * 3 + 1)
            rows = read_jsonl(root / 'observations.jsonl')
            first = next(r for r in rows if r['tool'] == 'INITIAL')
            self.assertEqual(first['view_id'], 0)
            self.assertAlmostEqual(first['duration_s'], (first['tool_ms'] + first['laya_ms']) / 1000)
            self.assertIn('laya_probs', first)
            self.assertFalse(any('label' in row or 'correct' in row for row in rows))
            self.assertTrue(any(e and 'YOLO predicted probabilities' in e for e in mock.evidence))
            self.assertTrue(any(e and any('smol' in k for k in e) for e in mock.evidence))
            before = dict(mock.calls)
            export_cache(args_for(root), mock.factories())
            self.assertEqual(before, mock.calls)
            manifest = json.loads((root / 'manifest.json').read_text())
            self.assertTrue(manifest['cache_complete'])
            self.assertTrue(manifest['synthetic_fixture'])
            self.assertEqual(manifest['latency_profile_source_ids'], ['train.png'])
            self.assertEqual(manifest['tool_latency_profiles']['INITIAL']['calibration_split'], 'train')
            (root / 'yolo.pt').write_text('changed checkpoint')
            with self.assertRaisesRegex(ValueError, 'provenance'):
                export_cache(args_for(root), mock.factories())

    def test_profile_then_full_resume_and_enhance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_bundle(Path(tmp) / 'bundle', duplicate=False)
            mock = Adapters()
            export_cache(args_for(root, profile_limit=1, enhance=True), mock.factories())
            manifest = json.loads((root / 'manifest.json').read_text())
            self.assertFalse(manifest['cache_complete'])
            self.assertTrue(manifest['profile_only'])
            self.assertEqual(manifest['latency_profile_source_ids'], ['train.png'])
            rows = read_jsonl(root / 'observations.jsonl')
            enhancements = [r for r in rows if r['tool'] == 'ENHANCE']
            self.assertEqual({r['view_id'] for r in enhancements}, {5, 6, 7})
            self.assertTrue(all(r['tool_ms'] > 0 and r['laya_ms'] > 0 for r in enhancements))
            export_cache(args_for(root, enhance=True), mock.factories())
            self.assertTrue(json.loads((root / 'manifest.json').read_text())['cache_complete'])

    def test_explicit_missing_optional_model_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_bundle(Path(tmp) / 'bundle')
            with self.assertRaisesRegex(ValueError, 'smol-base'):
                export_cache(args_for(root, smol_base=None), Adapters().factories())
            with self.assertRaisesRegex(ValueError, 'cloud-url'):
                export_cache(args_for(root, tools='CLOUD_QWEN8B'), Adapters().factories())

    def test_train_val_then_test_retains_prior_cache_and_repairs_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_bundle(Path(tmp) / 'bundle', duplicate=False)
            mock = Adapters()
            export_cache(args_for(root, splits='train,val'), mock.factories())
            manifest = json.loads((root / 'manifest.json').read_text())
            self.assertEqual(manifest['cache_complete_for_splits'], ['train', 'val'])
            self.assertFalse(manifest['cache_complete'])
            self.assertEqual({r['item_id'] for r in read_jsonl(root / 'observations.jsonl')}, {'train0', 'val0'})
            with (root / '.ppo_v2_cache' / 'stages.jsonl').open('ab') as stream:
                stream.write(b'{"source_id":')
            export_cache(args_for(root, splits='test'), mock.factories())
            manifest = json.loads((root / 'manifest.json').read_text())
            self.assertTrue(manifest['cache_complete'])
            self.assertEqual(manifest['cache_complete_for_splits'], ['train', 'val', 'test'])
            self.assertEqual({r['item_id'] for r in read_jsonl(root / 'observations.jsonl')},
                             {'train0', 'val0', 'test0'})
            self.assertEqual(manifest['latency_profile_source_ids'], ['train.png'])

    def test_optional_failures_are_recorded_and_do_not_destroy_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = make_bundle(Path(tmp) / 'bundle', duplicate=False)
            mock = Adapters()
            calls = [0]
            def sometimes_fails(image, evidence=None):
                calls[0] += 1
                if calls[0] > 1:
                    raise RuntimeError('Natural simulated failure, not a label')
                return {'explanation': 'Warmup evidence'}
            factories = mock.factories()
            factories['smol'] = lambda: sometimes_fails
            export_cache(args_for(root), factories)
            manifest = json.loads((root / 'manifest.json').read_text())
            self.assertTrue(manifest['cache_complete'])
            self.assertGreater(manifest['cache_counts']['invalid'], 0)
            self.assertEqual(manifest['cache_counts']['mandatory_initial_failures'], 0)
            failed = [r for r in read_jsonl(root / 'observations.jsonl') if not r['valid']]
            self.assertTrue(all(r['tool'] == 'SMOL_EXPLAIN' and r['duration_s'] > 0 for r in failed))
            self.assertTrue(all(r['laya_ms'] == 0 and 'laya_probs' not in r for r in failed))

    def test_letterbox_keeps_whole_image(self):
        image = Image.new('RGB', (40, 10), 'white')
        output = letterbox(image, 40)
        self.assertEqual(output.size, (40, 40))
        self.assertEqual(output.getpixel((0, 0)), (114, 114, 114))
        self.assertEqual(output.getpixel((0, 20)), (255, 255, 255))
        self.assertEqual(len(grid_boxes(image.size)), 4)

    def test_cli_help_does_not_initialize_models(self):
        root = Path(__file__).resolve().parents[1]
        for name in ('ppo_v2_yolo.py', 'ppo_v2_cache.py'):
            result = subprocess.run([sys.executable, str(root / 'scripts' / name), '--help'],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('--bundle', result.stdout)


if __name__ == '__main__':
    unittest.main()
