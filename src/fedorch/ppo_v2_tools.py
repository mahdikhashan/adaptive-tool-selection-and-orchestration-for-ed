"""Frozen perception adapters and resumable, label-free PPO-v2 GPU cache export.

Heavy libraries are imported only when an adapter is constructed. No model weights
are downloaded by this module: model arguments must refer to existing local files.
Tool pretraining is separate from the controller's five-percent audit track.
"""
from __future__ import annotations

import base64
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import time

CLASSES = ('good', 'rust', 'corrosion', 'missing_cap', 'bird_nest', 'other')
DESCRIPTIONS = ('good, no defect', 'rust', 'corrosion', 'missing cap', 'bird nest',
                'other defect (torn part or peeling paint)')
VERSION = 'ppo-v2-native-6-imagenet-yolo-evidence'
DEFAULT_TOOLS = ('YOLO_CLS', 'ROI_PROPOSE', 'ZOOM_ROI', 'IMAGE_QUALITY')
ALL_TOOLS = (*DEFAULT_TOOLS, 'SMOL_EXPLAIN', 'CLOUD_QWEN8B', 'ENHANCE')


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def fingerprint(path):
    """Content identity for local weights/configs; includes symlink target contents."""
    path = Path(path).expanduser().resolve(strict=True)
    files = sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path]
    if not files:
        raise ValueError(f'Empty model directory: {path}')
    digest = hashlib.sha256()
    for file in files:
        digest.update((str(file.relative_to(path)) if path.is_dir() else file.name).encode())
        with file.open('rb') as stream:
            for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                digest.update(chunk)
    return digest.hexdigest()


def normalized(values):
    values = [float(v) for v in values]
    if len(values) != 6 or any(not math.isfinite(v) or v < 0 for v in values) or sum(values) <= 0:
        raise ValueError('Expected six finite, nonnegative class probabilities')
    total = sum(values)
    return [v / total for v in values]


def reorder_probs(values, names):
    names = [names[i] for i in range(len(names))] if isinstance(names, dict) else list(names)
    if len(names) != 6 or set(names) != set(CLASSES):
        raise ValueError(f'Checkpoint must have exactly the six protocol classes, got {names}')
    return normalized([values[names.index(name)] for name in CLASSES])


def yolo_label_space(names):
    names = [names[i] for i in range(len(names))] if isinstance(names, dict) else list(names)
    if len(names) == 6 and set(names) == set(CLASSES):
        return 'condition6'
    if len(names) == 1000 and all(isinstance(name, str) and name for name in names):
        return 'imagenet1000'
    raise ValueError('YOLO must be the native pretrained 1000-class classifier or an optional canonical six-class checkpoint')


def native_yolo_summary(values, names):
    """Native visual context only; never remap ImageNet outputs into defect classes."""
    if yolo_label_space(names) != 'imagenet1000':
        raise ValueError('Native YOLO summary requires exactly1000 ImageNet classes')
    values = [float(value) for value in values]
    if len(values) != 1000 or any(not math.isfinite(value) or value < 0 for value in values) or sum(values) <= 0:
        raise ValueError('Expected1000 finite nonnegative native YOLO probabilities')
    total = sum(values)
    probabilities = [value / total for value in values]
    order = sorted(range(1000), key=lambda index: probabilities[index], reverse=True)
    return {'native_topk': [{'class_id': index, 'label': names[index], 'probability': probabilities[index]}
                            for index in order[:5]],
            'native_entropy': -sum(value * math.log(value) for value in probabilities if value > 0),
            'native_margin': probabilities[order[0]] - probabilities[order[1]],
            'native_class_count': 1000, 'native_valid': True,
            'yolo_label_space': 'imagenet1000', 'yolo_preprocessing': 'ultralytics_official_predict'}


def letterbox(image, size=224):
    """Whole-image deterministic PIL letterbox, shared by training and inference."""
    from PIL import Image, ImageOps
    return ImageOps.pad(image.convert('RGB'), (size, size), method=Image.Resampling.BILINEAR,
                        color=(114, 114, 114), centering=(0.5, 0.5))


class LetterboxTransform:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        import torchvision.transforms.functional as TF
        return TF.to_tensor(letterbox(image, self.size))


def grid_boxes(size):
    width, height = size
    if width < 2 or height < 2:
        raise ValueError('Image must be at least 2x2 for four bounded ROI views')
    return [(0, 0, width // 2, height // 2), (width // 2, 0, width, height // 2),
            (0, height // 2, width // 2, height), (width // 2, height // 2, width, height)]


def enhance(image, view_id):
    """Optional user-requested ablation; outputs never replace original source pixels.

    Wiener deblur assumes a Gaussian PSF (sigma=1, wrap boundary) and K=0.01;
    this is an inverse-filter assumption, not a known camera blur kernel.
    """
    from PIL import Image, ImageFilter
    import numpy as np
    if view_id == 5:
        return image.filter(ImageFilter.MedianFilter(3))
    if view_id == 6:
        return image.filter(ImageFilter.UnsharpMask(radius=1, percent=100, threshold=3))
    if view_id == 7:
        pixels = np.asarray(image, dtype=np.float32) / 255
        h, w = pixels.shape[:2]
        yy = np.fft.fftfreq(h)[:, None]
        xx = np.fft.fftfreq(w)[None, :]
        transfer = np.exp(-2 * np.pi**2 * (xx**2 + yy**2))
        restored = np.fft.ifft2(np.fft.fft2(pixels, axes=(0, 1)) *
                               (transfer / (transfer**2 + .01))[..., None], axes=(0, 1)).real
        return Image.fromarray(np.uint8(np.clip(restored, 0, 1) * 255))
    raise ValueError('Enhancement view must be 5, 6 or 7')


def quality(image):
    """Natural-image proxies, no restoration; compression proxy is block-edge excess."""
    import numpy as np
    gray = np.asarray(image.convert('L'), dtype=np.float32) / 255
    if min(gray.shape) < 3:
        raise ValueError('Quality metrics require at least 3x3 pixels')
    lap = -4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    dx, dy = np.abs(np.diff(gray, axis=1)), np.abs(np.diff(gray, axis=0))
    boundary = [a[:, 7::8].mean() for a in [dx] if a.shape[1] > 7]
    boundary += [dy[7::8].mean()] if dy.shape[0] > 7 else []
    compression = max(0., float(np.mean(boundary) - (dx.mean() + dy.mean()) / 2)) if boundary else 0.
    return [float(lap.var()), float(gray.std()), float(gray.mean()), compression]


def synchronize(device):
    if str(device).startswith('cuda'):
        import torch
        torch.cuda.synchronize(device)


def timed_call(fn, device='cpu'):
    synchronize(device)
    start = time.perf_counter()
    result = fn()
    synchronize(device)
    return result, time.perf_counter() - start


def release_model():
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class YoloAdapter:
    def __init__(self, checkpoint, device='cuda', size=224):
        import torch
        from ultralytics import YOLO
        Path(checkpoint).resolve(strict=True)
        self.wrapper = YOLO(str(checkpoint), task='classify')
        self.model = self.wrapper.model.float().to(device).eval()
        self.model.requires_grad_(False)
        self.device, self.size = device, size
        self.names = self.wrapper.names
        self.label_space = yolo_label_space(self.names)
        self.torch = torch

    def __call__(self, image):
        if self.label_space == 'imagenet1000':
            # Preserve the official checkpoint's prediction transforms as-is.
            # The deterministic custom letterbox is only for the optional
            # separately trained six-class checkpoint, never this native path.
            with self.torch.inference_mode():
                result = self.wrapper.predict(source=image, imgsz=self.size, device=self.device,
                                              half=False, augment=False, verbose=False)[0]
            native = native_yolo_summary(result.probs.data.float().cpu().tolist(), result.names)
            native['timing'] = {key + '_s': float(value) / 1000 for key, value in result.speed.items()
                                if value is not None}
            return native
        tensor, prep_s = timed_call(lambda: LetterboxTransform(self.size)(image).unsqueeze(0).to(self.device), self.device)
        with self.torch.inference_mode():
            result, model_s = timed_call(lambda: self.model(tensor), self.device)
        probs = result[0] if isinstance(result, tuple) else result
        values, post_s = timed_call(lambda: reorder_probs(probs[0].float().cpu().tolist(), self.names), self.device)
        return {'probs': values, 'yolo_label_space': 'condition6',
                'yolo_preprocessing': 'PIL_bilinear_letterbox_RGB114',
                'timing': {'preprocess_s': prep_s, 'model_execution_s': model_s,
                                          'postprocess_s': post_s}}


def four_bit(checkpoint):
    path = Path(checkpoint)
    if (path / 'backbone_quantized').is_dir():
        return True
    config = path / 'config.json'
    return config.is_file() and 'quantization_config' in json.loads(config.read_text())


class LayaAdapter:
    """fp32 checkpoint, or a quant_laya.py folder: nf4 backbone, fp16 compute, fp32 decision head."""

    def __init__(self, checkpoint, device='cuda'):
        import torch
        from laya.vlm import load_vlm
        Path(checkpoint).resolve(strict=True)
        self.precision = 'nf4/fp16' if four_bit(checkpoint) else 'fp32'
        self.agent = load_vlm(str(checkpoint), device=device, dtype='fp32')
        if self.precision == 'nf4/fp16':
            from transformers import AutoModel
            from fedorch import quant
            self.agent.model.encoder = quant.load(AutoModel, Path(checkpoint) / 'backbone_quantized',
                                                  attn_implementation='sdpa')
            self.agent._torch_dtype = lambda: torch.float16
            torch.cuda.empty_cache()
        self.agent.model.eval().requires_grad_(False)
        self.torch = torch

    def __call__(self, image, evidence=None):
        state = {'image': image}
        if evidence is not None:
            state['evidence'] = evidence
        questions = {'condition': {'type': 'choice',
                     'instructions': 'Which condition does this power line component have?',
                     'criteria': dict(zip(CLASSES, DESCRIPTIONS))}}
        with self.torch.inference_mode():
            answer = self.agent.predict(state, questions)['answers']['condition']
        if answer.get('truncated'):
            raise ValueError('Laya input/evidence truncated: ' + str(answer['truncated']))
        return {'probs': normalized([answer['probabilities'][c] for c in CLASSES])}


class SmolAdapter:
    """fp32 base + LoRA adapter, or a quant_smolvlm.py folder (LoRA merged, nf4, fp16 compute)."""

    def __init__(self, base, adapter, device='cuda', max_new_tokens=96):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        Path(base).resolve(strict=True)
        self.processor = AutoProcessor.from_pretrained(base, local_files_only=True)
        self.processor.image_processor.size = {'longest_edge': 512}
        self.precision = 'nf4/fp16' if four_bit(base) else 'fp32'
        if self.precision == 'nf4/fp16':
            if adapter:
                raise ValueError('The 4-bit SmolVLM folder already holds the merged fine-tuned weights; give no adapter')
            from fedorch import quant
            self.model = quant.load(AutoModelForImageTextToText, base)
            self.dtype = torch.float16
        else:
            from peft import PeftModel
            Path(adapter).resolve(strict=True)
            model = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.float32, local_files_only=True)
            self.model = PeftModel.from_pretrained(model, adapter, local_files_only=True).to(device).eval()
            self.dtype = torch.float32
        self.model.requires_grad_(False)
        self.device, self.max_new_tokens, self.torch = device, max_new_tokens, torch

    def __call__(self, image):
        synchronize(self.device)
        started = time.perf_counter()
        prompt = ('Describe visible evidence about this power-line component condition in two short '
                  'sentences. Mention uncertainty. Do not infer invisible details.')
        messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image},
                    {'type': 'text', 'text': prompt}]}]
        inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                     return_dict=True, return_tensors='pt')
        inputs = {k: v.to(self.device, self.dtype) if v.is_floating_point() else v.to(self.device)
                  for k, v in inputs.items()}
        synchronize(self.device)
        prep_s = time.perf_counter() - started
        with self.torch.inference_mode():
            generated, model_s = timed_call(lambda: self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False), self.device)
        started = time.perf_counter()
        text = self.processor.batch_decode(generated[:, inputs['input_ids'].shape[1]:],
                                            skip_special_tokens=True)[0].strip()
        if not text:
            raise ValueError('Empty Smol explanation')
        return {'explanation': text, 'timing': {'preprocess_s': prep_s, 'model_execution_s': model_s,
                                               'postprocess_s': time.perf_counter() - started}}


class CloudAdapter:
    """OpenAI-compatible endpoint, Qwen3-VL-8B only; never loads weights locally."""
    def __init__(self, url, model='Qwen/Qwen3-VL-8B-Instruct', timeout=60, max_new_tokens=64):
        if 'qwen3-vl-8b' not in model.lower():
            raise ValueError('Cloud model must be Qwen3-VL-8B')
        self.url, self.model, self.timeout = url, model, timeout
        self.max_new_tokens = max_new_tokens

    def __call__(self, image):
        import requests
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        data = base64.b64encode(buffer.getvalue()).decode()
        body = {'model': self.model, 'temperature': 0, 'max_tokens': self.max_new_tokens,
                'messages': [{'role': 'user', 'content': [
                    {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + data}},
                    {'type': 'text', 'text': 'Assess the visible power-line component condition. '
                     'Allowed classes: ' + ', '.join(CLASSES) + '. Explain briefly and state uncertainty.'}]}]}
        payload = json.dumps(body).encode()
        headers = {'Content-Type': 'application/json'}
        if os.environ.get('PPO_V2_CLOUD_API_KEY'):
            headers['Authorization'] = 'Bearer ' + os.environ['PPO_V2_CLOUD_API_KEY']
        response = requests.post(self.url, data=payload, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        text = response.json()['choices'][0]['message']['content']
        if not isinstance(text, str) or not text.strip():
            raise ValueError('Empty cloud assessment')
        return {'explanation': text, 'bytes_up': len(payload), 'bytes_down': len(response.content),
                'byte_measurement': 'HTTP body payload only, excludes protocol overhead'}


class LocalQwenAdapter:
    """Frozen Qwen3-VL-8B on the VM, sequential with the other model families.

    NF4 is an explicitly requested cloud-runtime amendment, with FP16 compute;
    Laya and Smol remain FP32. No CPU/disk offload, network traffic, or updates.
    Byte fields describe nominal PNG/text serialization for network simulation,
    never measured network-wire traffic. Actual local network bytes are zero.
    """
    def __init__(self, checkpoint, precision='nf4', device='cuda', max_side=512, max_new_tokens=64):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
        checkpoint = Path(checkpoint).resolve(strict=True)
        config = json.loads((checkpoint / 'config.json').read_text())
        text_config = config.get('text_config', {})
        if (config.get('model_type') != 'qwen3_vl' or text_config.get('hidden_size') != 4096
                or text_config.get('num_hidden_layers') != 36):
            raise ValueError('Local cloud checkpoint must be dense Qwen3-VL-8B (hidden4096, layers36)')
        if str(device) not in ('cuda', 'cuda:0') or not torch.cuda.is_available():
            raise RuntimeError('Local Qwen requires CUDA GPU0; CPU inference/offload is disabled')
        if precision not in ('nf4', 'fp16'):
            raise ValueError('Qwen precision must be nf4 or fp16')
        if precision == 'fp16' and torch.cuda.get_device_properties(0).total_memory < 22 * 1024**3:
            raise RuntimeError('Qwen8B FP16 requires a larger GPU (at least22GiB); use nf4 on2080Ti')
        kwargs = {'dtype': torch.float16, 'device_map': {'': 0},
                  'local_files_only': True, 'attn_implementation': 'sdpa'}
        if precision == 'nf4':
            kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
                llm_int8_enable_fp32_cpu_offload=False)
        self.processor = AutoProcessor.from_pretrained(str(checkpoint), local_files_only=True,
                                                       min_pixels=32 * 32, max_pixels=max_side**2)
        self.model = AutoModelForImageTextToText.from_pretrained(str(checkpoint), **kwargs).eval()
        self.model.requires_grad_(False)
        # Explicit mapping prevents automatic placement; verify no loader changed it.
        if any(parameter.device.type != 'cuda' for parameter in self.model.parameters()):
            raise RuntimeError('Local Qwen contains non-CUDA parameters; refusing CPU/disk offload')
        self.torch, self.device, self.max_side = torch, device, max_side
        self.max_new_tokens = max_new_tokens

    def __call__(self, image):
        from PIL import Image
        synchronize(self.device)
        started = time.perf_counter()
        image = image.convert('RGB').copy()
        image.thumbnail((self.max_side, self.max_side), Image.Resampling.BILINEAR)
        # Qwen smart-resize rounds to 32-pixel patch/merge boundaries. Pad to
        # those boundaries first so even very narrow crops stay <=512 per side.
        size = tuple(max(32, math.ceil(side / 32) * 32) for side in image.size)
        if image.size != size:
            padded = Image.new('RGB', size, (114, 114, 114))
            padded.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
            image = padded
        prompt = ('Assess only visible evidence about this power-line component. Allowed conditions: '
                  + ', '.join(CLASSES) + '. State uncertainty and explain briefly. Do not infer invisible details.')
        messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image},
                                                {'type': 'text', 'text': prompt}]}]
        inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                     return_dict=True, return_tensors='pt')
        inputs = {key: value.to(self.device, self.torch.float16) if value.is_floating_point()
                  else value.to(self.device) for key, value in inputs.items()}
        synchronize(self.device)
        prep_s = time.perf_counter() - started
        with self.torch.inference_mode():
            output, model_s = timed_call(lambda: self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False), self.device)
        started = time.perf_counter()
        text = self.processor.batch_decode(output[:, inputs['input_ids'].shape[1]:],
                                            skip_special_tokens=True)[0].strip()
        if not text:
            raise ValueError('Empty local Qwen assessment')
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        return {'explanation': text, 'bytes_up': buffer.tell() + len(prompt.encode('utf8')),
                'bytes_down': len(text.encode('utf8')), 'actual_network_bytes': 0,
                'byte_measurement': 'nominal bounded PNG+UTF8 prompt/response payload, simulated transport only',
                'timing': {'preprocess_s': prep_s, 'model_execution_s': model_s,
                           'postprocess_s': time.perf_counter() - started,
                           'transport_s': 0., 'remote_wall_includes_transport': False}}


def resolve_image(bundle, item):
    path = Path(item['image']).expanduser()
    if path.is_absolute():
        return path
    manifest = json.loads((Path(bundle) / 'manifest.json').read_text())
    root = Path(manifest.get('image_root', str(Path(bundle).resolve())))
    if not root.is_absolute():
        root = Path(bundle) / root
    return root / path


def prepare_yolo_dataset(bundle, output):
    """Read full labels ONLY for separate frozen-tool training; never controller labels.

    Deduplicates original sources and rejects conflicting labels and split overlap.
    Test labels/images are never placed into the classifier dataset.
    """
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    labels = {r['item_id']: r['label'] for r in read_jsonl(bundle / 'labels.jsonl')}
    items = read_jsonl(bundle / 'items.jsonl')
    sources, image_splits, towers = {}, {}, {}
    for item in items:
        split = item['split']
        source = item['source_id']
        path = resolve_image(bundle, item).resolve(strict=True)
        if source in sources and sources[source]['split'] != split:
            raise ValueError('Source crosses splits: ' + source)
        if path in image_splits and image_splits[path] != split:
            raise ValueError('Image crosses splits: ' + str(path))
        if item['tower'] in towers and towers[item['tower']] != split:
            raise ValueError('Tower crosses splits: ' + str(item['tower']))
        towers[item['tower']], image_splits[path] = split, split
        if split == 'test':
            sources.setdefault(source, {'split': split})
            continue
        label = labels[item['item_id']]
        if label not in CLASSES:
            raise ValueError('Unknown class: ' + str(label))
        if source in sources and sources[source].get('label') != label:
            raise ValueError('Conflicting six-class labels for source: ' + source)
        sources[source] = {'split': split, 'label': label, 'image': str(path)}
    entries = [{**row, 'source_id': source} for source, row in sorted(sources.items()) if row['split'] != 'test']
    signature = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
    manifest = output / 'dataset.json'
    if manifest.exists() and json.loads(manifest.read_text())['signature'] != signature:
        raise ValueError('Existing YOLO dataset differs; choose a fresh output directory')
    for split in ('train', 'val'):
        for name in CLASSES:
            (output / split / name).mkdir(parents=True, exist_ok=True)
    counts = {'train': 0, 'val': 0}
    for entry in entries:
        target = output / entry['split'] / entry['label'] / (
            hashlib.sha256(entry['source_id'].encode()).hexdigest() + Path(entry['image']).suffix.lower())
        if not target.exists():
            target.symlink_to(entry['image'])
        counts[entry['split']] += 1
    if not all(counts.values()):
        raise ValueError('Classifier needs both train and held-out validation images')
    write_json(manifest, {'signature': signature, 'classes': CLASSES, 'counts': counts,
                         'full_label_tool_pretraining': True, 'controller_audit_budget_applies': False,
                         'augmentation': 'none', 'preprocessing': 'PIL bilinear letterbox RGB114 to [0,1]'})
    return output, counts


def train_yolo(args):
    """One training invocation, one reusable frozen best checkpoint. No sweep/seeds."""
    args.output = str(Path(args.output).resolve())
    dataset, counts = prepare_yolo_dataset(args.bundle, Path(args.output) / 'dataset')
    if args.prepare_only:
        return {'dataset': str(dataset), 'counts': counts}
    if str(args.device).lower() not in ('0', 'cuda', 'cuda:0'):
        raise ValueError('Classifier training requires the GPU (--device 0); CPU training is disabled')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required; refusing to train the classifier on CPU')
    Path(args.base).resolve(strict=True)  # explicit local pretrained initialization
    from ultralytics import YOLO
    from ultralytics.data.dataset import ClassificationDataset
    from ultralytics.models.yolo.classify import ClassificationTrainer, ClassificationValidator
    class DeterministicDataset(ClassificationDataset):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.torch_transforms = LetterboxTransform(args.imgsz)
    class DeterministicValidator(ClassificationValidator):
        def build_dataset(self, img_path):
            return DeterministicDataset(root=img_path, args=self.args, augment=False,
                                        prefix=self.args.split, names=self.names)
    class DeterministicTrainer(ClassificationTrainer):
        def build_dataset(self, img_path, mode='train', batch=None):
            return DeterministicDataset(root=img_path, args=self.args, augment=False,
                                        prefix=mode, names=self.data['names'])
        def get_validator(self):
            self.loss_names = ['loss']
            from copy import copy
            return DeterministicValidator(self.test_loader, self.save_dir,
                                          args=copy(self.args), _callbacks=self.callbacks)
    checkpoint = Path(args.output) / 'frozen_yolo.pt'
    if checkpoint.exists():
        raise FileExistsError('Frozen checkpoint already exists; reuse it rather than train again')
    start = time.perf_counter()
    model = YOLO(args.base, task='classify')
    model.train(data=str(dataset), trainer=DeterministicTrainer, epochs=args.epochs,
                imgsz=args.imgsz, batch=args.batch, workers=args.workers, device=args.device,
                project=str(Path(args.output) / 'training'), name='sixclass', exist_ok=False,
                seed=0, deterministic=True, amp=False, plots=False, cache=False,
                fliplr=0., flipud=0., auto_augment=None, erasing=0.,
                hsv_h=0., hsv_s=0., hsv_v=0., patience=args.patience)
    import shutil
    best = Path(model.trainer.best)
    if not best.is_file():
        raise RuntimeError('Training produced no best checkpoint')
    shutil.copyfile(best, checkpoint)
    names = YOLO(str(checkpoint)).names
    reorder_probs([1] * 6, names)
    record = {'checkpoint': str(checkpoint), 'sha256': fingerprint(checkpoint),
              'base_sha256': fingerprint(args.base), 'classes': CLASSES, 'checkpoint_names': names,
              'dataset_manifest_sha256': fingerprint(dataset / 'dataset.json'),
              'canonical_to_checkpoint': [[names[i] for i in range(6)].index(c) for c in CLASSES],
              'preprocessing': 'PIL bilinear letterbox RGB114 to [0,1]', 'imgsz': args.imgsz,
              'augmentation': 'none', 'seed': 0, 'train_unique_images': counts['train'],
              'validation_unique_images': counts['val'], 'test_accessed_for_training': False,
              'full_label_tool_pretraining': True, 'training_wall_s': time.perf_counter() - start}
    write_json(Path(args.output) / 'frozen_yolo.json', record)
    return record


def _image_view(path, view):
    from PIL import Image
    with Image.open(path) as opened:
        image = opened.convert('RGB')
    if view == 0:
        return image, 0.
    if view <= 4:
        return timed_call(lambda: image.crop(grid_boxes(image.size)[view - 1]))
    return timed_call(lambda: enhance(image, view))


def _attempt(fn, device):
    synchronize(device)
    started = time.perf_counter()
    started_unix = time.time()
    try:
        result = fn()
        synchronize(device)
        return {**result, 'valid': True, 'duration_s': time.perf_counter() - started,
                'started_unix_s': started_unix, 'ended_unix_s': time.time()}
    except Exception as error:
        synchronize(device)
        return {'valid': False, 'error': f'{type(error).__name__}: {error}',
                'duration_s': time.perf_counter() - started,
                'started_unix_s': started_unix, 'ended_unix_s': time.time()}


def _percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * (len(values) - 1)))] if values else None


def export_cache(args, factories=None):
    """Two-pass cache: release YOLO/Smol before loading the frozen Laya model.

    Resume is permitted only for identical model, source-content, preprocessing,
    endpoint-identity and hardware signatures. A pilot never marks a bundle complete.
    Injectable factories support smoke tests without importing perception models.
    """
    bundle = Path(args.bundle)
    manifest_path = bundle / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    items = read_jsonl(bundle / 'items.jsonl')
    if not items:
        raise ValueError('No inspection items in bundle')
    requested = set(args.tools.split(',')) if args.tools else set(DEFAULT_TOOLS)
    if getattr(args, 'enhance', False):
        requested.add('ENHANCE')
    if requested - set(ALL_TOOLS):
        raise ValueError('Unknown tools: ' + str(requested - set(ALL_TOOLS)))
    tools = sorted(requested)
    checkpoints = {'yolo': args.yolo_checkpoint, 'laya': args.laya_checkpoint}
    if 'SMOL_EXPLAIN' in requested:
        if not args.smol_base or not (args.smol_adapter or four_bit(args.smol_base)):
            raise ValueError('SMOL_EXPLAIN needs --smol-base plus --smol-adapter, or a merged 4-bit --smol-base')
        checkpoints['smol_base'] = args.smol_base
        if args.smol_adapter:
            checkpoints['smol_adapter'] = args.smol_adapter
    local_cloud = getattr(args, 'cloud_local_checkpoint', None)
    cloud_precision = getattr(args, 'cloud_precision', 'nf4')
    cloud_max_tokens = getattr(args, 'cloud_max_tokens', 64)
    cloud_views = getattr(args, 'cloud_views', 'original')
    if local_cloud and args.cloud_url:
        raise ValueError('Use either --cloud-local-checkpoint or --cloud-url, not both')
    if cloud_views not in ('original', 'all') or cloud_max_tokens < 1:
        raise ValueError('cloud-views must be original|all and cloud-max-tokens positive')
    if 'CLOUD_QWEN8B' in requested:
        if not args.cloud_url and not local_cloud:
            raise ValueError('Requested cloud needs --cloud-local-checkpoint or --cloud-url')
        if local_cloud:
            checkpoints['qwen3_vl_8b'] = local_cloud
    weights = {name: fingerprint(path) for name, path in checkpoints.items()}
    yolo_manifest = Path(args.yolo_checkpoint).with_suffix('.json')
    if yolo_manifest.exists():
        frozen = json.loads(yolo_manifest.read_text())
        if frozen.get('sha256') != weights['yolo'] or frozen.get('imgsz') != args.imgsz:
            raise ValueError('YOLO checkpoint hash or input size differs from its frozen training manifest')
        if list(frozen.get('classes', [])) != list(CLASSES):
            raise ValueError('YOLO frozen manifest class order differs from the protocol')
    # No access to labels/audits in the perception exporter.
    unique = {}
    for item in items:
        source = item['source_id']
        path = resolve_image(bundle, item).resolve(strict=True)
        if source in unique and unique[source] != path:
            raise ValueError('One source_id points to multiple source files: ' + source)
        unique[source] = path
    # Profile training images first; a short pilot must not calibrate from held-out outputs.
    train_sources = {item['source_id'] for item in items if item['split'] == 'train'}
    sources = sorted(unique, key=lambda source: (source not in train_sources, source))
    source_hashes = {source: fingerprint(unique[source]) for source in sources}
    hardware = {'platform': platform.platform(), 'device': args.device,
                'warm_service': True, 'cold_initialization_excluded_from_mission': True}
    versions = {}
    if factories is None:
        import importlib.metadata
        import torch
        hardware.update(gpu=torch.cuda.get_device_name(args.device) if str(args.device).startswith('cuda') else None,
                        torch=str(torch.__version__), cuda=torch.version.cuda)
        for package in ('ultralytics', 'transformers', 'pillow', 'peft', 'laya', 'bitsandbytes', 'accelerate'):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
    edge_dtype = {'yolo': 'fp32', 'laya': 'nf4/fp16' if four_bit(args.laya_checkpoint) else 'fp32',
                  'smol': 'nf4/fp16' if args.smol_base and four_bit(args.smol_base) else 'fp32'}
    signature_data = {'version': VERSION, 'classes': CLASSES, 'weights': weights,
                      'source_hashes': source_hashes, 'tools': tools, 'hardware': hardware,
                      'versions': versions, 'imgsz': args.imgsz, 'edge_dtype': edge_dtype,
                      'yolo_label_space': 'auto:imagenet1000|condition6',
                      'yolo_preprocessing': {'imagenet1000': 'ultralytics_official_predict',
                                             'condition6': 'PIL bilinear letterbox RGB114'},
                      'smol_max_tokens': args.smol_max_tokens, 'cloud_url': args.cloud_url,
                      'cloud_model': args.cloud_model, 'cloud_timeout': args.cloud_timeout,
                      'cloud_revision': args.cloud_revision,
                      'cloud_backend': 'vm_local' if local_cloud else 'http',
                      'cloud_precision': cloud_precision if local_cloud else 'endpoint-declared',
                      'cloud_max_tokens': cloud_max_tokens, 'cloud_views': cloud_views,
                      'cloud_max_side': 512, 'cloud_compute_dtype': 'float16' if local_cloud else None,
                      'cloud_local_preprocessing': 'thumbnail512 then RGB114 pad to multiples32',
                      'enhance': {'median_kernel': 3, 'unsharp': [1, 100, 3],
                                  'wiener_gaussian_sigma': 1, 'wiener_K': .01, 'boundary': 'wrap'},
                      'preprocessing': 'original+4 native grid crops; YOLO preprocessing chosen by native label space',
                      'laya_condition': 'image + only latest tool evidence; no labels/filenames/history'}
    signature = hashlib.sha256(json.dumps(signature_data, sort_keys=True).encode()).hexdigest()
    staging = bundle / '.ppo_v2_cache'
    staging.mkdir(exist_ok=True)
    metadata_path = staging / 'provenance.json'
    if metadata_path.exists() and json.loads(metadata_path.read_text())['signature'] != signature:
        raise ValueError('Cache provenance changed. Use a fresh bundle/cache directory; refusing stale reuse.')
    write_json(metadata_path, {'signature': signature, **signature_data})
    profile_limit = args.profile_limit
    requested_splits = set(getattr(args, 'splits', 'train,val,test').split(','))
    if not requested_splits or requested_splits - {'train', 'val', 'test'}:
        raise ValueError('splits must be drawn from train,val,test')
    split_sources = {item['source_id'] for item in items if item['split'] in requested_splits}
    scheduled = [source for source in sources if source in split_sources]
    selected = scheduled[:profile_limit] if profile_limit else scheduled
    if not selected:
        raise ValueError('No images in requested splits')
    selected_set = set(selected)
    views = list(range(5))
    if factories is None:
        factories = {'yolo': lambda: YoloAdapter(args.yolo_checkpoint, args.device, args.imgsz),
                     'smol': lambda: SmolAdapter(args.smol_base, args.smol_adapter, args.device,
                                                args.smol_max_tokens),
                     'laya': lambda: LayaAdapter(args.laya_checkpoint, args.device),
                     'cloud': (lambda: LocalQwenAdapter(local_cloud, cloud_precision, args.device,
                                                       max_new_tokens=cloud_max_tokens)) if local_cloud else
                              (lambda: CloudAdapter(args.cloud_url, args.cloud_model,
                                                    args.cloud_timeout, cloud_max_tokens))}
    mock = bool(getattr(args, 'synthetic_fixture', False))
    stages_path = staging / 'stages.jsonl'
    stages = {}
    if stages_path.exists():
        # A killed writer may leave a partial last JSON record. Only that suffix
        # is truncated; an invalid complete/interior line is a hard data error.
        with stages_path.open('rb+') as journal:
            offset = 0
            while line := journal.readline():
                try:
                    json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    if line.endswith(b'\n') or journal.read(1):
                        raise ValueError('Corrupt interior cache journal record')
                    journal.seek(offset)
                    journal.truncate()
                    break
                if not line.endswith(b'\n'):
                    journal.write(b'\n')
                    break
                offset = journal.tell()
        for row in read_jsonl(stages_path):
            stages[(row['source_id'], row['view_id'], row['stage'])] = row
    counters, loads = {}, {}
    first_image, _ = _image_view(unique[selected[0]], 0)
    def run_stage(name, entries):
        todo = [(s, v, evidence) for s, v, evidence in entries if (s, v, name) not in stages]
        if not todo:
            return
        stage_device = args.device if name != 'cloud' or local_cloud else 'cpu'
        adapter, load_s = timed_call(factories[name], stage_device)
        loads[name] = {'model_load_s': load_s}
        # Real-input warmup; separate timing, never substitute its output for a cached row.
        warm = _attempt(lambda: adapter(first_image) if name != 'laya' else adapter(first_image, None),
                        stage_device)
        loads[name].update(warmup_s=warm['duration_s'], warmup_valid=warm['valid'])
        if not warm['valid'] and (name != 'cloud' or local_cloud):
            raise RuntimeError(f'{name} warmup failed: {warm["error"]}')
        stage_start = time.perf_counter()
        with stages_path.open('a') as output:
            for index, (source, view, evidence) in enumerate(todo, 1):
                image, view_s = _image_view(unique[source], view)
                if name == 'laya':
                    result = _attempt(lambda: adapter(image, evidence), args.device)
                else:
                    result = _attempt(lambda: adapter(image), stage_device)
                row = {**result, 'source_id': source, 'view_id': view, 'stage': name,
                       'view_processing_s': view_s, 'signature': signature}
                # Laya has multiple evidence conditions keyed by the internal stage key below.
                stages[(source, view, name)] = row
                output.write(json.dumps(row, allow_nan=False) + '\n')
                output.flush()
                if index == 1 or index % 25 == 0 or index == len(todo):
                    elapsed = time.perf_counter() - stage_start
                    print(json.dumps({'stage': name, 'done': index, 'pending': len(todo) - index,
                                      'observed_wall_s': elapsed,
                                      'stage_eta_s': elapsed / index * (len(todo) - index)}), flush=True)
        counters[name] = {'new_calls': len(todo), 'wall_s': time.perf_counter() - stage_start}
        if str(stage_device).startswith('cuda') and not mock:
            import torch
            loads[name]['gpu_memory'] = {'allocated_bytes': torch.cuda.memory_allocated(args.device),
                                         'reserved_bytes': torch.cuda.memory_reserved(args.device),
                                         'process_peak_allocated_bytes': torch.cuda.max_memory_allocated(args.device)}
        del adapter
        if not mock:
            release_model()
    entries = [(source, view, None) for source in selected for view in views]
    run_stage('yolo', entries)
    if 'SMOL_EXPLAIN' in requested:
        run_stage('smol', entries)
    if 'CLOUD_QWEN8B' in requested:
        run_stage('cloud', entries if cloud_views == 'all' else [entry for entry in entries if entry[1] == 0])
    # Laya loaded once; evidence-conditioned passes are independent and disk-resumable.
    laya_entries = []
    for source in selected:
        for view in views:
            laya_entries.append((source, view, 'image', None))
            raw = stages[(source, view, 'yolo')]
            if raw['valid']:
                if raw.get('native_valid'):
                    # Keep full numeric precision in the journal/observations;
                    # concise text evidence avoids wasting Laya's context window.
                    topk = [{**entry, 'probability': round(entry['probability'], 4)}
                            for entry in raw['native_topk']]
                    evidence = {'YOLO native ImageNet visual context (prediction, not condition or ground truth)': topk}
                else:
                    evidence = {'YOLO predicted probabilities': dict(zip(CLASSES, [round(p, 4) for p in raw['probs']]))}
                laya_entries.append((source, view, 'yolo', evidence))
            for tool in ('smol', 'cloud'):
                raw = stages.get((source, view, tool))
                if raw and raw['valid']:
                    laya_entries.append((source, view, tool, {tool + ' visual assessment (prediction, not truth)': raw['explanation']}))
        if 'ENHANCE' in requested:
            laya_entries.extend((source, view, 'image', None) for view in (5, 6, 7))
    todo = [e for e in laya_entries if (e[0], e[1], 'laya_' + e[2]) not in stages]
    if todo:
        adapter, load_s = timed_call(factories['laya'], args.device)
        warm = _attempt(lambda: adapter(first_image, None), args.device)
        if not warm['valid']:
            raise RuntimeError('Laya warmup failed: ' + warm['error'])
        loads['laya'] = {'model_load_s': load_s, 'warmup_s': warm['duration_s']}
        start = time.perf_counter()
        with stages_path.open('a') as output:
            for index, (source, view, condition, evidence) in enumerate(todo, 1):
                image, view_s = _image_view(unique[source], view)
                result = _attempt(lambda: adapter(image, evidence), args.device)
                row = {**result, 'source_id': source, 'view_id': view, 'stage': 'laya_' + condition,
                       'view_processing_s': view_s, 'signature': signature}
                stages[(source, view, row['stage'])] = row
                output.write(json.dumps(row, allow_nan=False) + '\n')
                output.flush()
                if index == 1 or index % 25 == 0 or index == len(todo):
                    elapsed = time.perf_counter() - start
                    print(json.dumps({'stage': 'laya', 'done': index, 'pending': len(todo) - index,
                                      'observed_wall_s': elapsed,
                                      'stage_eta_s': elapsed / index * (len(todo) - index)}), flush=True)
        counters['laya'] = {'new_calls': len(todo), 'wall_s': time.perf_counter() - start}
        if str(args.device).startswith('cuda') and not mock:
            import torch
            loads['laya']['gpu_memory'] = {'allocated_bytes': torch.cuda.memory_allocated(args.device),
                                           'reserved_bytes': torch.cuda.memory_reserved(args.device),
                                           'process_peak_allocated_bytes': torch.cuda.max_memory_allocated(args.device)}
        del adapter
        if not mock:
            release_model()
    # Rebuild from every finished source, retaining earlier train/val when adding test.
    def source_ready(source):
        needed = [(source, view, stage) for view in views for stage in ('yolo', 'laya_image')]
        for view in views:
            for name in ('yolo', 'smol', 'cloud'):
                if name == 'cloud' and view > 0 and cloud_views == 'original':
                    continue
                if name != 'yolo' and {'smol': 'SMOL_EXPLAIN', 'cloud': 'CLOUD_QWEN8B'}[name] not in requested:
                    continue
                raw = stages.get((source, view, name))
                if raw is None:
                    return False
                if raw['valid']:
                    needed.append((source, view, 'laya_' + name))
        if 'ENHANCE' in requested:
            needed += [(source, view, 'laya_image') for view in (5, 6, 7)]
        return all(key in stages for key in needed)
    ready_sources = {source for source in sources if source_ready(source)}
    chosen_items = [item for item in items if item['source_id'] in ready_sources]
    output_path = bundle / 'observations.jsonl'
    tmp = output_path.with_suffix('.jsonl.tmp')
    observations = []
    for item in chosen_items:
        source = item['source_id']
        for view in views:
            image, _ = _image_view(unique[source], view)
            for tool, stage in [('YOLO_CLS', 'yolo'), ('SMOL_EXPLAIN', 'smol'), ('CLOUD_QWEN8B', 'cloud')]:
                if tool not in requested and not (tool == 'YOLO_CLS' and view == 0):
                    continue
                raw = stages.get((source, view, stage))
                downstream = stages.get((source, view, 'laya_' + stage))
                if not raw:
                    continue
                valid = raw['valid'] and bool(downstream and downstream['valid'])
                duration = raw['duration_s'] + (downstream['duration_s'] if downstream else 0.)
                record = {'item_id': item['item_id'], 'view_id': view, 'tool': tool,
                          'valid': valid, 'duration_s': duration, 'tool_ms': raw['duration_s'] * 1000,
                          'laya_ms': downstream['duration_s'] * 1000 if downstream else 0.,
                          'bytes_up': raw.get('bytes_up', 0), 'bytes_down': raw.get('bytes_down', 0),
                          'bytes_measurement_valid': stage != 'cloud' or 'bytes_up' in raw,
                          'byte_measurement': raw.get('byte_measurement', 'no network traffic'),
                          'actual_network_bytes': raw.get('actual_network_bytes'),
                          'wire_bytes_measured': False,
                          'timing': {'tool_s': raw['duration_s'],
                                     'tool_breakdown': raw.get('timing'),
                                     'tool_started_unix_s': raw.get('started_unix_s'),
                                     'tool_ended_unix_s': raw.get('ended_unix_s'),
                                     'laya_s': downstream['duration_s'] if downstream else None,
                                     'laya_started_unix_s': downstream.get('started_unix_s') if downstream else None,
                                     'laya_ended_unix_s': downstream.get('ended_unix_s') if downstream else None,
                                     'laya_includes_preprocessing': True,
                                     'laya_input_condition': stage,
                                     'remote_wall_includes_transport': stage == 'cloud' and not bool(local_cloud),
                                     'cloud_backend': ('vm_local' if local_cloud else 'http') if stage == 'cloud' else None,
                                     'model_start': 'warm', 'energy_j': None},
                          'provenance_signature': signature}
                if raw.get('probs') is not None:
                    record['probs'] = raw['probs']
                for field in ('native_topk', 'native_entropy', 'native_margin', 'native_class_count',
                              'native_valid', 'yolo_label_space', 'yolo_preprocessing'):
                    if field in raw:
                        record[field] = raw[field]
                if raw.get('explanation') is not None:
                    record['explanation'] = raw['explanation']
                if downstream and downstream.get('probs') is not None:
                    record['laya_probs'] = downstream['probs']
                if not valid:
                    record['error'] = raw.get('error') or (downstream or {}).get('error', 'Downstream Laya unavailable')
                if tool in requested:
                    observations.append(record)
                if tool == 'YOLO_CLS' and view == 0:
                    observations.append({**record, 'tool': 'INITIAL'})
            if view > 0 and 'ZOOM_ROI' in requested:
                raw = stages[(source, view, 'laya_image')]
                observations.append({'item_id': item['item_id'], 'view_id': view, 'tool': 'ZOOM_ROI',
                                     'valid': raw['valid'], 'duration_s': raw['duration_s'] + raw['view_processing_s'],
                                     'tool_ms': raw['view_processing_s'] * 1000, 'laya_ms': raw['duration_s'] * 1000,
                                     'laya_probs': raw.get('probs'), 'bytes_up': 0, 'bytes_down': 0,
                                     'error': raw.get('error'), 'timing': {'crop_s': raw['view_processing_s'],
                                      'laya_input_condition': 'image_only', 'model_start': 'warm'},
                                     'provenance_signature': signature})
            for tool in ('IMAGE_QUALITY', 'ROI_PROPOSE'):
                if tool not in requested:
                    continue
                value, duration = timed_call(lambda: quality(image) if tool == 'IMAGE_QUALITY' else grid_boxes(image.size))
                record = {'item_id': item['item_id'], 'view_id': view, 'tool': tool, 'valid': True,
                          'duration_s': duration, 'tool_ms': duration * 1000, 'laya_ms': 0.,
                          'bytes_up': 0, 'bytes_down': 0, 'provenance_signature': signature}
                record['quality' if tool == 'IMAGE_QUALITY' else 'roi_boxes'] = value
                observations.append(record)
        if 'ENHANCE' in requested:
            for view in (5, 6, 7):
                raw = stages[(source, view, 'laya_image')]
                observations.append({'item_id': item['item_id'], 'view_id': view, 'tool': 'ENHANCE',
                                     'valid': raw['valid'], 'duration_s': raw['duration_s'] + raw['view_processing_s'],
                                     'tool_ms': raw['view_processing_s'] * 1000, 'laya_ms': raw['duration_s'] * 1000,
                                     'laya_probs': raw.get('probs'), 'error': raw.get('error'),
                                     'bytes_up': 0, 'bytes_down': 0, 'provenance_signature': signature,
                                     'timing': {'enhance_s': raw['view_processing_s'], 'laya_input_condition': 'image_only',
                                                'model_start': 'warm'},
                                     'enhancement': {5: 'median3', 6: 'unsharp1_100_3', 7: 'wiener_sigma1_K0.01_wrap'}[view]})
    with tmp.open('w') as stream:
        for row in observations:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    tmp.replace(output_path)
    durations = {}
    for name in sorted({r['stage'] for r in stages.values()}):
        rows = [r['duration_s'] for r in stages.values() if r['stage'] == name]
        durations[name] = {'calls': len(rows), 'mean_s': sum(rows) / len(rows),
                           'p50_s': _percentile(rows, .5), 'p95_s': _percentile(rows, .95)}
    total_estimate = 0.
    for name, stats in durations.items():
        multiplier = 3 if name == 'laya_image' and 'ENHANCE' in requested else 0
        call_views = 1 if name in ('cloud', 'laya_cloud') and cloud_views == 'original' else 5 + multiplier
        total_estimate += stats['mean_s'] * len(sources) * call_views
    invalid = sum(not row['valid'] for row in observations)
    profile = {'signature': signature, 'synthetic_fixture': mock,
               'items_exported': len(chosen_items), 'unique_images_exported': len(ready_sources),
               'unique_images_in_bundle': len(sources), 'observations': len(observations),
               'invalid_observations': invalid, 'stages': durations, 'model_loads': loads,
               'this_invocation': counters, 'estimated_full_cache_serial_inference_s': total_estimate,
               'estimate_caveat': 'Linear extrapolation from measured calls; excludes IO, warmup, loading, '
                                  'queue changes, failures and model length variation. Not target mission timing.'}
    write_json(bundle / 'cache_profile.json', profile)
    train_ids = {item['item_id'] for item in chosen_items if item['split'] == 'train'}
    latency_profiles = {}
    for tool in ['INITIAL', *tools]:
        values = [row['duration_s'] for row in observations
                  if row['item_id'] in train_ids and row['tool'] == tool and row['valid']]
        if values:
            latency_profiles[tool] = {'p50_s': _percentile(values, .5), 'p95_s': _percentile(values, .95),
                                      'count': len(values), 'calibration_split': 'train'}
    initial_invalid_ids = {row['item_id'] for row in observations
                           if row['tool'] == 'INITIAL' and not row['valid']}
    complete_splits = []
    for split in ('train', 'val', 'test'):
        split_items = [item for item in items if item['split'] == split]
        if split_items and all(item['source_id'] in ready_sources and item['item_id'] not in initial_invalid_ids
                               for item in split_items):
            complete_splits.append(split)
    manifest.update(available_tools=['INITIAL', *tools],
                    cache_complete=not profile_limit and not initial_invalid_ids and len(ready_sources) == len(sources),
                    cache_complete_for_splits=complete_splits,
                    cache_counts={'items': len(chosen_items), 'unique_images': len(ready_sources),
                                  'observations': len(observations), 'invalid': invalid,
                                  'mandatory_initial_failures': len(initial_invalid_ids)},
                    cache_profile='cache_profile.json', timing_hardware=hardware,
                    cloud_transport_in_duration='CLOUD_QWEN8B' in requested and not bool(local_cloud),
                    cloud_backend='vm_local' if local_cloud else ('http' if 'CLOUD_QWEN8B' in requested else None),
                    cloud_views=cloud_views,
                    cloud_byte_accounting='nominal simulated serialization, no network transfer' if local_cloud else 'HTTP body payload only',
                    profile_only=bool(profile_limit), controller_only_audit_budget=True,
                    enhance='ENHANCE' in requested, tool_latency_profiles=latency_profiles,
                    latency_profile_source_ids=sorted(train_sources & ready_sources))
    manifest.setdefault('provenance', {})['tool_cache'] = {'signature': signature,
        'checkpoint_sha256': weights, 'edge_tool_precision': edge_dtype,
        'qwen_runtime_precision': cloud_precision if local_cloud else None, 'tensorRT': False,
        'yolo_training_performed_by_exporter': False,
        'tool_pretraining_may_use_full_training_labels': True,
        'cache_preprocessing_version': VERSION, 'details': '.ppo_v2_cache/provenance.json',
        'yolo_label_spaces': sorted({row.get('yolo_label_space', 'condition6') for row in stages.values()
                                     if row['stage'] == 'yolo' and row['valid']}),
        'yolo_preprocessing_by_space': signature_data['yolo_preprocessing']}
    if mock:
        manifest['synthetic_fixture'] = True
    write_json(manifest_path, manifest)
    return profile
