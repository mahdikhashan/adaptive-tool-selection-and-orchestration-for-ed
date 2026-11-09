import tempfile
from pathlib import Path

import torch
from transformers import BitsAndBytesConfig

from fedorch.common import WORK

STORE = WORK / "models" / "quantized"
NF4 = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)


def load(cls, path, **kwargs):
    return cls.from_pretrained(str(path), quantization_config=NF4, dtype=torch.float16, device_map={"": 0}, **kwargs).eval()


def convert(model, cls, out, extra=()):
    with tempfile.TemporaryDirectory() as tmp:
        model.half().save_pretrained(tmp)
        for item in extra:
            item.save_pretrained(tmp)
        quantized = load(cls, tmp)
        quantized.save_pretrained(str(out))
    for item in extra:
        item.save_pretrained(str(out))
    del quantized
    torch.cuda.empty_cache()


def weight_mb(model):
    tensors = list(model.parameters()) + list(model.buffers())
    return round(sum(t.numel() * t.element_size() for t in tensors) / 1e6, 1)


def folder_mb(path):
    return round(sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file()) / 1e6, 1)


def quantized_layers(model):
    return sum(type(m).__module__.startswith("bitsandbytes") for m in model.modules())


def allocated_mb():
    return round(torch.cuda.memory_allocated() / 1e6, 1)


def float_dtypes(model):
    return sorted({str(t.dtype).replace("torch.", "") for t in list(model.parameters()) + list(model.buffers()) if t.is_floating_point()})
