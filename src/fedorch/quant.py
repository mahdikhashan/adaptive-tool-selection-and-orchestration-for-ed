import tempfile
from pathlib import Path

import torch
from transformers import BitsAndBytesConfig

from fedorch.common import WORK

FORMATS = ["fp32", "fp16", "int8", "nf4"]
STORE = WORK / "models" / "quantized"


def options(fmt):
    if fmt == "int8":
        return {"quantization_config": BitsAndBytesConfig(load_in_8bit=True), "dtype": torch.float32}
    if fmt == "nf4":
        return {"quantization_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                          bnb_4bit_compute_dtype=torch.float32), "dtype": torch.float32}
    return {"dtype": torch.float16 if fmt == "fp16" else torch.float32}


def load(cls, path, fmt, **kwargs):
    return cls.from_pretrained(str(path), device_map={"": 0}, **options(fmt), **kwargs).eval()


def convert(model, cls, fmt, out, extra=()):
    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        for item in extra:
            item.save_pretrained(tmp)
        quantized = load(cls, tmp, fmt)
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
