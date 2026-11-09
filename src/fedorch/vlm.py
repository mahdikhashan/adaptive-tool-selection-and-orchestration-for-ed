import io

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MAX_SIDE = 512
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "out_proj", "fc1", "fc2"]


def load_image(path):
    image = Image.open(path).convert("RGB")
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    return image


def jpeg_bytes(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return buffer.tell()


class LetterScorer:
    def __init__(self, model_id, dtype=torch.float32, device="cuda", adapter=None, model=None):
        self.processor = AutoProcessor.from_pretrained(model_id)
        if "SmolVLM" in str(model_id):
            self.processor.image_processor.size = {"longest_edge": MAX_SIDE}
        self.model = model or AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype).to(device).eval()
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter).eval()
        self.dtype = dtype
        self.device = device
        self.letter_ids = {letter: self.token_ids(letter) for letter in "ABCDEFGH"}

    def token_ids(self, letter):
        tokenizer = self.processor.tokenizer
        return sorted({tokenizer.encode(text, add_special_tokens=False)[0] for text in (letter, " " + letter)})

    def batch(self, image, prompt):
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        batch = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                   return_dict=True, return_tensors="pt")
        return {k: v.to(self.device, self.dtype) if v.is_floating_point() else v.to(self.device) for k, v in batch.items()}

    def letter_logits(self, image, prompt, letters):
        logits = self.model(**self.batch(image, prompt)).logits[0, -1].float()
        return torch.stack([torch.logsumexp(logits[self.letter_ids[letter]], 0) for letter in letters])


def connector_of(model):
    return next(module for name, module in model.named_modules() if name.endswith("connector"))


class Int8Split:
    def __init__(self, model):
        self.payload_bytes = 0
        connector_of(model).register_forward_hook(self)

    def __call__(self, module, inputs, output):
        scale = output.abs().amax().clamp(min=1e-8) / 127
        self.payload_bytes = output.numel() + 4
        return (output / scale).round().clamp(-127, 127) * scale
