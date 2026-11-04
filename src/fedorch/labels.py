"""ElecVQA-Bench label handling.

ElecVQA-Bench prompts are written in Chinese. Each item stores its answer as an option
letter; this module maps the option text to a canonical English label.
"""

from __future__ import annotations

import re

# Order matters: the first matching key wins.
_LABEL_KEYS: list[tuple[str, str]] = [
    ("无法判断", "cannot-judge"),
    ("状态正常", "good"),
    ("存在缺陷", "defect"),  # binary task (T2): "defect present"
    ("锈蚀", "rust"),
    ("corros", "corrosion"),  # InsPLAD keeps the Portuguese "corrosão"
    ("盖帽缺失", "missing-cap"),
    ("鸟巢", "nest"),
    ("撕裂", "torn-up"),
    ("油漆剥落", "peeling-paint"),
]

_OPTION_RE = re.compile(r"([A-Z])\.\s*([^\s]+)")


def canonical(option_text: str) -> str:
    for key, label in _LABEL_KEYS:
        if key in option_text:
            return label
    return "unknown:" + option_text


def parse_options(prompt_text: str) -> dict[str, str]:
    """Return {letter: canonical label} from an ElecVQA prompt."""
    tail = prompt_text.split("选项：")[-1] if "选项：" in prompt_text else prompt_text
    return {letter: canonical(text) for letter, text in _OPTION_RE.findall(tail)}


def item_label(item: dict) -> str:
    """Canonical ground-truth label of a chat-format split item (split_comparison/*)."""
    text = item["messages"][0]["content"][-1]["text"]
    answer = item["messages"][1]["content"].strip()
    return parse_options(text).get(answer, "unknown:" + answer)


def item_image(item: dict) -> str:
    """Image path relative to the InsPLAD root (the 'data/raw/' prefix is stripped)."""
    path = item.get("_image_path") or item["messages"][0]["content"][0]["image"]
    return path[len("data/raw/"):] if path.startswith("data/raw/") else path
