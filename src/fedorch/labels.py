import re

LABELS = [
    ("无法判断", "cannot-judge"),
    ("状态正常", "good"),
    ("存在缺陷", "defect"),
    ("锈蚀", "rust"),
    ("corros", "corrosion"),
    ("盖帽缺失", "missing-cap"),
    ("鸟巢", "nest"),
    ("撕裂", "torn-up"),
    ("油漆剥落", "peeling-paint"),
]

OPTION = re.compile(r"([A-Z])\.\s*([^\s]+)")


def canonical(option_text):
    for key, label in LABELS:
        if key in option_text:
            return label
    return "unknown:" + option_text


def parse_options(prompt):
    options = prompt.split("选项：")[-1]
    return {letter: canonical(text) for letter, text in OPTION.findall(options)}


def item_label(item):
    prompt = item["messages"][0]["content"][-1]["text"]
    answer = item["messages"][1]["content"].strip()
    return parse_options(prompt).get(answer, "unknown:" + answer)


def item_image(item):
    path = item.get("_image_path") or item.get("image_path") or item["messages"][0]["content"][0]["image"]
    return path.removeprefix("data/raw/")
