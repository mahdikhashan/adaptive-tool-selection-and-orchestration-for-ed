"""Inference-only tool orchestration demo. No training, network model fetching or pickle loading."""
from __future__ import annotations

import argparse
import base64
import importlib
import io
import json
import math
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

FEATURES = ["remaining_fraction", "steps_fraction", "has_evaluation", "defect_probability",
            "has_explanation", "is_cropped", "reacquired", "last_failed"]
ACTIONS = ["laya", "crop_zoom", "smol_explain", "reacquire_simulated", "accept", "flag", "abstain"]
SCHEMA = "fedorch-demo-v1"
MAX_BYTES = 12 * 1024 * 1024
MAX_STEPS = 12


def number(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or abs(x) > 1e6:
        raise ValueError("Expected finite numeric weight/value with absolute value <= 1e6")
    return float(x)


def vector(value, size):
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"Expected vector of length {size}")
    return [number(x) for x in value]


class Policy:
    """Safe, bounded JSON actor import. Training provenance is supplied, never verified here."""
    def __init__(self, doc):
        if not isinstance(doc, dict) or set(doc) != {"schema", "features", "actions", "mean", "scale", "layers", "provenance"}:
            raise ValueError("Policy keys must exactly match documented schema")
        if doc["schema"] != SCHEMA or doc["features"] != FEATURES or doc["actions"] != ACTIONS:
            raise ValueError("Policy version, feature order or action order mismatch")
        if not isinstance(doc["provenance"], str) or not 1 <= len(doc["provenance"]) <= 300:
            raise ValueError("Supply a short training provenance string")
        self.mean = vector(doc["mean"], len(FEATURES))
        self.scale = vector(doc["scale"], len(FEATURES))
        if any(x <= 0 for x in self.scale):
            raise ValueError("Normalization scales must be positive")
        layers = doc["layers"]
        if not isinstance(layers, list) or not 1 <= len(layers) <= 4:
            raise ValueError("Expected 1 to 4 dense layers")
        self.layers = []
        width = len(FEATURES)
        for i, layer in enumerate(layers):
            if not isinstance(layer, dict) or set(layer) != {"weight", "bias", "activation"}:
                raise ValueError("Invalid layer fields")
            w = layer["weight"]
            if not isinstance(w, list) or not 1 <= len(w) <= 128:
                raise ValueError("Layer width must be 1..128")
            weights = [vector(row, width) for row in w]
            bias = vector(layer["bias"], len(w))
            final = i == len(layers) - 1
            if layer["activation"] not in (("linear",) if final else ("relu", "tanh")):
                raise ValueError("Hidden activation must be relu/tanh; output must be linear")
            self.layers.append((weights, bias, layer["activation"]))
            width = len(w)
        if width != len(ACTIONS):
            raise ValueError("Output width must match actions")
        self.label = "Imported JSON actor (unverified provenance): " + doc["provenance"]

    def choose(self, observation, allowed):
        x = [(v - m) / s for v, m, s in zip(observation, self.mean, self.scale)]
        for weights, bias, activation in self.layers:
            x = [sum(a * b for a, b in zip(row, x)) + c for row, c in zip(weights, bias)]
            if activation == "relu":
                x = [max(0., a) for a in x]
            elif activation == "tanh":
                x = [math.tanh(a) for a in x]
        if not all(math.isfinite(a) for a in x):
            raise ValueError("Policy produced non-finite logits")
        return max(allowed, key=lambda a: x[ACTIONS.index(a)])


def sync(device):
    torch = sys.modules.get("torch")
    if torch is not None and device == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif torch is not None and device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


class LocalAdapters:
    """Frozen local checkpoints, loaded lazily. HF offline flags are set before imports."""
    def __init__(self, laya=None, smol=None, device="mps", smol_adapter=None):
        self.paths = {"laya": laya, "smol_explain": smol}
        self.device = device
        self.smol_adapter = smol_adapter
        self.models = {}

    def available(self, action):
        return bool(self.paths.get(action))

    def invoke(self, action, image, context):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        path = Path(self.paths[action]).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(f"Checkpoint directory missing: {path}")
        import torch
        load_start = time.perf_counter()
        if action == "laya":
            if action not in self.models:
                from laya.vlm import load_vlm
                self.models[action] = load_vlm(str(path), device=self.device, dtype="fp32")
                self.models[action].model.eval()
                self.models[action].model.requires_grad_(False)
            questions = {"defect": {"type": "noul", "instructions": "Does this power line component have a defect?"},
                         "judgeable": {"type": "noul", "instructions": "Is the image clear enough to judge the condition?"}}
            sync(self.device)
            load_ms = (time.perf_counter() - load_start) * 1000
            inference_start = time.perf_counter()
            with torch.inference_mode():
                result = self.models[action].predict({"image": image, "explanation_context": context}, questions)
            sync(self.device)
            return {"defect_probability": result["answers"]["defect"]["noul"], "raw": result,
                    "timing": {"load_or_lookup_ms": load_ms,
                               "predict_ms": (time.perf_counter() - inference_start) * 1000,
                               "note": "Synchronized Laya predict wall time includes internal image processing"}}
        if action not in self.models:
            from transformers import AutoProcessor, AutoModelForImageTextToText
            processor = AutoProcessor.from_pretrained(str(path), local_files_only=True)
            model = AutoModelForImageTextToText.from_pretrained(str(path), local_files_only=True,
                                                               dtype=torch.float32).to(self.device).eval()
            if self.smol_adapter:
                adapter_path = Path(self.smol_adapter).expanduser().resolve()
                if not adapter_path.is_dir():
                    raise ValueError(f"Smol adapter directory missing: {adapter_path}")
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, str(adapter_path), local_files_only=True).eval()
            model.requires_grad_(False)
            self.models[action] = (processor, model)
        sync(self.device)
        load_ms = (time.perf_counter() - load_start) * 1000
        processor, model = self.models[action]
        processing_start = time.perf_counter()
        messages = [{"role": "user", "content": [{"type": "image", "image": image},
                    {"type": "text", "text": "Describe visible damage and uncertainty in this power line image. Do not invent details."}]}]
        inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                return_dict=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        sync(self.device)
        preprocessing_ms = (time.perf_counter() - processing_start) * 1000
        inference_start = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=96, do_sample=False)
        sync(self.device)
        generate_ms = (time.perf_counter() - inference_start) * 1000
        text = processor.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return {"explanation": text, "timing": {"load_or_lookup_ms": load_ms,
                "preprocessing_ms": preprocessing_ms, "generate_ms": generate_ms}, "warning": "Explanation quality and faithfulness are unvalidated"}


def decode_image(encoded):
    from PIL import Image, ImageOps
    if not isinstance(encoded, str) or len(encoded) > MAX_BYTES:
        raise ValueError("Image payload too large")
    data = base64.b64decode(encoded, validate=True)
    with Image.open(io.BytesIO(data)) as image:
        if image.width * image.height > 20_000_000 or image.width < 2 or image.height < 2:
            raise ValueError("Image must have 4..20M pixels and dimensions >=2")
        image.load()
        return ImageOps.exif_transpose(image).convert("RGB")


def thumbnail(image):
    image = image.copy()
    image.thumbnail((640, 640))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def run_mission(payload, adapters, emit):
    start = time.perf_counter()
    mode = payload.get("mode", "demo")
    if mode not in ("demo", "real"):
        raise ValueError("Mode must be demo or real")
    budget = number(payload.get("budget_seconds", 60))
    if not 1 <= budget <= 3600:
        raise ValueError("Budget must be 1..3600 seconds")
    policy = Policy(payload["policy"]) if payload.get("policy") is not None else None
    image = decode_image(payload["image"])
    original = image.copy()
    trace = []
    p = None
    explanation = ""
    evaluated = False
    cropped = reacquired = last_failed = False
    blocked = set()
    controller_ms = tool_ms = 0.

    def event(row):
        row["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 3)
        trace.append(row)
        emit(row)

    event({"type": "start", "mode": mode, "controller": policy.label if policy else "Rule demo (not trained PPO)",
           "image": thumbnail(image), "notice": "Inference only. Schema provisional. Deadline guard cannot interrupt an in-flight tool."})
    for step in range(MAX_STEPS):
        remaining = budget - (time.perf_counter() - start)
        allowed = ["abstain"]
        if evaluated:
            allowed += ["accept", "flag"]
        if remaining > 0:
            if "laya" not in blocked and (mode == "demo" or adapters.available("laya")):
                allowed += ["laya"]
            if not cropped:
                allowed += ["crop_zoom"]
            if not explanation and "smol_explain" not in blocked and (mode == "demo" or adapters.available("smol_explain")):
                allowed += ["smol_explain"]
            if not reacquired:
                allowed += ["reacquire_simulated"]
        observation = [max(0., remaining / budget), step / MAX_STEPS, float(evaluated), p if p is not None else .5,
                       float(bool(explanation)), float(cropped), float(reacquired), float(last_failed)]
        decision_start = time.perf_counter()
        if policy:
            action = policy.choose(observation, allowed)
        else:
            preferred = ("laya" if not evaluated else "crop_zoom" if not cropped else
                         "smol_explain" if not explanation else "flag" if p >= .5 else "accept")
            action = preferred if preferred in allowed else "abstain"
        decision_ms = (time.perf_counter() - decision_start) * 1000
        controller_ms += decision_ms
        deadline_override = time.perf_counter() - start >= budget
        if deadline_override:
            action = "abstain"
        event({"type": "decision", "step": step, "observation": dict(zip(FEATURES, observation)),
               "allowed": allowed, "action": action, "controller_ms": decision_ms, "deadline_override": deadline_override})
        if action in ("accept", "flag", "abstain"):
            event({"type": "terminal", "action": action, "warning": "accept/flag are controller decisions, not verified ground truth"})
            break
        event({"type": "tool_start", "step": step, "action": action})
        sync(adapters.device)
        tool_start = time.perf_counter()
        last_failed = False
        try:
            if action == "crop_zoom":
                w, h = image.size
                image = image.crop((w // 4, h // 4, w - w // 4, h - h // 4)).resize((w, h))
                cropped = True
                evaluated = False
                p = None
                explanation = ""
                result = {"operation": "Center crop and resize; no new visual evidence", "image": thumbnail(image)}
            elif action == "reacquire_simulated":
                image = original.copy()
                reacquired = True
                cropped = evaluated = False
                p = None
                explanation = ""
                result = {"operation": "SIMULATED reacquisition: reset to same uploaded image; no camera/video", "image": thumbnail(image)}
            elif mode == "demo":
                result = ({"defect_probability": .6, "simulation": "Fixed illustrative output; not model inference"}
                          if action == "laya" else {"explanation": "SIMULATED explanation: possible damage; inspect more closely."})
            else:
                result = adapters.invoke(action, image, explanation)
            if action == "laya":
                p = number(result["defect_probability"])
                if not 0 <= p <= 1:
                    raise ValueError("defect_probability must be in [0,1]")
                evaluated = True
            if action == "smol_explain":
                explanation = result["explanation"]
                if not isinstance(explanation, str) or not explanation or len(explanation) > 10000:
                    raise ValueError("Adapter explanation must be 1..10000 characters")
                # Require a fresh Laya decision using the new explanation.
                evaluated = False
                p = None
            sync(adapters.device)
            elapsed = (time.perf_counter() - tool_start) * 1000
            tool_ms += elapsed
            event({"type": "tool_result", "step": step, "action": action, "result": result, "wall_ms": elapsed,
                   "timing_kind": "measured adapter wall time (includes lazy loading)" if mode == "real" and action in ("laya", "smol_explain") else "measured demo/processing wall time; not model inference",
                   "deadline_exceeded": time.perf_counter() - start > budget})
        except Exception as error:
            elapsed = (time.perf_counter() - tool_start) * 1000
            tool_ms += elapsed
            blocked.add(action)
            if action == "laya":
                evaluated = False
                p = None
            last_failed = True
            event({"type": "tool_error", "step": step, "action": action, "wall_ms": elapsed, "error": str(error)})
    else:
        event({"type": "terminal", "action": "abstain", "warning": "Maximum steps reached"})
    total = (time.perf_counter() - start) * 1000
    summary = {"type": "summary", "mode": mode, "controller_ms": controller_ms, "tools_ms": tool_ms,
               "processing_ms": max(0., total - controller_ms - tool_ms), "total_ms": total,
               "deadline_exceeded": total > budget * 1000,
               "missing_adapters": [a for a in ("laya", "smol_explain") if mode == "real" and not adapters.available(a)]}
    event(summary)
    return trace

HTML = r'''<!doctype html><html><head><meta charset="utf-8"><title>Local tool orchestration demo</title>
<style>body{font:16px system-ui;max-width:1100px;margin:30px auto;padding:0 20px;background:#f6f7fa;color:#192231}label{display:inline-block;margin:10px}button,select,input{font:inherit;padding:6px}#trace{background:#172232;color:#e9f2ff;padding:16px;white-space:pre-wrap;overflow-wrap:anywhere;max-height:600px;overflow:auto}img{max-width:360px;max-height:280px} .notice{background:#fff0ce;padding:15px}h1{font-size:26px}</style></head><body>
<h1>Adaptive tool selection · local image demo</h1>
<p class="notice">Inference only · provisional state/action schema. Default: <b>rule controller + simulated model outputs</b>, not trained PPO. No models are downloaded. Federated PPO clients are drones; the VM can emulate their training. Laya stays frozen.</p>
<label>Image <input id="image" type="file" accept="image/*"></label>
<label>Controller JSON (optional) <input id="policy" type="file" accept=".json,application/json"></label><br>
<label>Mode <select id="mode"><option value="demo">DEMO — simulated model outputs</option><option value="real">REAL — configured local adapters</option></select></label>
<label>Mission budget, seconds <input id="budget" type="number" min="1" max="3600" value="60"></label>
<button id="run">Run mission</button> <button id="save" disabled>Save trace JSON</button>
<p>REAL requires checkpoint paths at server startup. Missing tools are masked and reported. Deadline checks include controller time; an in-flight model call cannot be interrupted, so budgets are soft. Timings include lazy model loading on first use.</p>
<p>Smol explanations are unvalidated, potentially unfaithful context. Reacquire resets the same image: simulation only. Center crop adds no visual evidence.</p>
<img id="preview" alt="Uploaded image"><h2>Live tool trace</h2><p id="status" role="status"></p><pre id="trace"></pre>
<script>
const $=id=>document.getElementById(id); let rows=[];
$('image').onchange=()=>{if($('image').files[0]){const r=new FileReader();r.onload=()=>$('preview').src=r.result;r.readAsDataURL($('image').files[0]);}};
function append(row){rows.push(row);const clean={...row};if(clean.image){$('preview').src=clean.image;delete clean.image;}if(clean.result?.image){$('preview').src=clean.result.image;clean.result={...clean.result,image:'[image preview]'};}$('trace').textContent+=JSON.stringify(clean,null,2)+'\n';$('trace').scrollTop=$('trace').scrollHeight;}
$('run').onclick=async()=>{rows=[];$('trace').textContent='';$('run').disabled=true;$('save').disabled=true;$('status').textContent='Running…';try{const f=$('image').files[0];if(!f)throw Error('Choose an image first.');if(f.size>8*1024*1024)throw Error('Image must be <=8 MiB.');const bytes=new Uint8Array(await f.arrayBuffer());let binary='';for(let i=0;i<bytes.length;i+=8192)binary+=String.fromCharCode(...bytes.subarray(i,i+8192));const pf=$('policy').files[0];if(pf&&pf.size>1024*1024)throw Error('Policy must be <=1 MiB.');const payload={image:btoa(binary),mode:$('mode').value,budget_seconds:Number($('budget').value),policy:pf?JSON.parse(await pf.text()):null};const response=await fetch('/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});if(!response.ok)throw Error(await response.text());const reader=response.body.getReader(), decoder=new TextDecoder();let pending='';for(;;){const {done,value}=await reader.read();pending+=decoder.decode(value||new Uint8Array(),{stream:!done});let idx;while((idx=pending.indexOf('\n'))>=0){const line=pending.slice(0,idx);pending=pending.slice(idx+1);if(line)append(JSON.parse(line));}if(done)break;}$('status').textContent=rows.some(r=>r.type==='error')?'Failed — see trace.':'Complete — inspect mode and timing labels.';}catch(e){$('status').textContent=e.message;}finally{$('run').disabled=false;$('save').disabled=rows.length===0;}};
$('save').onclick=()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(rows,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='agentic-demo-trace.json';a.click();URL.revokeObjectURL(url);};
</script></body></html>'''


def handler_for(adapters):
    class Handler(BaseHTTPRequestHandler):
        def local_request(self):
            expected = f"127.0.0.1:{self.server.server_port}"
            return (self.headers.get("Host") == expected and
                    self.headers.get("Origin", f"http://{expected}") == f"http://{expected}")

        def do_GET(self):
            if not self.local_request():
                self.send_error(403, "Open the printed 127.0.0.1 URL")
                return
            if self.path != "/":
                self.send_error(404)
                return
            data = HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            if not self.local_request():
                self.send_error(403)
                return
            if self.path != "/run":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= MAX_BYTES or self.headers.get("Content-Type") != "application/json":
                    raise ValueError("Expected application/json, <=12 MiB")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("Expected object")
            except (ValueError, UnicodeError) as error:
                self.send_error(400, str(error))
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

            def emit(row):
                self.wfile.write((json.dumps(row, allow_nan=False, default=str) + "\n").encode())
                self.wfile.flush()
            try:
                run_mission(payload, adapters, emit)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as error:
                emit({"type": "error", "error": str(error)})
    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inference-only local image/tool demo. No downloads or training.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--laya-checkpoint", help="Existing local frozen Laya-vision checkpoint directory")
    parser.add_argument("--smol-checkpoint", help="Existing complete fine-tuned SmolVLM checkpoint, or local base directory with --smol-adapter")
    parser.add_argument("--smol-adapter", help="Existing local SmolVLM PEFT/LoRA adapter; requires local --smol-checkpoint base")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--adapter", help="Trusted installed Python module:factory; returns documented adapter interface. Never uploaded code.")
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("port must be 0..65535")
    if args.smol_adapter and not args.smol_checkpoint:
        parser.error("--smol-adapter requires --smol-checkpoint pointing to the local base model")
    adapters = LocalAdapters(args.laya_checkpoint, args.smol_checkpoint, args.device, args.smol_adapter)
    if args.adapter:
        module, factory = args.adapter.split(":", 1)
        adapters = getattr(importlib.import_module(module), factory)(args)
    server = HTTPServer(("127.0.0.1", args.port), handler_for(adapters))
    print(f"Open http://127.0.0.1:{server.server_port} — default is rule/DEMO; Ctrl-C stops", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
