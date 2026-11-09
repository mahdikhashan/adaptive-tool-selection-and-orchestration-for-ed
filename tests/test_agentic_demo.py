import base64
import copy
import io
import json
import sys
import threading
import unittest
import urllib.request
from pathlib import Path
from http.server import HTTPServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fedorch.agentic_demo import ACTIONS, FEATURES, SCHEMA, LocalAdapters, Policy, handler_for, main, run_mission


def actor():
    return {"schema": SCHEMA, "features": FEATURES[:], "actions": ACTIONS[:], "mean": [0] * len(FEATURES),
            "scale": [1] * len(FEATURES), "provenance": "unit-test fixture, not trained",
            "layers": [{"weight": [[0] * len(FEATURES) for _ in ACTIONS],
                        "bias": list(range(len(ACTIONS))), "activation": "linear"}]}


def payload(mode="demo"):
    from PIL import Image
    data = io.BytesIO()
    Image.new("RGB", (16, 16), "gray").save(data, format="PNG")
    return {"image": base64.b64encode(data.getvalue()).decode(), "mode": mode, "budget_seconds": 60}


class DemoTests(unittest.TestCase):
    def test_policy_schema_and_mask(self):
        p = Policy(actor())
        self.assertEqual(p.choose([0] * len(FEATURES), ["laya", "crop_zoom"]), "crop_zoom")
        for change in (lambda d: d["features"].reverse(), lambda d: d["actions"].reverse(),
                       lambda d: d["scale"].__setitem__(0, 0), lambda d: d["mean"].__setitem__(0, float("nan")),
                       lambda d: d["layers"][0]["weight"][0].pop(), lambda d: d.update(schema="other"),
                       lambda d: d["layers"][0].update(activation="relu"), lambda d: d.update(python="unsafe")):
            doc = copy.deepcopy(actor())
            change(doc)
            with self.assertRaises(ValueError):
                Policy(doc)

    def test_demo_trace(self):
        rows = []
        run_mission(payload(), LocalAdapters(), rows.append)
        self.assertEqual(rows[0]["mode"], "demo")
        laya = [r for r in rows if r["type"] == "tool_result" and r["action"] == "laya"]
        self.assertEqual(len(laya), 3)
        self.assertTrue(all("not model inference" in r["timing_kind"] for r in laya))
        self.assertEqual(rows[-2]["action"], "flag")
        self.assertIn("total_ms", rows[-1])

    def test_missing_real_adapter_and_errors(self):
        rows = []
        run_mission(payload("real"), LocalAdapters(), rows.append)
        self.assertEqual(rows[-2]["action"], "abstain")
        self.assertEqual(rows[-1]["missing_adapters"], ["laya", "smol_explain"])
        class Failing(LocalAdapters):
            def available(self, action):
                return True
            def invoke(self, *args):
                raise RuntimeError("checkpoint unavailable")
        rows = []
        run_mission(payload("real"), Failing(), rows.append)
        self.assertEqual(sum(r["type"] == "tool_error" for r in rows), 1)
        self.assertEqual(rows[-2]["action"], "abstain")

    def test_context_and_deadline(self):
        class Stub(LocalAdapters):
            def available(self, action):
                return True
            def invoke(self, action, image, context):
                if action == "smol_explain":
                    return {"explanation": "test context"}
                self.context = context
                return {"defect_probability": .2}
        adapters = Stub()
        rows = []
        run_mission(payload("real"), adapters, rows.append)
        self.assertEqual(adapters.context, "test context")
        self.assertEqual(rows[-2]["action"], "accept")
        from unittest.mock import patch
        ticks = iter([0., 0., 0., 2., 2., 2., 2., 2., 2., 2., 2., 2.])
        rows = []
        request = payload()
        request["budget_seconds"] = 1
        with patch("fedorch.agentic_demo.time.perf_counter", side_effect=lambda: next(ticks, 2.)):
            run_mission(request, LocalAdapters(), rows.append)
        self.assertEqual(rows[-2]["action"], "abstain")
        self.assertFalse(any(r["type"] == "tool_start" for r in rows))

    def test_smol_adapter_cli_requires_base(self):
        from contextlib import redirect_stderr
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            main(["--smol-adapter", "/local/adapter"])
        self.assertEqual(error.exception.code, 2)

    def test_local_smol_peft_loading_is_frozen_and_offline(self):
        import tempfile
        from unittest.mock import MagicMock, patch
        torch = MagicMock()
        transformers = MagicMock()
        peft = MagicMock()
        processor = transformers.AutoProcessor.from_pretrained.return_value
        token_ids = MagicMock()
        token_ids.shape = (1, 2)
        processor.apply_chat_template.return_value = {"input_ids": token_ids}
        processor.batch_decode.return_value = ["Visible detail; uncertain damage"]
        with tempfile.TemporaryDirectory() as base, tempfile.TemporaryDirectory() as adapter:
            runtime = LocalAdapters(smol=base, smol_adapter=adapter, device="cpu")
            with patch.dict(sys.modules, {"torch": torch, "transformers": transformers, "peft": peft}):
                result = runtime.invoke("smol_explain", object(), "")
            self.assertEqual(result["explanation"], "Visible detail; uncertain damage")
            self.assertTrue(transformers.AutoProcessor.from_pretrained.call_args.kwargs["local_files_only"])
            self.assertTrue(transformers.AutoModelForImageTextToText.from_pretrained.call_args.kwargs["local_files_only"])
            self.assertTrue(peft.PeftModel.from_pretrained.call_args.kwargs["local_files_only"])
            self.assertEqual(peft.PeftModel.from_pretrained.call_args.args[1], str(Path(adapter).resolve()))
            wrapped = peft.PeftModel.from_pretrained.return_value.eval.return_value
            wrapped.requires_grad_.assert_called_once_with(False)
            wrapped.generate.assert_called_once()
            peft.PeftModel.from_pretrained.return_value.merge_and_unload.assert_not_called()

    def test_http_upload_flow(self):
        server = HTTPServer(("127.0.0.1", 0), handler_for(LocalAdapters()))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}"
        try:
            self.assertIn(b"Live tool trace", urllib.request.urlopen(url).read())
            req = urllib.request.Request(url + "/run", data=json.dumps(payload()).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req) as response:
                rows = [json.loads(line) for line in response]
            self.assertEqual(rows[-1]["type"], "summary")
            self.assertEqual(rows[0]["mode"], "demo")
            imported = payload()
            imported["policy"] = actor()
            req = urllib.request.Request(url + "/run", data=json.dumps(imported).encode(), headers={"Content-Type": "application/json"})
            rows = [json.loads(line) for line in urllib.request.urlopen(req)]
            self.assertIn("Imported JSON actor", rows[0]["controller"])
            self.assertEqual(rows[-2]["action"], "abstain")
            self.assertFalse(any(r["type"] == "tool_start" for r in rows))
            req = urllib.request.Request(url + "/run", data=json.dumps({"image": "not-base64"}).encode(), headers={"Content-Type": "application/json"})
            rows = [json.loads(line) for line in urllib.request.urlopen(req)]
            self.assertEqual(rows[-1]["type"], "error")
        finally:
            server.shutdown()
            thread.join()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
