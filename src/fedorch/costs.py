COST = {
    "frame_interval_s": 4 / 3.45,
    "deadline_s": 4.0,
    "jetson_factor": 9.0,
    "split_edge_fraction": 1.41 / 2.76,
    "rtt_s": 0.05,
    "timeout_s": 3.0,
}

SOURCES = {
    "frame_interval_s": "EVLM Table V: 3.45 key frames per 4 s segment",
    "deadline_s": "one EVLM segment (4 s)",
    "jetson_factor": "GV100 / Jetson Orin NX 16 GB: fp32 16.7 / 1.9 TFLOPS and memory bandwidth 870 / 102 GB/s, both about 9x; sensitivity 5x and 35x (35x = 2.8 s SmolVLM with text generation on Jetson Orin, From Pixels to Semantics)",
    "split_edge_fraction": "split paper: onboard encoder 1.41 s of 2.76 s fully onboard",
    "rtt_s": "assumed 4G round-trip time",
    "timeout_s": "assumed cloud request timeout",
}


def onboard_s(tool, row):
    edge = row["compute_ms"] / 1000 * COST["jetson_factor"]
    return edge * COST["split_edge_fraction"] if tool == "smolvlm_split" else edge


def server_s(tool, row):
    if tool == "cloud":
        return row["compute_ms"] / 1000
    if tool == "smolvlm_split":
        return row["compute_ms"] / 1000 * (1 - COST["split_edge_fraction"])
    return 0.0


def uplink_s(row, mbps):
    if row["payload_bytes"] == 0:
        return 0.0
    if mbps <= 0:
        return None
    seconds = row["payload_bytes"] * 8 / (mbps * 1e6) + COST["rtt_s"]
    return seconds if seconds <= COST["timeout_s"] else None
