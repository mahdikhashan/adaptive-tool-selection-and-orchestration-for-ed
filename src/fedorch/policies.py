def always(tool):
    def policy(item, history, state):
        if not history:
            return tool
        if tool != "classifier" and len(history) == 1 and not history[0][2]:
            return "classifier"
        return None
    return policy


def split_paper_rule(high_mbps=10.0):
    def policy(item, history, state):
        if not history:
            return "cloud" if state["mbps"] >= high_mbps else "smolvlm_local"
        if len(history) == 1 and not history[0][2]:
            return "classifier"
        return None
    return policy


def cascade(threshold, choose_target):
    def policy(item, history, state):
        if not history:
            return "classifier"
        if len(history) == 1 and history[0][1]["confidence"] < threshold:
            return choose_target(item, state)
        return None
    return policy


def to_local(item, state):
    return "smolvlm_local"


def to_cloud_if_link(min_mbps):
    def choose(item, state):
        return "cloud" if state["mbps"] >= min_mbps else "smolvlm_local"
    return choose


def by_task(min_mbps):
    def choose(item, state):
        return "cloud" if item["task"] == "T2" and state["mbps"] >= min_mbps else "smolvlm_local"
    return choose


def oracle(cache):
    def policy(item, history, state):
        rows = cache[item["item_id"]]
        if not history:
            return "classifier"
        if len(history) > 1 or rows["classifier"]["correct"]:
            return None
        if rows["smolvlm_local"]["correct"]:
            return "smolvlm_local"
        if rows["cloud"]["correct"] and state["mbps"] > 0:
            return "cloud"
        return None
    return policy


THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]


def baselines(cache):
    fixed = {
        "always_classifier": always("classifier"),
        "always_local (EVLM)": always("smolvlm_local"),
        "always_split": always("smolvlm_split"),
        "always_cloud": always("cloud"),
        "split_paper_rule": split_paper_rule(),
        "oracle": oracle(cache),
    }
    tuned = {
        "cascade_local": lambda t: cascade(t, to_local),
        "cascade_cloud (INAR-VL guard)": lambda t: cascade(t, to_cloud_if_link(15.0)),
        "cascade_by_task": lambda t: cascade(t, by_task(1.0)),
    }
    return fixed, tuned
