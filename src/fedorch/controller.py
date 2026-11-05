from fedorch.costs import COST, onboard_s, server_s, uplink_s
from fedorch.simulate import RISK

LAMBDAS = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]


def expected_latency(tool, row, state):
    if tool == "smolvlm_local":
        latency = state["backlog_s"] + onboard_s(tool, row)
        return latency if latency <= COST["deadline_s"] else None
    transfer = uplink_s(row, state["mbps"])
    return None if transfer is None else transfer + server_s(tool, row)


def risk_rule(q, cache, lam):
    def policy(item, history, state):
        if not history:
            return "classifier"
        if len(history) > 1:
            return None
        quality = q[item["item_id"]]
        p_good = history[0][1]["probs"].get("good", 0.0)
        wrong_cost = RISK["miss"] * (1 - p_good) + RISK["false_alarm"] * p_good
        best, best_score = None, (1 - quality["classifier"]) * wrong_cost
        for tool in ("smolvlm_local", "cloud"):
            latency = expected_latency(tool, cache[item["item_id"]][tool], state)
            if latency is None:
                continue
            score = (1 - quality[tool]) * wrong_cost + lam * latency
            if score < best_score:
                best, best_score = tool, score
        return best
    return policy


def true_q(cache):
    return {item_id: {tool: float(rows[tool]["correct"]) for tool in ("classifier", "smolvlm_local", "cloud")}
            for item_id, rows in cache.items()}
