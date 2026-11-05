# Step 3 results: replay of the baselines

Date: 2026-11-05. The replay ran on the laptop CPU. It uses the step 2 tool cache, the 40 LTE traces and the cost model in `src/fedorch/costs.py`.
Source files: `results/step3/metrics.json` (Jetson factor 9), `metrics_jetson5.json`, `metrics_jetson35.json`.
Each value is the mean of 5 seeds. A seed changes the start point in the trace and the outage times.

## Set-up

- Mission: the 7,057 test-tower items, in tower order. One key frame arrives every 1.16 s (EVLM: 3.45 key frames in 4 s).
- Deadline: 4 s for each frame. A late answer is a miss.
- Drone: one compute queue for the onboard tools. Cloud computation does not use the queue.
- Onboard times: the GV100 time multiplied by 9. This factor is an estimate from the hardware ratio GV100 / Jetson Orin NX 16 GB (fp32 throughput and memory bandwidth). The sensitivity runs use 5 and 35.
- Uplink: the LTE trace in Mbps, multiplied by a regime scale. A cloud request fails when the transfer takes more than 3 s or the link is down.
- Regimes: `good` (scale 1, no outages), `weak` (scale 0.1, 6 outages of 20 s per hour), `poor` (scale 0.02, 20 outages of 30 s per hour). We constructed these regimes.
- Risk per 1,000 items: 10 for each missed defect, 1 for each false alarm, 0.5 for each abstention. These weights are assumptions.
- We tune the cascade thresholds on the validation towers, for each regime. The test towers are not used for tuning.

## Results (Jetson factor 9, weak regime)

| Policy | Risk / 1,000 | Defect recall | False alarm rate | Escalation rate | Uplink MB / 1,000 items |
|---|---|---|---|---|---|
| Oracle (hindsight) | 62.9 | 0.952 | 0.002 | 0.031 | 0.17 |
| Static cascade to `smolvlm_local`, θ = 0.95 | **155.7** | 0.896 | 0.015 | 0.144 | 0 |
| Cascade by task (T2 to `cloud`), θ = 0.95 | 161.0 | 0.905 | 0.037 | 0.144 | 2.64 |
| Classifier only | 177.0 | 0.888 | 0.026 | 0 | 0 |
| EVLM pipeline (`smolvlm_local` for each frame) | 188.3 | 0.879 | 0.030 | 1 | 0 |
| Split-paper rule (cloud if 10 Mbps or more) | 188.6 | 0.878 | 0.030 | 1 | 0.02 |
| `smolvlm_split` for each frame | 277.4 | 0.835 | 0.028 | 1 | 35.98 |
| `cloud` for each frame | 648.5 | 0.628 | 0.138 | 1 | 37.25 |

In the weak and poor regimes, the INAR-VL bandwidth guard (cloud only at 15 Mbps or more) never selects `cloud`. Its result is equal to the static cascade.

## Findings

1. The static cascade is the best baseline. It decreases the risk by 12% (177.0 to 155.7) relative to the classifier alone. It does not use the uplink. Its result is the same in all three regimes.
2. The oracle has a risk of 61 to 66. This value is 60% less than the static cascade. The oracle escalates only 3.1% of the items. The static cascade escalates 14.4%. Thus the problem is the selection of the correct items, not the quantity of escalation. A learned controller has a large margin to improve.
3. The cascade by task gives the highest defect recall (0.905) and the lowest miss rate (0.085). But `cloud` adds false alarms. Thus its risk is higher than the static cascade.
4. The EVLM pipeline (onboard VLM for each frame) is worse than the classifier alone (risk 188.3).
5. Policies that send each frame to the cloud fail in the poor regime. About 43% of their answers are late.
6. The split-paper rule is bad in the good regime (risk 546.8) because it then sends each frame to the weak `cloud` tool.

## Sensitivity to the Jetson factor

- Factor 5: all risks are equal to the factor-9 risks within 1%. The findings do not change.
- Factor 35: the onboard VLM needs about 2.8 s for each frame. Escalation to `smolvlm_local` then blocks the queue. The tuned cascade escalates almost no items (θ = 0.5), and its risk becomes the classifier risk (177.0). The EVLM pipeline is late on all frames. The oracle risk increases to 117 to 123.

## Limits

- The LTE traces are downlink measurements. We use them as the uplink.
- The Jetson factor is an estimate. A measurement on a Jetson is necessary to remove this limit.
- The risk weights are assumptions. The paper must show the risk for more than one set of weights.
