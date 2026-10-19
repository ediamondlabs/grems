# Latency model

Latency is not measured on real hardware during training. It is computed by `run_rl_live.py` (function `measure_rl_plan_latency_ms`) from lookup tables, which makes millions of episodes affordable and makes all methods comparable.

## Ingredients

- **Per layer compute time** `LAYER_IOT_FWD_MS` (in `config.py`). Layer times were profiled on a Xeon E5-2690 and scaled so that the full VGG-16 pass takes 7253 ms, the Jetson Nano MAXN figure from the DiSNet paper.
- **Node speed.** The compute time of a layer slice is `LAYER_IOT_FWD_MS[l] * fraction / speedup(w) * slowdown`, with `speedup(w) = 1 + 0.4 * (w - 1)` and `slowdown = min(1 / (1 - rho), 4)` where `rho` is the current MMPP background load of the node.
- **Transfers.** `bytes / (bandwidth_MBps * 1024^2) * 1000` ms plus per transfer overhead (`OVERHEAD_PER_DATA_TRANSFER = 0.5 ms`). If two nodes are not directly connected, the transfer is routed along the Dijkstra shortest path (store and forward).
- **Overheads.** 5 ms per remote worker setup and 1.5 ms per chunk (`OVERHEAD_REMOTE_NODE_SETUP`, `OVERHEAD_PER_CHUNK`).
- **Receptive field overlap.** A worker that processes a slice of a convolutional stage needs extra input rows from its neighbours. `group_fov_overlap` in `config.py` computes how many.

## Recursive worker to worker model

The communication model follows the paper section on the latency model. Each worker `w` of group `g` finishes at

```
Delta[g, w] = delta_C[w] + max over u in W[g-1, w] of ( Delta[g-1, u] + delta_comm[u, w] )
```

where `W[g-1, w]` is the set of workers of the previous group whose output rows `w` needs, `delta_C` is the compute time of `w` for its slice and `delta_comm` the transfer time from `u` to `w`.
This yields a dependency DAG over workers. The end to end latency is the finish time of the last group, plus the transfer of the final activation to the node that runs the fully connected layers.

- Hub to group 0 workers: input rows plus overlap, once per request.
- Group `g` to group `g + 1`: every worker sends exactly the rows each consumer needs, over a direct link or a multi hop route.
- Final spatial group to the FC node: the full activation. Fully connected layers run on one agent chosen node without horizontal split.

!!! info "Version note"
    The latency model was revised shortly before the release (commit "Updated Latency Model", September 2026). Checkpoints trained with the older bottleneck model are not comparable and are not included.

## Limits

The model is analytical. It captures compute, bandwidth, routing and overlap, but not packet loss, queueing inside the network, or memory contention beyond the penalty term. See [Extending GReMS](../extending.md) for how to plug in a different latency estimator.
