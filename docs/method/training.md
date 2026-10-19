# Reward and training

## Reward

```
reward = -ALPHA * log(latency_ms)          # ALPHA = 5.0, latency floored at 1 ms
```

The logarithm compresses latencies from about 100 ms to 100 000 ms into a stable signal. Shaped terms are added in `environment.py`:

| Term | Value | Purpose |
|------|-------|---------|
| Connectivity penalty | per disconnected hub to worker pair, ramps from 5 to 50 over 500k steps | forces plans that use existing links |
| Weak link penalty | beta 10, link below 2 MB/s but not zero | avoids nearly dead first hop links |
| Single node plan | flat 140 | keeps "everything on the hub" dominated by real splits |
| Memory overload | up to 200, scales with layers above `NUM_LAYERS / ceil(sqrt(N))` | pushes towards more stages on small networks |
| Routing quality bonus | up to +5, fraction of inter stage transfers on GOOD links | low variance signal for link awareness |
| Node utilisation bonus | up to +25 by default, **0 for the released models** | disabled with `NODE_UTIL_BONUS=0.0` because it destabilised fine tuning |

Rewards are normalised online with a running mean and variance (Welford, `RunningMeanStd`) before the advantage computation, so the critic target stays near zero mean and unit variance when the channel regime shifts.

## PPO

| Parameter | Stage 1 | Stage 2 |
|-----------|---------|---------|
| Algorithm | PPO with GAE, discount 0.99, lambda 0.95 | same |
| Learning rate | 3e-4, linear decay to 0 | 3e-5, linear decay to 0 |
| Rollout and batch size | 512 and 512 | 512 and 512 |
| Epochs per update | 4 | 4 |
| Clip epsilon | 0.2 | 0.2 |
| Entropy coefficient | 0.02, decays to 0.002 over 2M steps | same |
| Value coefficient | 0.5 | 0.5 |
| Gradient clipping | 0.5 | 0.5 |
| Timesteps | 30 000 000 from scratch | 10 000 000 from Stage 1 `grems_final.pt` |
| Sizes | N = 4 to 10 | N = 4 to 20, sizes 11 and above with weight 2 |
| `CONN_PENALTY_START` | 5 (default) | 50 (already converged) |

## Why two stages

Training from scratch on all sizes up to 20 is slow because large graphs are expensive to simulate and have a much larger action space. Stage 1 learns the core behaviour on small networks. Stage 2 exposes the converged policy to larger graphs at a low learning rate so it extends without forgetting. Several other fine tuning variants (node bonus on, small learning rate only, different penalty schedules) were explored during development. They destabilised the converged policy, which is why Stage 2 pins the connectivity penalty and disables the node bonus.

## Logging and checkpoints

- Console and `logs/train_GReMS.log` show reward, latency and PPO statistics every `--log-interval` rollouts.
- Every `--eval-interval` steps the policy is evaluated on a fixed set of held out networks, and a row `step, seen_lat_ms_all, seen_lat_ms_<topology>` is appended to `learning_curve.csv` next to the checkpoints.
- `grems_best.pt` is the best checkpoint by that evaluation, `grems_final.pt` the last one, `grems_latest.pt` a crash recovery snapshot every 50 rollouts.
- Weights & Biases is optional. Pass `--wandb-project <name>` to enable it.
