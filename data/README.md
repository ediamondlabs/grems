# data/

| Path | Content |
|------|---------|
| `networks/*.tar.gz` | The exact train and test topologies used in the paper, one archive per family. Extract with `bash scripts/setup_networks.sh`. Each file is a pickled NetworkX graph `varnet_<N>nodes[_<tag>]_<i>.txt` where node attribute `weight` is the compute capability (1 to 10) and edge attribute `weight` the bandwidth. Files tagged `test` are held out. |
| `offline_profiles/layer_profiles.json` | Per layer profile of VGG-16 (FLOPs, activation size, reference forward time). Loaded when `--offline` is given. |
| `baselines.json` | Latency, energy and accuracy of DiSNet and MoDNN measured on the four original DiSNet testbed networks. Reference only, not read by the code. |

Archives: `variable_networks_rgg` (train and test, used for training), `variable_networks_ws`, `variable_networks_er`, `variable_networks` (Barabasi-Albert m = 2), and test only sets `variable_networks_ba_m1`, `_m3`, `_m4`.
