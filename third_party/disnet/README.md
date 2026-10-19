# Vendored DiSNet sources

These six files are an unmodified copy of the original DiSNet implementation
(Samikwa, Di Maio and Braun, IEEE Internet of Things Journal, 2023, doi:10.1109/JIOT.2023.3313514).
GReMS imports them from `run_disnet_live.py` and `run_modnn_live.py` to re-use DiSNet's
path selection, hub selection and horizontal split ratio logic, so that the DiSNet and
MoDNN baselines are evaluated with the authors' original decision code.

| File | Role |
|------|------|
| `configurations.py` | DiSNet parameters (partition limits, energy constants) |
| `network.py` | Graph reading, path selection, neighbour (hub) selection |
| `utils.py` | Split ratio, subset selection, bandwidth cap helpers |
| `main/inference.py` | Partition geometry (`get_partiton_info`, `get_partiton_info_DiSNet`) |
| `main/field_vgg.py`, `main/field_DiSNet.py` | Receptive field calculators for VGG-16 |

Original repository: the DiSNet project of the same authors.
