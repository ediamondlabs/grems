#!/bin/bash
# Unpack the exact topology files (train + held-out test networks) used in the paper.
#
# Why archives instead of regenerating? generate_variable_networks.py is seeded,
# but the shipped networks were produced by earlier revisions of the generator,
# so regenerating gives different graphs. The archives are the ground truth.
#
# Usage: bash scripts/setup_networks.sh [topology ...]
#        (default: rgg ws er ba ba_m1 ba_m3 ba_m4)
set -e
cd "$(dirname "$0")/.."

TOPOS=("$@")
[ ${#TOPOS[@]} -eq 0 ] && TOPOS=(rgg ws er ba ba_m1 ba_m3 ba_m4)

for t in "${TOPOS[@]}"; do
    if [ "$t" = "ba" ]; then
        archive="data/networks/variable_networks.tar.gz"
        dest="variable_networks"
    else
        archive="data/networks/variable_networks_${t}.tar.gz"
        dest="variable_networks_${t}"
    fi
    if [ -d "$dest" ] && [ -n "$(ls -A "$dest" 2>/dev/null)" ]; then
        echo "[skip] $dest/ already exists"
        continue
    fi
    echo "[extract] $archive -> $dest/"
    tar xzf "$archive"
done
echo "Done. Test networks: variable_networks_<topo>/varnet_<N>nodes_test_<i>.txt"
