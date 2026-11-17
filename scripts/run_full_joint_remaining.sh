#!/bin/sh
set -eu
until test -f results/round3/full_joint/seed2/training.json && test -f results/round3/flows_balanced_rig/seed2/training.json; do sleep 5; done
.venv/bin/python scripts/cache_full_joint_latents.py > data/external/full-joint-latents.log 2>&1
.venv/bin/python scripts/evaluate_joint_trellis.py --joint-run full_joint --full-joint > data/external/full-joint-reconstruction.log 2>&1 &
reconstruction_job=$!
.venv/bin/python scripts/adapt_full_joint_local_flows.py > data/external/full-joint-local-flows.log 2>&1
.venv/bin/python scripts/evaluate_trellis_generation.py --joint-run full_joint --flow-run flows_full_joint --full-joint --name generation_full_joint > data/external/full-joint-generation.log 2>&1
.venv/bin/python scripts/evaluate_trellis_generation.py --joint-run full_joint --flow-run flows_full_joint_balanced --full-joint --name generation_full_joint_balanced > data/external/full-joint-balanced-generation.log 2>&1
wait "$reconstruction_job"
