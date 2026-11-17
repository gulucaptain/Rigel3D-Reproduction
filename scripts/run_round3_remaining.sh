#!/bin/sh
# Continue this session's fixed-seed jobs; never overwrite completed results.
set -eu
until test -f results/round3/joint_frozen_surface/seed2/training.json; do sleep 5; done
.venv/bin/python scripts/cache_joint_latents.py
.venv/bin/python scripts/evaluate_joint_trellis.py > data/external/joint-formal-eval.log 2>&1 &
reconstruction_job=$!
.venv/bin/python scripts/audit_joint_eos.py > data/external/joint-formal-eos.log 2>&1
.venv/bin/python scripts/audit_trellis_coordinates.py > data/external/joint-formal-coordinates.log 2>&1
.venv/bin/python scripts/verify_round3_models.py > data/external/joint-formal-replay.log 2>&1
.venv/bin/python scripts/train_trellis_flows.py > data/external/trellis-formal-flows.log 2>&1
.venv/bin/python scripts/evaluate_trellis_generation.py > data/external/trellis-formal-generation.log 2>&1 &
generation_job=$!
.venv/bin/python scripts/train_trellis_flows.py --name flows_balanced_rig --balanced-rig-occupancy > data/external/trellis-balanced-flows.log 2>&1
wait "$reconstruction_job"
wait "$generation_job"
.venv/bin/python scripts/evaluate_trellis_generation.py --flow-run flows_balanced_rig --name generation_balanced_rig > data/external/trellis-balanced-generation.log 2>&1
