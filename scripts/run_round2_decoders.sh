#!/bin/sh
set -eu
.venv/bin/python scripts/train_subset.py --steps 2000 --seeds 0 1 2 --device cpu --name round2_geometry --cohort --balanced-stop
.venv/bin/python scripts/train_subset.py --steps 2000 --seeds 0 1 2 --device cpu --name round2_dino --cohort --context dino --balanced-stop
.venv/bin/python scripts/train_subset.py --steps 2000 --seeds 0 1 2 --device cpu --name round2_vae --cohort --context vae --balanced-stop
