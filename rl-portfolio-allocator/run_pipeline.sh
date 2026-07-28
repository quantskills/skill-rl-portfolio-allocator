#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

: "${PANDA_DATA_USERNAME:?PANDA_DATA_USERNAME not set}"
: "${PANDA_DATA_PASSWORD:?PANDA_DATA_PASSWORD not set}"

echo "== [1/6] features =="
python scripts/features.py

echo "== [2/6] train smoke =="
python scripts/train.py --timesteps "${PIPELINE_TRAIN_TIMESTEPS:-5000}"

echo "== [3/6] backtest (research + tradeable) =="
python scripts/backtest.py --timesteps "${PIPELINE_TIMESTEPS:-200000}"

echo "== [4/6] stress test (four forward segments) =="
python scripts/stress_test.py --timesteps "${PIPELINE_TIMESTEPS:-100000}"

echo "== [5/6] allocate (production retrain + infer) =="
python scripts/allocate.py --retrain --timesteps "${PIPELINE_PROD_TIMESTEPS:-500000}"

echo "== [6/6] validate =="
python scripts/validate.py

echo "OK"
