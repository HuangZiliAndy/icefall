#!/usr/bin/env bash
# Train (fine-tune, or from scratch if the config has no init_encoder) one
# config on one GPU.
#   usage: ./run_finetune.sh <cfg index> <gpu id>     e.g. ./run_finetune.sh 35 0
# conf/cfg<i>.yaml -> exp/cfg<i>/ (stdout.log, EXIT)
# Fine-tuning configs read the averaged SSL encoder via init_encoder, e.g.
# ../../librispeech/SSL/zipformer_causal_fbank/exp/cfg8/epoch-84-avg-10.pt

set -eou pipefail
cd "$(dirname "$0")"

cfg=$1
export CUDA_VISIBLE_DEVICES=$2
mkdir -p exp/cfg$cfg
start=$(date +%s)
rc=0
python zipformer/train.py --config conf/cfg$cfg.yaml > exp/cfg$cfg/stdout.log 2>&1 || rc=$?
echo "exit_code=$rc gpu=$2 elapsed=$(( $(date +%s) - start ))s" > exp/cfg$cfg/EXIT
exit $rc
