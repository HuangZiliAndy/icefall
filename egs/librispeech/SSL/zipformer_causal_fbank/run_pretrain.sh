#!/usr/bin/env bash
# Pre-train one config, then export the encoder used for fine-tuning
# (average of the last 10 epochs).
#   usage: ./run_pretrain.sh <cfg index> <gpu ids>   e.g. ./run_pretrain.sh 8 0,1,2,3
# conf/cfg<i>.yaml -> exp/cfg<i>/ (world_size in the config: 4 GPUs for cfg3-cfg10)
# -> exp/cfg<i>/epoch-<num_epochs>-avg-10.pt, read by the init_encoder of the
#    fine-tuning configs in egs/ami/TSOT/conf/.

set -eou pipefail
cd "$(dirname "$0")"

cfg=$1
export CUDA_VISIBLE_DEVICES=$2
mkdir -p exp/cfg$cfg
python zipformer/pretrain.py --config conf/cfg$cfg.yaml > exp/cfg$cfg/nohup.log 2>&1

num_epochs=$(grep -E '^num_epochs:' conf/cfg$cfg.yaml | awk '{print $2}')
python zipformer/export_averaged_model.py --exp-dir exp/cfg$cfg --epoch $num_epochs --avg 10
