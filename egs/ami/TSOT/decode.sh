#!/usr/bin/env bash
# Decode test_1spk / test_2spk at the 4 latencies (modified beam search,
# beam 4) and score cpWER.
#   usage: ./decode.sh [--epoch 50 --avg 10 --use-averaged-model 1] <cfg index> <gpu id>
#
# Default: the averaged model of epochs 41-50 (fixed rule, used for all
# results in RESULTS.md). For a released checkpoint exp/cfg<i>/pretrained.pt:
#   ln -sf pretrained.pt exp/cfg35/epoch-999.pt
#   ./decode.sh --epoch 999 --avg 1 --use-averaged-model 0 35 0
# Results: exp/cfg<i>/modified_beam_search/, exp/cfg<i>/cpwer.tsv

set -eou pipefail
cd "$(dirname "$0")"

epoch=50
avg=10
use_averaged_model=1
. shared/parse_options.sh || exit 1

cfg=$1
export CUDA_VISIBLE_DEVICES=$2
exp=exp/cfg$cfg

# chunk:left-context (frames at 50 Hz): 320 ms, 640 ms, 1280 ms, full
for lat in 16:128 32:256 64:256 -1:-1; do
  IFS=: read chunk left <<< "$lat"
  python zipformer/decode.py --train-config $exp/config.yaml \
    --epoch $epoch --avg $avg --use-averaged-model $use_averaged_model \
    --chunk-size $chunk --left-context-frames $left \
    --decoding-method modified_beam_search --beam-size 4 \
    --max-duration 300 --test-sets test_1spk,test_2spk
done

python local/score_cpwer.py --exp-dirs $exp --sets test_1spk test_2spk \
  --epoch $epoch --avg $avg --out $exp/cpwer.tsv
