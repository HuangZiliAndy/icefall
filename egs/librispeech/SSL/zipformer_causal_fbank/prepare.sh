#!/usr/bin/env bash
# Data preparation for MSP / bi-label MSP pre-training of the causal Zipformer2
# encoder on LibriSpeech 960 h + AMI IHM non-overlapped speech, with k-means
# (k = 500) targets from layer 9 of WavLM Base+.
#
# Outputs:
#   data/kmeans/wavlm_base_plus_L9_km500/  k-means subset, km500.bin,
#                                          km500_centers.npy, labels/<set>.{tsv,km}
#   data/noise/dns_noise.tsv               DNS Challenge 3 noise list
#   data/manifests/ssl_cuts_*.jsonl.gz     lhotse cuts with k-means labels
#
# The released km500.bin / km500_centers.npy can be dropped into
# data/kmeans/wavlm_base_plus_L9_km500/; then start at stage 6
# (--stage 6 after stages 0-2) to label the data with exactly the same
# clusters, or also drop in the released labels/*.km and use
# --stage 6 --dump-labels false.

set -eou pipefail

stage=0
stop_stage=100
nj=32
ngpu=8   # feature / label dumping: one shard per GPU

librispeech_flac_dir=$PWD/download/LibriSpeech   # train-*/ dev-*/ (FLAC)
librispeech_dir=$PWD/download/LibriSpeech_wav    # WAV copy written by stage 0
ami_dir=$PWD/download/AMI       # from egs/ami/TSOT/local/download_ami.sh
ami_nonoverlap_dir=$PWD/download/AMI_IHM_nonoverlap
dns_noise_dir=$PWD/download/DNS_Challenge_3/datasets/noise  # DNS Challenge 3 (wideband) noise
dns_noise_16k_dir=$PWD/download/DNS_Challenge_3/noise_16k
wavlm_ckpt=$PWD/download/WavLM-Base+.pt
# Directory with WavLM.py / modules.py from microsoft/unilm (wavlm/)
export WAVLM_DIR=${WAVLM_DIR:-$PWD/download/unilm/wavlm}
# A python with fairseq + scikit-learn, for fairseq's learn_kmeans.py
fairseq_python=python
# false: only write labels/<set>.tsv for the local audio paths and keep the
# released labels/<set>.km (same files, same sorted order)
dump_labels=true

. shared/parse_options.sh || exit 1

K=data/kmeans/wavlm_base_plus_L9_km500
mkdir -p $K/log data/noise data/manifests

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: LibriSpeech FLAC -> WAV"
  python local/convert_librispeech_to_wav.py $librispeech_flac_dir $librispeech_dir $nj
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: AMI IHM non-overlapped single-speaker pieces (train, dev)"
  python local/prepare_ami_ihm_nonoverlap.py --ami-dir $ami_dir \
    --out-dir $ami_nonoverlap_dir --nj $nj
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: DNS noise, 16 kHz"
  python local/prepare_dns_noise.py --in-dir $dns_noise_dir \
    --out-dir $dns_noise_16k_dir --list data/noise/dns_noise.tsv
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Sample the k-means subset (~90 h LibriSpeech + ~10 h AMI)"
  python local/sample_kmeans_subset.py --librispeech-dir $librispeech_dir \
    --ami-dir $ami_nonoverlap_dir --out-dir $K --seed 0
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "Stage 4: Dump WavLM Base+ layer-9 features of the k-means subset"
  for r in $(seq 0 $((ngpu - 1))); do
    CUDA_VISIBLE_DEVICES=$r python local/dump_wavlm_feature.py $K kmeans_subset \
      $wavlm_ckpt 9 $ngpu $r $K/feats > $K/log/dump.$r.log 2>&1 &
  done
  wait
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "Stage 5: Fit k-means (fairseq learn_kmeans.py, default settings)"
  # The script may crash after saving km500.bin, in the final inertia
  # computation over all frames (OpenBLAS); the saved model is unaffected.
  learn_kmeans=$($fairseq_python -c "import fairseq, os; print(os.path.join(os.path.dirname(fairseq.__file__), 'examples/hubert/simple_kmeans/learn_kmeans.py'))")
  $fairseq_python $learn_kmeans $K/feats kmeans_subset $ngpu $K/km500.bin 500 \
    --percent -1 --seed 0 > $K/log/learn_kmeans.log 2>&1 || true
  [ -f $K/km500.bin ] || { log "k-means failed, see $K/log/learn_kmeans.log"; exit 1; }
  $fairseq_python -c "
import joblib, numpy as np
km = joblib.load('$K/km500.bin')
np.save('$K/km500_centers.npy', km.cluster_centers_.astype(np.float32))
"
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  log "Stage 6: k-means labels for all sets"
  python local/make_label_tsvs.py --librispeech-dir $librispeech_dir \
    --ami-dir $ami_nonoverlap_dir --out-dir $K/labels
  sets="ami_ihm_dev ls_dev-clean ls_dev-other ami_ihm_train ls_train-clean-100 ls_train-clean-360 ls_train-other-500"
  if $dump_labels; then
    for r in $(seq 0 $((ngpu - 1))); do
      CUDA_VISIBLE_DEVICES=$r OMP_NUM_THREADS=2 python local/dump_wavlm_km_label.py \
        --tsv-dir $K/labels --out-dir $K/labels --sets $sets \
        --ckpt $wavlm_ckpt --layer 9 --centers $K/km500_centers.npy \
        --nshard $ngpu --rank $r > $K/log/label.$r.log 2>&1 &
    done
    wait
    # Merge the shards (in rank order = tsv order) into <set>.km
    mkdir -p $K/labels/shards
    for s in $sets; do
      for r in $(seq 0 $((ngpu - 1))); do cat $K/labels/${s}_${r}_${ngpu}.km; done > $K/labels/$s.km
      mv $K/labels/${s}_*_${ngpu}.km $K/labels/shards/
    done
    for x in $(seq 0 499); do echo "$x 1"; done > $K/labels/dict.km.txt
  fi
fi

if [ $stage -le 7 ] && [ $stop_stage -ge 7 ]; then
  log "Stage 7: lhotse cuts with k-means labels"
  python local/prepare_ssl_manifests.py --label-dir $K/labels --out-dir data/manifests
fi
