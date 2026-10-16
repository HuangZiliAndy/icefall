#!/usr/bin/env bash
# Data preparation for streaming t-SOT on AMI IHM-Mix utterance groups.
#
# Outputs (all in data/):
#   ihm_mix_utterance_group/{train,dev,test,test_1spk,test_2spk}/  Kaldi-style dirs
#   manifests/   lhotse recordings / supervisions (t-SOT text), MUSAN
#   fbank/       ami_ihm_mix_utterance_group_cuts_{train-shuf,dev,test_1spk,test_2spk}.jsonl.gz,
#                musan_cuts.jsonl.gz
#   lang_bpe_500/bpe.model  BPE 500 with <cc> as a user-defined symbol

set -eou pipefail

stage=0
stop_stage=100
nj=32

# Downloaded by stage 0 (local/download_ami.sh) if missing.
ami_dir=$PWD/download/AMI                 # <meeting>/audio/*.Headset-*.wav, ami_public_manual_1.6.2/
mix_dir=$PWD/download/AMI_IHM_MIX_OFFICIAL  # <meeting>.Mix-Headset.wav
# MUSAN (http://www.openslr.org/17/), with music/ noise/ speech/
musan_dir=$PWD/download/musan

. shared/parse_options.sh || exit 1

vocab_size=500
annotations_dir=$ami_dir/ami_public_manual_1.6.2
ug=data/ihm_mix_utterance_group
mkdir -p data

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  log "Stage 0: Download AMI IHM, manual annotations and Mix-Headset audio"
  ./local/download_ami.sh $ami_dir $mix_dir
  if [ ! -d $musan_dir ]; then
    lhotse download musan $(dirname $musan_dir)
  fi
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  log "Stage 1: Utterance groups from the manual annotations"
  common="--annotations-dir $annotations_dir --audio-dir $mix_dir"
  common="$common --wav-template {meeting}.Mix-Headset.wav --out-dir $ug"
  # train/dev: split jointly by #speakers and length (<= 2 speakers, <= 25 s)
  python local/prepare_utt_groups.py $common --splits train dev \
    --max-dur 25 --max-speakers 2 --max-cut-level 3 \
    > data/prepare_utt_groups_train_dev.log 2>&1
  # test: natural groups, never split
  python local/prepare_utt_groups.py $common --splits test --max-dur 0 \
    > data/prepare_utt_groups_test.log 2>&1
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  log "Stage 2: lhotse manifests with t-SOT transcripts"
  python local/prepare_tsot_manifests.py --kaldi-dir $ug \
    --output-dir data/manifests --splits train dev --min-dur 0.5
  python local/prepare_tsot_manifests.py --kaldi-dir $ug \
    --output-dir data/manifests --splits test --min-dur 0
fi

if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  log "Stage 3: Compute fbank"
  python local/compute_fbank_tsot.py --split train --perturb-speed 1 --num-jobs $nj
  python local/compute_fbank_tsot.py --split dev --num-jobs $nj
  python local/compute_fbank_tsot.py --split test --num-jobs $nj
  # Fixed shuffle of the training cuts (zipformer/asr_datamodule.py reads -shuf)
  f=data/fbank/ami_ihm_mix_utterance_group_cuts_train
  gunzip -c $f.jsonl.gz | shuf --random-source=<(yes) | gzip -c > $f-shuf.jsonl.gz
fi

if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  log "Stage 4: Split test by number of speakers (test_1spk, test_2spk)"
  python local/subset_by_num_speakers.py --split test --num-speakers 1 2
fi

if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  log "Stage 5: MUSAN manifests and fbank"
  lhotse prepare musan $musan_dir data/manifests
  python local/compute_fbank_musan.py
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  log "Stage 6: BPE $vocab_size with <cc>"
  lang_dir=data/lang_bpe_${vocab_size}
  mkdir -p $lang_dir
  # t-SOT training texts, split at <cc> so that BPE never sees the symbol
  python - <<EOF
from lhotse import load_manifest
sups = load_manifest("data/manifests/ami_ihm_mix_utterance_group_supervisions_train.jsonl.gz")
with open("$lang_dir/transcript_words.txt", "w") as f:
    for s in sups:
        for chunk in s.text.split("<cc>"):
            if chunk.strip():
                f.write(chunk.strip() + "\n")
EOF
  python local/train_bpe_model.py --lang-dir $lang_dir --vocab-size $vocab_size \
    --transcript $lang_dir/transcript_words.txt --extra-symbols "<cc>"
fi
