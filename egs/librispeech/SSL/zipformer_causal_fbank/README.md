# MSP / bi-label MSP pre-training of a causal Zipformer2 encoder

Masked speech prediction (HuBERT / WavLM style) for the streaming encoder of
`egs/ami/TSOT`, on LibriSpeech 960 h + 49 h of non-overlapped AMI IHM speech,
with k-means (k = 500) targets from layer 9 of WavLM Base+. The input is
80-dim fbank computed on the fly; the encoder (`encoder_embed` + `encoder`)
is the same causal medium Zipformer2 as in the t-SOT recipe, trained with
random chunk sizes.

- WavLM-style mixing: with prob. `mix_noise_prob` a DNS noise clip
  (U(-5, 20) dB), with prob. `mix_speech_prob` another utterance of the
  batch (U(-5, 5) dB, at most half the length).
- Bi-label MSP (`bi_label: true`): a second head predicts the k-means labels
  of the mixed-in utterance (an extra SIL class where it is absent).
- `ami_repeat`: AMI pieces repeated per epoch (x5 in the best setup).

Configs and fine-tuning results: `egs/ami/TSOT/RESULTS.md` ("SSL cfg<i>").
The best setup is `conf/cfg8.yaml` (bi-label, AMI x5, speech mixing 0.4).

## Requirements

icefall + k2 + lhotse; for the targets, the WavLM Base+ checkpoint
(`WavLM-Base+.pt`) and `WavLM.py` / `modules.py` from
[microsoft/unilm/wavlm](https://github.com/microsoft/unilm/tree/master/wavlm)
(`WAVLM_DIR`), and a python with fairseq + scikit-learn for fairseq's
`examples/hubert/simple_kmeans/learn_kmeans.py`. Data: LibriSpeech (FLAC),
AMI IHM + manual annotations (`egs/ami/TSOT/local/download_ami.sh`), DNS
Challenge 3 wideband noise.

## Steps

1. `./prepare.sh` (see the variables at its top):
   - 0: LibriSpeech FLAC -> WAV
   - 1: AMI IHM non-overlapped single-speaker pieces
     (`local/prepare_ami_ihm_nonoverlap.py`; segment and word times from the
     manual annotations)
   - 2: DNS noise at 16 kHz
   - 3-5: k-means on a ~100 h subset (seed 0, fairseq defaults)
   - 6: labels for all sets (`dump_labels=false` keeps released labels)
   - 7: lhotse cuts
2. `./run_pretrain.sh 8 0,1,2,3` trains `conf/cfg8.yaml` on 4 GPUs (~17 h on
   RTX 5090) and exports `exp/cfg8/epoch-84-avg-10.pt` (average of the last
   10 epochs), the encoder that the t-SOT fine-tuning configs load.
