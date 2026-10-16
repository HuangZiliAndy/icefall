# Streaming t-SOT on AMI IHM-Mix

Streaming multi-talker ASR with token-level serialized output training
(t-SOT): a causal Zipformer2 pruned RNN-T trained on AMI IHM-Mix utterance
groups (<= 2 speakers), with `<cc>` marking speaker changes. The encoder can
be initialised from MSP / bi-label MSP pre-training
(`egs/librispeech/SSL/zipformer_causal_fbank`). Results: [RESULTS.md](RESULTS.md).

## Requirements

icefall + k2 + lhotse, sentencepiece, and
[MeetEval](https://github.com/fgnt/meeteval) for cpWER (`local/score_cpwer.py`).

## Steps

1. Data (`./prepare.sh`, stages 0-6): downloads AMI IHM headsets, the NXT
   manual annotations and the official Mix-Headset audio
   (`local/download_ami.sh`, set `--ami-dir` / `--mix-dir` / `--musan-dir`
   to reuse existing copies), builds utterance groups and t-SOT manifests,
   fbank (train x3 speed perturbation), the `test_1spk` / `test_2spk` subsets,
   MUSAN and BPE 500 with `<cc>`.

2. Train one config on one GPU (`conf/cfg<i>.yaml` -> `exp/cfg<i>/`):

   ```bash
   ./run_finetune.sh 2 0    # from scratch (baseline)
   ./run_finetune.sh 35 0   # fine-tune the bi-label MSP encoder (SSL cfg8)
   ```

   Fine-tuning configs load the averaged pre-trained encoder through
   `init_encoder`, e.g.
   `../../librispeech/SSL/zipformer_causal_fbank/exp/cfg8/epoch-84-avg-10.pt`.
   Which config uses which SSL model and learning rate is listed in
   RESULTS.md.

3. Decode at 320 / 640 / 1280 ms / full and score cpWER
   (epochs 41-50 averaged):

   ```bash
   ./decode.sh 35 0          # -> exp/cfg35/cpwer.tsv
   ```

## Released checkpoints

`exp/cfg{2,35,38}/pretrained.pt` hold exactly the averaged model evaluated
in RESULTS.md (`local/export_averaged_model.py`, weights only). To decode one:

```bash
ln -sf pretrained.pt exp/cfg35/epoch-999.pt
./decode.sh --epoch 999 --avg 1 --use-averaged-model 0 35 0
```

`data/lang_bpe_500/bpe.model` must be the released one (token ids).
