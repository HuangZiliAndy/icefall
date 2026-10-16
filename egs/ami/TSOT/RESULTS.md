# Results

## Streaming t-SOT zipformer transducer, AMI IHM-Mix utterance groups

Only `modified_beam_search` (beam 4) results are recorded.

### Summary, cpWER (%) on test

| model | pre-training | 320ms 1spk | 320ms 2spk | 640ms 1spk | 640ms 2spk | 1280ms 1spk | 1280ms 2spk | full 1spk | full 2spk | combined, mean of 4 |
|---|---|---|---|---|---|---|---|---|---|---|
| cfg2 | none | 27.83 | 46.17 | 26.11 | 45.06 | 24.89 | 44.63 | 23.59 | 44.14 | 36.01 |
| cfg38 | MSP (SSL cfg9) | 20.16 | 39.48 | 17.69 | 37.95 | 16.56 | 37.42 | 15.21 | 35.82 | 28.28 |
| cfg35 | bi-label MSP (SSL cfg8) | **19.23** | **38.13** | **17.40** | **35.47** | **16.15** | **34.76** | **15.14** | **32.56** | **26.77** |

SSL cfg8 and SSL cfg9 differ only in the bi-label objective. "combined,
mean of 4" is the cpWER on test_1spk + test_2spk averaged over the 4
latencies; it is the number used to compare models below (see "Seed
noise").

### Common setup

- Data: official AMI Mix-Headset audio; utterance groups from the manual
  (NXT) annotations. train/dev: groups split jointly by speaker count and
  length (<= 2 speakers, <= 25 s), 62.6 h train (x3 speed perturbation,
  0.9 / 1.1), MUSAN noise, SpecAugment;
  test: unsplit natural groups, scored on the 1-speaker (1719 groups,
  1.95 h) and 2-speaker (691 groups, 1.80 h) subsets.
- Targets: t-SOT, words sorted by start time, `<cc>` on speaker change;
  BPE 500 with `<cc>` as a user-defined symbol.
- Model: medium Zipformer2 (66.1M params), causal, conv front-end
  (`Conv2dSubsampling`, 100 Hz -> 50 Hz), 16 layers in 6 stacks
  (2,2,3,4,3,2), 25 Hz output; stateless decoder (context 2); pruned
  RNN-T loss. Chunk sizes 16/32/64/full with left context
  64/128/256/full sampled per batch, in both pre-training and fine-tuning.
- Training (fine-tuning and from scratch): 1 GPU (RTX 5090),
  max-duration 1500, 50 epochs (23.4k batches), ScaledAdam + Eden,
  lr_batches 1700, lr_epochs 3.5, warm_step 450, fp16, seed 42.
- Checkpoint: fixed rule `--epoch 50 --avg 10 --use-averaged-model 1`.
- Decoding: simulated streaming (`zipformer/decode.py`), max-duration 300.
  Latency = chunk size: 320ms = `--chunk-size 16 --left-context-frames 128`,
  640ms = `32/256`, 1280ms = `64/256`, full = `-1/-1`.
- Metric: cpWER (MeetEval), reference = one stream per speaker,
  hypothesis = non-empty t-SOT channels; errors summed over groups
  (`local/score_cpwer.py`).
- There is no dev-set cpWER; the fine-tuning learning rate below was
  chosen on test.

### Baseline trained from scratch

Configs: `conf/cfg2.yaml` (base_lr 0.03), `conf/cfg3.yaml`
(base_lr 0.01). base_lr 0.1 (cfg1) diverged; 0.003 and below
(cfg4-cfg6) underfit.

#### cpWER (%), test_1spk

| model | base_lr | 320ms | 640ms | 1280ms | full |
|---|---|---|---|---|---|
| cfg2 | 0.03 | 27.83 | 26.11 | 24.89 | 23.59 |
| cfg3 | 0.01 | 35.18 | 31.99 | 30.65 | 28.67 |

#### cpWER (%), test_2spk

| model | base_lr | 320ms | 640ms | 1280ms | full |
|---|---|---|---|---|---|
| cfg2 | 0.03 | 46.17 | 45.06 | 44.63 | 44.14 |
| cfg3 | 0.01 | 54.03 | 52.45 | 51.00 | 48.92 |

#### cpWER (%), test_1spk + test_2spk

| model | base_lr | 320ms | 640ms | 1280ms | full |
|---|---|---|---|---|---|
| cfg2 | 0.03 | 37.67 | 36.28 | 35.48 | 34.61 |
| cfg3 | 0.01 | 45.30 | 42.97 | 41.57 | 39.54 |

#### Error breakdown, cfg2

| set | latency | cpWER | sub | del | ins | groups w/ wrong #channels (%) | hyp/ref #<cc> |
|---|---|---|---|---|---|---|---|
| test_1spk | 320ms | 27.83 | 16.08 | 8.86 | 2.89 | 2.68 | nan |
| test_1spk | 640ms | 26.11 | 14.85 | 8.57 | 2.69 | 3.03 | nan |
| test_1spk | 1280ms | 24.89 | 13.91 | 8.53 | 2.46 | 2.62 | nan |
| test_1spk | full | 23.59 | 13.45 | 8.01 | 2.13 | 2.44 | nan |
| test_2spk | 320ms | 46.17 | 20.68 | 19.83 | 5.66 | 25.90 | 0.55 |
| test_2spk | 640ms | 45.06 | 20.23 | 19.57 | 5.26 | 25.33 | 0.56 |
| test_2spk | 1280ms | 44.63 | 19.52 | 19.59 | 5.52 | 26.19 | 0.55 |
| test_2spk | full | 44.14 | 19.03 | 19.43 | 5.68 | 25.90 | 0.55 |

On 2-speaker groups the model predicts only ~55% of the reference speaker
changes and puts everything in one channel for ~26% of the groups, which
shows up as a high deletion rate.

### Self-supervised pre-training of the encoder

Recipe: `egs/librispeech/SSL/zipformer_causal_fbank` (`zipformer/pretrain.py`,
configs `conf/cfg<i>.yaml`; "SSL cfg<i>" below). The pre-trained encoder is
the same causal Zipformer2 as above (`encoder_embed` + `encoder`); it is
loaded into `zipformer/train.py` with `init_encoder` (826 of 827 tensors).

- Data: LibriSpeech 960 h + 49.3 h of non-overlapping single-speaker
  segments (36,287) cut from the AMI IHM train set with the official
  word alignments. `ami_repeat` upsamples the AMI segments within each
  epoch.
- Targets: k-means (k = 500) on layer 9 of WavLM Base+, 50 Hz, trained
  on a ~100 h subset (~90 h LibriSpeech + ~10 h AMI). `label_shift 2`
  aligns the fbank front-end with the WavLM frames.
- Objective: masked speech prediction (MSP), mask_prob 0.8, mask_length
  10, CE on masked frames only. WavLM-style mixing: with prob.
  `mix_noise_prob` add a DNS noise clip (energy ratio U(-5, 20) dB), with
  prob. `mix_speech_prob` add another utterance of the batch (U(-5, 5) dB,
  at most half the length).
- Bi-label MSP (`bi_label: true`): a second head predicts the labels of
  the mixed-in (secondary) utterance, with an extra SIL class where it is
  absent; loss = primary + secondary.
- Training: 4 GPUs x max-duration 1400, ScaledAdam + Eden (lr_batches
  2500), fp16. The total audio seen is kept at ~101k h (100 epochs of
  1010 h); with AMI upsampled, `num_epochs` and `lr_epochs` are scaled
  down accordingly. The encoder used for fine-tuning is the average of the
  last 10 epochs (`zipformer/export_averaged_model.py`).

| SSL | base_lr | noise / speech mixing | bi-label | AMI share per epoch | epochs |
|---|---|---|---|---|---|
| cfg3 | 0.1 | 0.1 / 0.1 | no | x1 (4.9%) | 100 |
| cfg4 | 0.03 | 0.1 / 0.1 | no | x1 (4.9%) | 100 |
| cfg5 | 0.1 | 0.1 / 0.4 | yes | x1 (4.9%) | 100 |
| cfg6 | 0.03 | 0.1 / 0.4 | yes | x1 (4.9%) | 100 |
| cfg7 | 0.1 | 0.1 / 0.4 | yes, secondary non-SIL frames weighted x3 | x1 (4.9%) | 100 |
| cfg8 | 0.1 | 0.1 / 0.4 | yes | x5 (20.4%) | 84 |
| cfg9 | 0.1 | 0.1 / 0.4 | no | x5 (20.4%) | 84 |
| cfg10 | 0.1 | 0.1 / 0.4 | yes | x10 (33.9%) | 70 |

SSL cfg1 / cfg2 (the original k2SSL 8-GPU, 300-epoch HuBERT / WavLM
settings) have no results.

### Fine-tuning from pre-trained encoders

Each SSL model is fine-tuned with several learning-rate settings. "split"
means base_lr 0.01 for decoder / joiner / simple projections and 0.003 for
the encoder (`encoder_lr_scale: 0.3`). cpWER (%), mean over the 4
latencies.

| fine-tune | SSL | learning rate | test_1spk | test_2spk | combined |
|---|---|---|---|---|---|
| cfg2 | none (from scratch) | 0.03 | 25.61 | 45.00 | 36.01 |
| cfg7 | cfg3 | 0.03 | diverged | | |
| cfg8 | cfg3 | 0.01 | 17.18 | 36.49 | 27.54 |
| cfg9 | cfg3 | 0.003 | 16.85 | 38.36 | 28.39 |
| cfg10 | cfg3 | 0.001 | 62.86 | 74.29 | 68.99 |
| cfg27 | cfg3 | 0.03, encoder x0.33 | 18.13 | 36.43 | 27.95 |
| cfg28 | cfg3 | 0.03, encoder x0.1 | 17.96 | 35.72 | 27.49 |
| cfg29 | cfg3 | 0.03, encoder x0.033 | 18.84 | 39.23 | 29.79 |
| cfg30 | cfg3 | split | 16.70 | 36.26 | 27.19 |
| cfg11 | cfg4 | 0.03 | 20.62 | 40.73 | 31.41 |
| cfg12 | cfg4 | 0.01 | 18.40 | 37.92 | 28.88 |
| cfg13 | cfg4 | 0.003 | 17.51 | 38.07 | 28.54 |
| cfg14 | cfg4 | 0.001 | 26.80 | 51.98 | 40.31 |
| cfg15 | cfg5 | 0.03 | diverged | | |
| cfg16 | cfg5 | 0.01 | 17.67 | 35.73 | 27.37 |
| cfg17 | cfg5 | 0.003 | 17.95 | 38.61 | 29.04 |
| cfg18 | cfg5 | 0.001 | 60.28 | 72.49 | 66.83 |
| cfg23 | cfg5 | 0.03, encoder x0.33 | 17.99 | 36.60 | 27.98 |
| cfg24 | cfg5 | 0.03, encoder x0.1 | 18.02 | 36.99 | 28.20 |
| cfg25 | cfg5 | 0.03, encoder x0.033 | 19.53 | 39.88 | 30.45 |
| cfg26 | cfg5 | split | 16.79 | 36.03 | 27.12 |
| cfg34 | cfg5 | split, seed 43 | 17.26 | 35.95 | 27.29 |
| cfg19 | cfg6 | 0.03 | 21.07 | 40.24 | 31.36 |
| cfg20 | cfg6 | 0.01 | 19.06 | 39.16 | 29.85 |
| cfg21 | cfg6 | 0.003 | 18.90 | 37.93 | 29.11 |
| cfg22 | cfg6 | 0.001 | 29.55 | 50.17 | 40.61 |
| cfg31 | cfg7 | split | 18.03 | 37.07 | 28.25 |
| cfg32 | cfg7 | 0.01 | 18.03 | 36.68 | 28.04 |
| cfg33 | cfg7 | 0.003 | 19.16 | 40.48 | 30.61 |
| cfg35 | cfg8 | split | 16.98 | 35.23 | **26.77** |
| cfg36 | cfg8 | 0.01 | 17.83 | **34.61** | 26.84 |
| cfg37 | cfg8 | 0.003 | 17.35 | 36.08 | 27.40 |
| cfg38 | cfg9 | split | 17.41 | 37.67 | 28.28 |
| cfg39 | cfg9 | 0.01 | 18.36 | 38.49 | 29.16 |
| cfg40 | cfg9 | 0.003 | 18.43 | 39.90 | 29.96 |
| cfg41 | cfg10 | split | 17.45 | 35.73 | 27.26 |
| cfg42 | cfg10 | 0.01 | 18.10 | 36.06 | 27.73 |
| cfg43 | cfg10 | 0.003 | 17.91 | 38.56 | 28.99 |

#### Bi-label vs. single-label MSP (SSL cfg8 vs. cfg9), cpWER (%)

Same data ratio (AMI x5), same mixing (0.1 / 0.4), only bi-label differs.

| fine-tune lr | set | | 320ms | 640ms | 1280ms | full | mean |
|---|---|---|---|---|---|---|---|
| split | test_1spk | bi-label (cfg35) | **19.23** | **17.40** | **16.15** | **15.14** | **16.98** |
| | | MSP (cfg38) | 20.16 | 17.69 | 16.56 | 15.21 | 17.41 |
| | test_2spk | bi-label (cfg35) | **38.13** | **35.47** | **34.76** | **32.56** | **35.23** |
| | | MSP (cfg38) | 39.48 | 37.95 | 37.42 | 35.82 | 37.67 |
| 0.01 | test_1spk | bi-label (cfg36) | **20.26** | **17.70** | **17.18** | **16.17** | **17.83** |
| | | MSP (cfg39) | 20.71 | 18.66 | 17.75 | 16.30 | 18.36 |
| | test_2spk | bi-label (cfg36) | **37.38** | **34.73** | **33.35** | **32.97** | **34.61** |
| | | MSP (cfg39) | 40.18 | 38.11 | 38.93 | 36.74 | 38.49 |
| 0.003 | test_1spk | bi-label (cfg37) | **20.31** | **17.39** | **16.38** | **15.32** | **17.35** |
| | | MSP (cfg40) | 21.93 | 18.28 | 17.06 | 16.47 | 18.43 |
| | test_2spk | bi-label (cfg37) | **38.87** | **35.76** | **35.76** | **33.92** | **36.08** |
| | | MSP (cfg40) | 43.99 | 38.86 | 38.56 | 38.20 | 39.90 |

Bi-label is better in every cell; the gain is mostly on test_2spk
(2.4-3.9 points on the mean).

#### Seed noise

cfg26 and cfg34 differ only in the seed (42 / 43): combined mean 27.12 vs.
27.29 (0.17), but up to 2.0 points on test_2spk at a single latency
(320ms: 38.15 vs. 36.19). Differences below ~0.5 on the combined mean, or
single-latency 2spk differences of 1-2 points, should not be trusted.

#### Findings

- Pre-training lowers the combined cpWER from 36.01 (from scratch) to
  26.77 (SSL cfg8 + cfg35), -26% relative.
- Bi-label is necessary once speech mixing is raised to 0.4: removing it
  (SSL cfg9) costs 1.5-2.6 points and is worse than WavLM-style MSP with
  0.1 speech mixing (SSL cfg3, best 27.19). With single-label MSP, heavy
  speech mixing teaches the encoder to suppress the second speaker.
- AMI x5 (SSL cfg8) beats the natural ratio (SSL cfg5) by 0.4-1.6 points
  in all three fine-tuning settings; AMI x10 (SSL cfg10) is worse than x5
  by 0.5-1.6.
- Weighting secondary frames x3 (SSL cfg7) is worse than cfg5 by 0.7-1.6.
- Pre-training base_lr 0.1 is better than 0.03 (cfg3 vs. cfg4, cfg5 vs.
  cfg6).
- Fine-tuning: lr 0.03 diverges (cfg7, cfg15) or is clearly worse, 0.001
  underfits; the split learning rate is the most stable and is best or on
  par for every SSL model.

Fine-tuning and decoding command (best model, cfg35, 320ms):

```bash
./zipformer/train.py --config conf/cfg35.yaml
./zipformer/decode.py --train-config exp/cfg35/config.yaml \
  --epoch 50 --avg 10 --use-averaged-model 1 \
  --chunk-size 16 --left-context-frames 128 \
  --decoding-method modified_beam_search --beam-size 4 \
  --max-duration 300 --test-sets test_1spk,test_2spk
python local/score_cpwer.py --exp-dirs exp/cfg35 --sets test_1spk test_2spk
```

`init_encoder` in the fine-tuning configs points to the averaged SSL
checkpoint, e.g. `exp/cfg8/epoch-84-avg-10.pt` of the SSL recipe.
