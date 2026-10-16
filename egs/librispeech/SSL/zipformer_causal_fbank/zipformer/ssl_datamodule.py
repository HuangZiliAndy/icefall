"""
Data pipeline for fbank-based HuBERT-style pre-training.

Batching follows k2SSL / fairseq HuBERT (pad_audio=False, random_crop=True):
all utterances of a batch are cropped to the shortest one in the batch (at
most --max-sample-size samples, 250000 = 15.6 s), at a random offset in
training and at offset 0 for validation. Crop offsets are multiples of 320
samples (the label stride), so the 50 Hz k-means labels are cut exactly:
labels[off // 320 : off // 320 + (L - 400) // 320 + 1].

fbank (80-dim lhotse Fbank, default config, dither 0) is computed on the fly
for each cropped waveform separately. Extracting per utterance (instead of
lhotse's OnTheFlyFeatures, which zero-pads waveforms before batch extraction
and so changes the last 1-2 frames) gives exactly the same features as the
offline extraction used for ASR (up to lilcom rounding there), and is faster.

Training batches can be augmented WavLM-style (wavlm_augment.py) on the
cropped waveforms, before fbank extraction: each utterance gets DNS noise
(--mix-noise-prob) or another utterance of the batch (--mix-speech-prob), or
nothing. Labels are those of the clean utterance. Validation is never
augmented.

Bi-label training (--bi-label, off by default): besides the primary labels,
each utterance gets secondary labels on the same 50 Hz frames: inside the
region where another utterance of the batch was mixed in, the clean labels
of that utterance (shifted to the primary's time axis; mixing positions are
multiples of the 320-sample label stride), and SIL elsewhere (also for
noise-mixed and unmixed utterances). Validation then additionally runs on
"mixed" copies of the dev sets, where every utterance gets another dev
utterance of the batch mixed in (speech only), with a fixed seed per batch.
"""
import argparse
import logging
import random
import zlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from lhotse import CutSet, Fbank, FbankConfig, load_manifest_lazy
from lhotse.dataset import DynamicBucketingSampler
from lhotse.utils import fix_random_seed
from torch.utils.data import DataLoader

from icefall.utils import str2bool
from wavlm_augment import SPEECH, WavLMMixer

SR = 16000
LABEL_STRIDE = 320  # samples per label frame (WavLM conv front-end)
LABEL_WIN = 400  # receptive field of a label frame in samples
SIL = 500  # secondary-label class "no secondary speaker" (= number of k-means clusters)


class _SeedWorkers:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        fix_random_seed(self.seed + worker_id)


class FbankHubertDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        max_sample_size: int = 250000,
        random_crop: bool = True,
        mixer: Optional[WavLMMixer] = None,
        bi_label: bool = False,
        mix_seed: Optional[int] = None,
    ):
        """
        mixer: WavLM augmentation (None: no augmentation).
        bi_label: also return secondary labels ("sec_labels").
        mix_seed: if given, the mixer's randomness is seeded from it and the
          batch's cut ids (deterministic validation mixing).
        """
        super().__init__()
        self.max_sample_size = max_sample_size
        self.random_crop = random_crop
        self.mixer = mixer
        self.bi_label = bi_label
        self.mix_seed = mix_seed
        self.extractor = Fbank(FbankConfig(num_mel_bins=80))

    def __getitem__(self, cuts: CutSet) -> Dict[str, Any]:
        audios = []
        for c in cuts:
            a = c.load_audio()
            assert a.shape[0] == 1 and c.sampling_rate == SR, c.id
            audios.append(a[0])
        size = min(min(len(a) for a in audios), self.max_sample_size)
        n_lab = (size - LABEL_WIN) // LABEL_STRIDE + 1

        segs, labels = [], []
        for c, a in zip(cuts, audios):
            max_off = (len(a) - size) // LABEL_STRIDE
            k = random.randint(0, max_off) if (self.random_crop and max_off > 0) else 0
            off = k * LABEL_STRIDE
            segs.append(a[off : off + size])
            lab = np.array(c.custom["kmeans"].split(), dtype=np.int64)
            assert len(lab) == (len(a) - LABEL_WIN) // LABEL_STRIDE + 1, c.id
            labels.append(torch.from_numpy(lab[k : k + n_lab]))
        segs = np.stack(segs)  # (B, size)

        B = len(segs)
        mix_pos = np.full((B, 4), -1, dtype=np.int64)
        if self.mixer is not None:
            rng = None
            if self.mix_seed is not None:
                key = "|".join(c.id for c in cuts).encode()
                rng = random.Random(self.mix_seed * 1000003 + zlib.crc32(key))
            segs, mix_kinds, mix_pos = self.mixer(segs, rng=rng)
        else:
            mix_kinds = np.zeros(B, dtype=np.int64)

        feats = [torch.from_numpy(self.extractor.extract(seg, SR)) for seg in segs]
        features = torch.stack(feats)  # same length for all crops
        feature_lens = torch.full((len(feats),), features.shape[1], dtype=torch.int32)
        batch = {"features": features, "feature_lens": feature_lens,
                 "labels": labels, "cuts": cuts,
                 "mix_kinds": torch.from_numpy(mix_kinds)}
        if self.bi_label:
            batch["sec_labels"] = self._secondary_labels(labels, mix_kinds, mix_pos, n_lab)
        return batch

    @staticmethod
    def _secondary_labels(labels, mix_kinds, mix_pos, n_lab):
        """Secondary labels on the primary's label frames: the secondary
        utterance's labels inside the mixed region, SIL elsewhere. Label frame
        i covers samples [320 i, 320 i + 400); with s_pri, s_sec, l multiples
        of 320, frames s_pri/320 ... (s_pri + l)/320 - 1 (more than half
        inside the region) get secondary frame i - (s_pri - s_sec)/320."""
        sec_labels = []
        for i in range(len(labels)):
            sec = torch.full((n_lab,), SIL, dtype=torch.int64)
            if mix_kinds[i] == SPEECH:
                j, s_pri, s_sec, l = (int(x) for x in mix_pos[i])
                assert s_pri % LABEL_STRIDE == 0 and s_sec % LABEL_STRIDE == 0 \
                    and l % LABEL_STRIDE == 0, mix_pos[i]
                a, b, m = s_pri // LABEL_STRIDE, s_sec // LABEL_STRIDE, l // LABEL_STRIDE
                m = min(m, n_lab - a, len(labels[j]) - b)
                if m > 0:
                    sec[a : a + m] = labels[j][b : b + m]
            sec_labels.append(sec)
        return sec_labels


class FbankHubertDataModule:
    def __init__(self, args: argparse.Namespace):
        self.args = args

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        g = parser.add_argument_group("SSL data")
        g.add_argument("--manifest-dir", type=Path, default=Path("data/manifests"))
        g.add_argument("--max-duration", type=float, default=600.0,
                       help="Max total duration (s, before cropping) of a batch per GPU.")
        g.add_argument("--num-buckets", type=int, default=30)
        g.add_argument("--shuffle", type=str2bool, default=True)
        g.add_argument("--drop-last", type=str2bool, default=True)
        g.add_argument("--num-workers", type=int, default=8)
        g.add_argument("--max-sample-size", type=int, default=250000,
                       help="Crop utterances to at most this many samples (as HuBERT).")
        g.add_argument("--mix-noise-prob", type=float, default=0.1,
                       help="WavLM augmentation: probability that a training utterance "
                       "gets a DNS noise clip (energy ratio U(-5, 20) dB).")
        g.add_argument("--mix-speech-prob", type=float, default=0.1,
                       help="WavLM augmentation: probability that a training utterance "
                       "gets another utterance of the batch (energy ratio U(-5, 5) dB).")
        g.add_argument("--noise-list", type=Path, default=Path("data/noise/dns_noise.tsv"),
                       help="DNS noise list from local/prepare_dns_noise.py.")
        g.add_argument("--ami-repeat", type=int, default=1,
                       help="Up-sample AMI IHM train: each epoch contains every AMI piece "
                       "this many times (randomly interleaved with LibriSpeech). 1: natural "
                       "proportion (AMI ~4.9%% of the hours).")

    def train_dataloaders(self, cuts: CutSet, sampler_state_dict=None) -> DataLoader:
        bi_label = getattr(self.args, "bi_label", False)
        mixer = None
        if self.args.mix_noise_prob > 0 or self.args.mix_speech_prob > 0:
            mixer = WavLMMixer(self.args.noise_list, self.args.mix_noise_prob,
                               self.args.mix_speech_prob,
                               quantum=LABEL_STRIDE if bi_label else 1)
            logging.info(f"WavLM augmentation: noise {self.args.mix_noise_prob}, "
                         f"speech {self.args.mix_speech_prob}, bi-label {bi_label}")
        dataset = FbankHubertDataset(self.args.max_sample_size, random_crop=True,
                                     mixer=mixer, bi_label=bi_label)
        sampler = DynamicBucketingSampler(
            cuts, max_duration=self.args.max_duration, shuffle=self.args.shuffle,
            num_buckets=self.args.num_buckets, buffer_size=self.args.num_buckets * 5000,
            drop_last=self.args.drop_last,
        )
        if sampler_state_dict is not None:
            sampler.load_state_dict(sampler_state_dict)
        seed = torch.randint(0, 100000, ()).item()
        return DataLoader(dataset, sampler=sampler, batch_size=None,
                          num_workers=self.args.num_workers, persistent_workers=False,
                          worker_init_fn=_SeedWorkers(seed))

    def valid_dataloaders(self, cuts: CutSet, mixed: bool = False) -> DataLoader:
        """mixed: every utterance gets another utterance of the batch mixed in
        (speech only, fixed seed per batch); used for bi-label validation."""
        bi_label = getattr(self.args, "bi_label", False)
        mixer = WavLMMixer(None, prob_noise=0.0, prob_speech=1.0,
                           quantum=LABEL_STRIDE) if mixed else None
        dataset = FbankHubertDataset(self.args.max_sample_size, random_crop=False,
                                     mixer=mixer, bi_label=bi_label,
                                     mix_seed=0 if mixed else None)
        sampler = DynamicBucketingSampler(cuts, max_duration=self.args.max_duration, shuffle=False)
        return DataLoader(dataset, sampler=sampler, batch_size=None,
                          num_workers=2, persistent_workers=False)

    @lru_cache()
    def train_cuts(self) -> CutSet:
        logging.info("About to get train cuts (LibriSpeech 960 h + AMI IHM train, shuffled)")
        cuts = load_manifest_lazy(self.args.manifest_dir / "ssl_cuts_train-shuf.jsonl.gz")
        k = getattr(self.args, "ami_repeat", 1)
        if k == 1:
            return cuts
        # Same shuffled manifest split by source; AMI repeated k times (new cut
        # ids), then randomly interleaved with weights = number of cuts, so both
        # streams end together (fixed seed: identical on every DDP rank).
        ls = cuts.filter(lambda c: c.id.startswith("ls_"))
        ami = cuts.filter(lambda c: c.id.startswith("ami_ihm_train"))
        n_ls, n_ami = 281241, 36287  # ls_train-* / ami_ihm_train cuts
        logging.info(f"Up-sampling AMI x{k}: {n_ami * k} AMI + {n_ls} LibriSpeech cuts per epoch")
        return CutSet.mux(ls, ami.repeat(times=k), weights=[n_ls, n_ami * k], seed=0)

    @lru_cache()
    def dev_cuts(self) -> Dict[str, CutSet]:
        return {n: load_manifest_lazy(self.args.manifest_dir / f"ssl_cuts_{n}.jsonl.gz")
                for n in ["ls_dev-other", "ami_ihm_dev"]}
