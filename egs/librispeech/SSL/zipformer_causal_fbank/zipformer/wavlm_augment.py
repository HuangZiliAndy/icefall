"""
WavLM noisy/overlapped speech simulation (Chen et al., "WavLM: Large-Scale
Self-Supervised Pre-Training for Full Stack Speech Processing",
arXiv:2110.13900, Algorithm 1), applied to a batch of equal-length waveforms
before fbank extraction. Pseudo-labels stay those of the clean primary
utterance.

Each utterance of the batch independently gets one of
  - no augmentation          with probability 1 - prob_noise - prob_speech
  - a DNS noise clip         with probability prob_noise,  energy ratio r ~ U(-5, 20) dB
  - another utterance of the with probability prob_speech, energy ratio r ~ U(-5, 5) dB
    same batch (never itself)
(the paper describes this as Bernoulli(p) for "mix or not", then p_n for
noise vs. speech, and says "20 % utterances" with p_n = 10 %; the intended
setting is none / noise / speech = 0.8 / 0.1 / 0.1, which is the default here).

For a primary utterance u_pri of length L:
  l ~ U{1, ..., L // 2},  s_pri ~ U{0, ..., L - l},  s_sec ~ U{0, ..., len(u_sec) - l}
  scl = sqrt(E_pri / (10^(r/10) E_sec)),  E = mean energy of the whole signal
  u_pri[s_pri : s_pri + l] += scl * u_sec[s_sec : s_sec + l]
Since l <= L / 2, the primary speaker is always dominant in time.

Secondary utterances are taken from the batch before any mixing (a copy).
With quantum > 1 (bi-label training: quantum = 320, the label stride),
l, s_pri and s_sec are multiples of `quantum`, so the secondary utterance's
frame labels map exactly onto the primary's frames. The mixing position of
each utterance is returned (see __call__).
Noise clips shorter than l (DNS clips are mostly 10 s, some down to 1.5 s)
are repeated to length l. Mixing is skipped if either energy is 0.
"""
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf

NONE, NOISE, SPEECH = 0, 1, 2


class WavLMMixer:
    def __init__(
        self,
        noise_list: Path,
        prob_noise: float = 0.1,
        prob_speech: float = 0.1,
        noise_ratio_db: Tuple[float, float] = (-5.0, 20.0),
        speech_ratio_db: Tuple[float, float] = (-5.0, 5.0),
        quantum: int = 1,
    ):
        assert 0 <= prob_noise and 0 <= prob_speech and prob_noise + prob_speech <= 1
        assert quantum >= 1
        self.quantum = quantum
        self.prob_noise = prob_noise
        self.prob_speech = prob_speech
        self.noise_ratio_db = noise_ratio_db
        self.speech_ratio_db = speech_ratio_db
        self.noises: List[str] = []
        if prob_noise > 0:
            with open(noise_list) as f:
                root = Path(f.readline().rstrip())
                self.noises = [str(root / line.split("\t")[0]) for line in f]
            assert self.noises, noise_list

    def _load_noise(self, rng) -> np.ndarray:
        wav, sr = sf.read(rng.choice(self.noises), dtype="float32")
        assert sr == 16000 and wav.ndim == 1, (sr, wav.shape)
        return wav

    def __call__(
        self, wavs: np.ndarray, rng=None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        wavs: (B, L) float32, equal-length waveforms of a batch.
        rng: a random.Random (default: the module-level `random`, seeded per
          DataLoader worker); validation passes a seeded one.
        Returns:
          - the mixed waveforms (B, L);
          - the augmentation type of each utterance (B,) in {NONE, NOISE, SPEECH};
          - the mixing position (B, 4) int64: [secondary index in the batch
            (-1 for noise / none), s_pri, s_sec, l] (all -1 if not mixed).
        """
        rng = rng if rng is not None else random
        q = self.quantum
        B, L = wavs.shape
        orig = wavs
        out = wavs.copy()
        kinds = np.full(B, NONE, dtype=np.int64)
        pos = np.full((B, 4), -1, dtype=np.int64)
        for i in range(B):
            v = rng.random()
            sec_idx = -1
            if v < self.prob_speech:
                if B < 2:
                    continue
                j = rng.randrange(B - 1)
                sec_idx = j if j < i else j + 1  # any utterance but i
                sec = orig[sec_idx]
                lo, hi = self.speech_ratio_db
                kind = SPEECH
            elif v < self.prob_speech + self.prob_noise:
                sec = self._load_noise(rng)
                lo, hi = self.noise_ratio_db
                kind = NOISE
            else:
                continue

            if (L // 2) // q < 1:
                continue
            l = q * rng.randint(1, (L // 2) // q)
            if len(sec) < l:
                sec = np.tile(sec, -(-l // len(sec)))
            e_pri = float(np.mean(orig[i].astype(np.float64) ** 2))
            e_sec = float(np.mean(sec.astype(np.float64) ** 2))
            if e_pri == 0.0 or e_sec == 0.0:
                continue
            r = rng.uniform(lo, hi)
            scl = np.sqrt(e_pri / (10 ** (r / 10) * e_sec))
            s_pri = q * rng.randint(0, (L - l) // q)
            s_sec = q * rng.randint(0, (len(sec) - l) // q)
            out[i, s_pri : s_pri + l] += scl * sec[s_sec : s_sec + l]
            kinds[i] = kind
            pos[i] = (sec_idx, s_pri, s_sec, l)
        return out, kinds, pos
