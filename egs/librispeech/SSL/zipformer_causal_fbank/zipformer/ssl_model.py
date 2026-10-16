"""
HuBERT-style masked prediction with an fbank front-end and a causal Zipformer2,
built so that `encoder_embed` and `encoder` load unchanged into the ASR model
(egs/librispeech/ASR/zipformer/model.py, same parameter names).

  fbank (100 Hz) -> encoder_embed (Conv2dSubsampling, 50 Hz)
    -> span masking (masked frames := mask_emb)
    -> encoder (Zipformer2, causal, chunk-wise; output kept at 50 Hz)
    -> final_proj / logit_temp -> cross-entropy on masked frames only

Training recipe follows k2SSL (egs/librispeech/SSL/zipformer/hubert_ce.py):
mask_prob / mask_length span masking, CE with reduction "sum", masked-frame
loss weight 1 and unmasked weight 0, logit_temp. The wav2vec2 CNN front-end
specific parts (feature_grad_mult, feature L2 penalty, waveform
normalisation) do not apply to the fbank front-end and are omitted.
HuBERT's dropout_input is also omitted: encoder_embed already ends with a
dropout (ScheduledFloat 0.3 -> 0.1), so the path into the encoder is the same
as in the ASR model.

Output frame rate: the encoder is constructed exactly as for ASR
(output_downsampling_factor=2), then its final `downsample_output` is replaced
by an identity so that outputs stay at 50 Hz, the rate of the k-means labels.
(Zipformer2.forward then returns halved lengths; we keep our own lengths.)
When fine-tuning, only `encoder.downsample_output.bias` (2 numbers) is not
initialised from the pre-trained model.

Label alignment: encoder_embed frame j is centred at fbank frame 2j+4, i.e.
20j + 45 ms (snip_edges=False: fbank frame k centred at 10k + 5 ms; measured
with gradients); WavLM label i is centred at 320i + 200 samples = 20i + 12.5 ms.
So frame j is trained to predict label j + label_shift with label_shift = 2
(+7.5 ms).

Bi-label training (bi_label=True, off by default): a second head
final_proj_sec (num_classes + 1 outputs; class num_classes = SIL, "no
secondary speaker") predicts the secondary labels (see ssl_datamodule.py) on
the same masked frames; loss = CE_primary + sec_loss_scale * CE_secondary,
where in CE_secondary the frames with a secondary speaker (non-SIL) are
weighted by sec_spk_weight (1: plain CE).
With bi_label=False the model and its parameters are exactly as before.
"""
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mask_utils import compute_mask_indices

from icefall.utils import make_pad_mask


class FbankHubertModel(nn.Module):
    def __init__(
        self,
        encoder_embed: nn.Module,
        encoder: nn.Module,
        embed_dim: int,
        encoder_out_dim: int,
        num_classes: int = 500,
        label_shift: int = 2,
        mask_prob: float = 0.8,
        mask_length: int = 10,
        logit_temp: float = 0.1,
        bi_label: bool = False,
        sec_loss_scale: float = 1.0,
        sec_spk_weight: float = 1.0,
    ):
        super().__init__()
        self.encoder_embed = encoder_embed
        self.encoder = encoder
        # keep 50 Hz output (see module docstring)
        self.encoder.downsample_output = nn.Identity()
        self.mask_emb = nn.Parameter(torch.empty(embed_dim).uniform_())
        self.final_proj = nn.Linear(encoder_out_dim, num_classes)
        self.bi_label = bi_label
        self.sec_loss_scale = sec_loss_scale
        self.sec_spk_weight = sec_spk_weight
        if bi_label:
            self.final_proj_sec = nn.Linear(encoder_out_dim, num_classes + 1)
        self.num_classes = num_classes
        self.label_shift = label_shift
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.logit_temp = logit_temp

    def align_labels(self, labels: List[torch.Tensor], num_frames: int, device) -> torch.Tensor:
        """(N, num_frames) targets; frame j <- label j + label_shift, -100 where
        no label exists."""
        out = torch.full((len(labels), num_frames), -100, dtype=torch.long)
        for i, lab in enumerate(labels):
            lab = lab[self.label_shift : self.label_shift + num_frames]
            out[i, : len(lab)] = lab
        return out.to(device)

    def forward(
        self,
        features: torch.Tensor,
        feature_lens: torch.Tensor,
        labels: List[torch.Tensor],
        mask: bool = True,
        mask_seed: Optional[int] = None,
        sec_labels: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        features: (N, T, 80) fbank; feature_lens: (N,); labels: list of 1-D
        LongTensor (50 Hz k-means labels of each utterance).
        sec_labels: secondary labels (same frames as labels; required iff
        bi_label), values in [0, num_classes], num_classes = SIL.
        mask_seed: if given, masks are a deterministic function of it (used for
        validation); otherwise compute_mask_indices draws fresh OS entropy.
        Returns the summed CE over masked frames and stats.
        """
        x, x_lens = self.encoder_embed(features, feature_lens)  # (N, T', C)
        N, T, _ = x.shape
        padding_mask = make_pad_mask(x_lens, max_len=T)  # True = padding
        targets = self.align_labels(labels, T, x.device)
        valid = (~padding_mask) & (targets >= 0)

        if mask:
            mask_indices = compute_mask_indices(
                (N, T), padding_mask.cpu(), self.mask_prob, self.mask_length,
                "static", 0.0, min_masks=2,
                seed=mask_seed, epoch=0 if mask_seed is not None else None,
                indices=torch.arange(N) if mask_seed is not None else None,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x = x.clone()
            x[mask_indices] = self.mask_emb.to(x.dtype)
        else:
            mask_indices = torch.zeros_like(padding_mask)

        enc, _ = self.encoder(x.permute(1, 0, 2), x_lens, padding_mask)
        enc = enc.permute(1, 0, 2)  # (N, T', D), 50 Hz
        assert enc.shape[1] == T, (enc.shape, T)

        sel = mask_indices & valid if mask else valid
        logits = self.final_proj(enc[sel]).float() / self.logit_temp
        tgt = targets[sel]
        loss = F.cross_entropy(logits, tgt, reduction="sum")
        with torch.no_grad():
            correct = (logits.argmax(-1) == tgt).sum()
        stats = {
            "frames": sel.sum(),
            "correct": correct,
            "masked_frac": (mask_indices & valid).sum() / valid.sum().clamp(min=1),
        }

        if self.bi_label:
            assert sec_labels is not None
            sec_targets = self.align_labels(sec_labels, T, x.device)[sel]
            assert (sec_targets >= 0).all()  # same frames as the primary labels
            sec_logits = self.final_proj_sec(enc[sel]).float() / self.logit_temp
            sec_ce = F.cross_entropy(sec_logits, sec_targets, reduction="none")
            spk = sec_targets != self.num_classes  # frames with a secondary speaker
            # optimised: secondary-speech frames weighted by sec_spk_weight (SIL: 1);
            # logged sec_loss: unweighted, comparable across settings
            sec_loss_w = torch.where(spk, sec_ce * self.sec_spk_weight, sec_ce).sum()
            sec_loss = sec_ce.sum()
            with torch.no_grad():
                hit = sec_logits.argmax(-1) == sec_targets
            stats.update(
                pri_loss=loss.detach(),
                sec_loss=sec_loss.detach(),
                sec_correct=hit.sum(),
                sec_spk_frames=spk.sum(),
                sec_spk_correct=(hit & spk).sum(),
                sec_sil_frames=(~spk).sum(),
                sec_sil_correct=(hit & ~spk).sum(),
            )
            loss = loss + self.sec_loss_scale * sec_loss_w
        return loss, stats
