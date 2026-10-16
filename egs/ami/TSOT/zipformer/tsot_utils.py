"""
Helpers for t-SOT transcripts, i.e. text with "<cc>" (channel change) tokens.

sentencepiece's own encode() turns the space in front of a user-defined symbol
into a lone "▁" piece ("A <cc> B" -> ▁A ▁ <cc> ▁B), and decode() glues <cc> to
its neighbours ("A<cc> B"). So t-SOT text must always be encoded/decoded with
the functions below: each chunk between <cc> is processed on its own and the
<cc> id is inserted/split on explicitly.
"""
from typing import List, Tuple, Union

import sentencepiece as spm

CC = "<cc>"


def encode_tsot(
    sp: spm.SentencePieceProcessor, texts: Union[str, List[str]]
) -> Union[List[int], List[List[int]]]:
    """Encode one t-SOT text (or a list of them) into token ids."""
    if not isinstance(texts, str):
        return [encode_tsot(sp, t) for t in texts]
    cc_id = sp.piece_to_id(CC)
    assert cc_id != sp.unk_id(), f"{CC} is not in the BPE model"
    ids: List[int] = []
    for k, chunk in enumerate(texts.split(CC)):
        if k > 0:
            ids.append(cc_id)
        chunk = chunk.strip()
        if chunk:
            ids.extend(sp.encode(chunk, out_type=int))
    return ids


def decode_tsot(sp: spm.SentencePieceProcessor, ids: List[int]) -> str:
    """Decode token ids into a t-SOT text "w1 w2 <cc> w3 ...".

    Every <cc> is kept (also leading, trailing or repeated ones), so that
    split_channels() on the result toggles channels exactly as the model did.
    """
    cc_id = sp.piece_to_id(CC)
    chunks: List[List[int]] = [[]]
    for i in ids:
        if i == cc_id:
            chunks.append([])
        else:
            chunks[-1].append(i)
    text = f" {CC} ".join(sp.decode(c).strip() for c in chunks)
    return " ".join(text.split())


def split_channels(text: str, num_channels: int = 2) -> Tuple[str, ...]:
    """Split a t-SOT text into virtual channels: start on channel 0 and switch
    to the next channel at every <cc>."""
    chans: List[List[str]] = [[] for _ in range(num_channels)]
    for k, chunk in enumerate(text.split(CC)):
        chunk = chunk.strip()
        if chunk:
            chans[k % num_channels].append(chunk)
    return tuple(" ".join(c) for c in chans)
