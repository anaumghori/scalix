from typing import Callable, Optional
import functools
import torch
from torch.nn.attention.flex_attention import create_block_mask


# ── Score modification ────────────────────────────

def build_score_mod(
    softcap: Optional[float],
    causal_mask: Optional[torch.Tensor],
) -> Optional[Callable]:
    # If neither modifier is requested, there is nothing to compose; return early.
    if softcap is None and causal_mask is None:
        return None

    # score_mod is a pointwise function applied to each attention logit before softmax.
    # It receives the raw dot-product score and the (batch, head, query, key) indices.
    def score_mod(score: torch.Tensor, b: int, h: int, q_idx: int, kv_idx: int) -> torch.Tensor:
        if softcap is not None:
            # Softcapping squashes logits into (-softcap, softcap) via a scaled tanh,
            # preventing extreme values from dominating softmax.
            score = softcap * torch.tanh(score / softcap)
        if causal_mask is not None:
            # The causal_mask tensor stores additive biases; positions that should be
            # masked carry a large negative value so they contribute nothing after softmax.
            score = score + causal_mask[b][0][q_idx][kv_idx]
        return score

    return score_mod


def build_attention_mod(
    flex_attention_mask: Optional[str],
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> Optional[Callable]:
    # Score modification is not applied through this path; returns None unconditionally.
    return None


# ── Primitive mask functions ────────────────────────────

# These are the elementary building blocks used to compose more complex masks
# such as document-aware or sliding-window variants. They follow the signature
# expected by create_block_mask: (b, h, q_idx, kv_idx) -> bool.


def causal_mask(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
    # A query position can only attend to key positions that came before it or at the same step.
    return q_idx >= kv_idx


def sliding_window_mask(window_size: int, b: int, h: int, q_idx: int, kv_idx: int) -> bool:
    # Combines standard causality with a maximum lookback distance.
    # Positions further than window_size steps in the past are masked out.
    causal = q_idx >= kv_idx
    window = q_idx - kv_idx <= window_size
    return causal & window


def lengths_to_offsets(lengths: torch.Tensor, device: torch.device) -> torch.Tensor:
    # Converts a sequence of document lengths into a cumulative offsets tensor.
    # Accepts either a plain Python list or a 1D tensor of lengths.
    # Example: lengths = [3, 2, 4] , offsets = [0, 3, 5, 9]
    if not isinstance(lengths, torch.Tensor):
        offsets = [0]
        offsets.extend(lengths)
        offsets = torch.tensor(offsets, device=device, dtype=torch.int32)
    else:
        offsets = torch.cat([torch.zeros(1, device=device, dtype=torch.int32), lengths])
    offsets = torch.cumsum(offsets, dim=-1)
    return offsets


def _offsets_to_doc_ids(offsets: torch.Tensor) -> torch.Tensor:
    # Converts an offsets boundary tensor into a flat per-token document ID tensor.
    # Example: offsets = [0, 3, 5, 9], We compute: counts = [3, 2, 4], Then expand: doc_ids = [0, 0, 0, 1, 1, 2, 2, 2, 2]
    device = offsets.device
    counts = offsets[1:] - offsets[:-1]  # length of each document.
    return torch.repeat_interleave(
        torch.arange(len(counts), device=device, dtype=torch.int32), counts
    )


def build_doc_mask(inner_mask_func: Callable, offsets: torch.Tensor) -> Callable:
    # Wraps an arbitrary inner mask function with document-boundary awareness.
    # Two tokens may only interact if they belong to the same document AND satisfy
    # the inner mask (e.g. causal). Query and key indices are converted to
    # document-local positions before being passed to the inner mask so that
    # the inner function does not need to be aware of packing.
    document_ids = _offsets_to_doc_ids(offsets)

    def doc_mask_mod(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
        same_doc = document_ids[q_idx] == document_ids[kv_idx]  # Prevent cross-document attention
        # Convert global token indices to document-local indices before applying the inner mask.
        q_local_idx = q_idx - offsets[document_ids[q_idx]]
        kv_local_idx = kv_idx - offsets[document_ids[kv_idx]]
        inner_mask = inner_mask_func(b, h, q_local_idx, kv_local_idx)
        return same_doc & inner_mask

    return doc_mask_mod


# ── Block mask construction ────────────────────────────

# The attention mask starts as a rule: given (q_idx, kv_idx), decide if attention is allowed.
# The kernel instead needs a full tensor encoding this rule for all index pairs.
# create_block_mask performs this function-to-tensor conversion, which is expensive
# because it evaluates the mask over the full Q_LEN × KV_LEN grid. This wrapper
# memoizes the result so that identical masking rules and sequence shapes reuse the
# same materialized mask instead of recomputing it.

@functools.lru_cache(maxsize=32)
def cached_block_mask(
    mask_func: Callable,
    B: Optional[int],
    H: Optional[int],
    Q_LEN: int,
    KV_LEN: int,
    device: str = "cuda",
) -> torch.Tensor:
    return create_block_mask(mask_func, B, H, Q_LEN, KV_LEN, device=device)


# Sliding window masking is defined by window_size, but turning it into a function
# (via partial) creates a new object each time. Since cached_block_mask keys on the 
# function object, using partial directly would prevent cache reuse. This function 
# avoids that by caching on the primitive parameters (window_size, seq_len, device), 
# which uniquely determine the mask.

@functools.lru_cache(maxsize=32)
def _cached_sliding_window_block_mask(window_size: int, seq_len: int, device: str) -> torch.Tensor:
    mask_func = functools.partial(sliding_window_mask, window_size)
    return create_block_mask(mask_func, B=None, H=None, Q_LEN=seq_len, KV_LEN=seq_len, device=device)


def build_block_mask(
    flex_attention_mask: Optional[str],
    query: torch.Tensor,
    key: torch.Tensor,
    sliding_window: Optional[int] = None,
    position_ids: Optional[torch.Tensor] = None,
    document_ids: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:

    # Returns None when flex_attention is disabled or the mask type is "causal",
    # since FlexAttention handles standard causality natively without a block mask.
    if flex_attention_mask is None:
        return None
    seq_len = query.size(2)
    device = query.device
    if flex_attention_mask == "causal":
        return None

    elif flex_attention_mask == "sliding_window":
        assert sliding_window is not None and sliding_window > 0, (
            f"For 'sliding_window' mask type, sliding_window must be provided and > 0, got {sliding_window}"
        )
        mask_func = functools.partial(sliding_window_mask, sliding_window)
        return _cached_sliding_window_block_mask(sliding_window, seq_len, str(device))


    elif flex_attention_mask == "document":
        assert document_ids is not None or position_ids is not None, (
            "For 'document' mask type, either document_ids or position_ids must be provided"
        )
        if document_ids is not None:
            document_ids = document_ids.to(device)
            assert document_ids.numel() > 0, "document_ids tensor cannot be empty"

            # The offsets conversion below counts tokens per document in sorted ID order, which
            # implicitly assumes all tokens for a given document appear contiguously. Interleaved
            # IDs such as [0, 1, 0, 1] would produce offsets that do not reflect actual token
            # positions, causing build_doc_mask to compute incorrect local indices.
            # Document IDs: [0, 0, 0, 1, 1, 2, 2, 2, 2] -> 0 tokens are doc one, 1 tokens are doc two and more...

            assert torch.all(document_ids[1:] >= document_ids[:-1]), (
                "document_ids must be monotonically non-decreasing; interleaved document IDs are not supported"
            )

            # Convert explicit per-token document IDs into a boundary offsets tensor.
            # Iterates over unique document IDs in sorted order and accumulates lengths.
            unique_docs = torch.unique(document_ids, sorted=True)
            offsets = [0]
            current_offset = 0
            for doc_id in unique_docs:
                doc_length = (document_ids == doc_id).sum().item()
                current_offset += doc_length
                offsets.append(current_offset)
            offsets = torch.tensor(offsets, device=device, dtype=torch.int32)

        
        # Encode position within a document -> [0, 1, 2, 0, 1, 0, 1, 2, 3]
        # Interpretation: every time we see 0, a new document starts

        elif position_ids is not None:
            position_ids = position_ids.to(device)
            assert position_ids.numel() > 0, "position_ids tensor cannot be empty"

            if position_ids.dim() == 2:
                position_ids = position_ids.view(-1)
            doc_starts = torch.where(position_ids == 0)[0]
            assert doc_starts.numel() > 0, "No document boundaries found in position_ids (no zeros found)"

            # Infer document end positions from the start of the next document, or the end of the sequence.
            doc_ends = torch.cat([doc_starts[1:], torch.tensor([len(position_ids)], device=device)])
            doc_lengths = doc_ends - doc_starts
            offsets = lengths_to_offsets(doc_lengths, device)

            # If the first document does not start at token 0, shift all offsets accordingly.
            # When a slice of the full sequence is provided for efficiency, lengths_to_offsets 
            # produces offsets relative to that slice. Shifting by doc_starts[0] converts them 
            # to absolute positions in the full sequence. Example: slice tokens = [B1,B2,C1,C2,C3,C4], 
            # local offsets = [0,2,6]; slice starts at global index 3 → global offsets = [3,5,9].

            if doc_starts[0] != 0:
                offsets = offsets + doc_starts[0]

        # Wrap the causal primitive with document-boundary awareness.
        doc_mask_mod = build_doc_mask(causal_mask, offsets)

        return cached_block_mask(
            doc_mask_mod,
            B=None,  # Not needed since document boundaries are handled inside the mask function.
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=device,
        )

    else:
        raise ValueError(
            f"Unknown flex_attention_mask type: {flex_attention_mask}. "
            f"Supported types are: 'causal', 'sliding_window', 'document'"
        )


# ── Argument validation ────────────────────────────


def validate_flex_attention_args(
    flex_attention_mask: Optional[str],
    sliding_window: Optional[int] = None,
    position_ids: Optional[torch.Tensor] = None,
    document_ids: Optional[torch.Tensor] = None,
) -> None:
    if flex_attention_mask is None:
        return

    if flex_attention_mask == "causal":
        return

    elif flex_attention_mask == "sliding_window":
        if sliding_window is None or sliding_window <= 0:
            raise ValueError(
                f"For 'sliding_window' mask type, sliding_window must be a positive integer, got {sliding_window}"
            )

    elif flex_attention_mask == "document":
        if document_ids is None and position_ids is None:
            raise ValueError(
                "For 'document' mask type, either document_ids or position_ids must be provided"
            )
        if position_ids is not None and position_ids.numel() == 0:
            raise ValueError("position_ids tensor cannot be empty")
        if document_ids is not None and document_ids.numel() == 0:
            raise ValueError("document_ids tensor cannot be empty")

    else:
        raise ValueError(
            f"Unknown flex_attention_mask type: {flex_attention_mask}. "
            f"Supported types are: 'causal', 'sliding_window', 'document'"
        )