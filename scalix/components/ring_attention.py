""" Adapted from https://github.com/zhuzilin/ring-flash-attention; original author: Zhuzilin"""

import torch
import triton
import triton.language as tl
import inspect
from functools import cache
from typing import Optional, Tuple
import torch.nn.functional as F
import torch.distributed as dist
from flash_attn.flash_attn_interface import _flash_attn_varlen_backward, _flash_attn_varlen_forward

__all__ = ["update_out_and_lse", "RingComm", "get_default_args"]


# ── Argument inspection utilities ──────────────────────────────────────────────


@cache
def _get_default_args(func):
    spec = inspect.getfullargspec(func)
    defaults = spec.defaults if spec.defaults is not None else ()
    # Example: args = [a, b, c, d] and defaults = (3, 4)
    # After padding: (None, None, 3, 4)
    padded_defaults = (None,) * (len(spec.args) - len(defaults)) + defaults
    args = dict(zip(spec.args, padded_defaults))  # creates a dictionary mapping argument names to default values

    # Some versions of Flash Attention include a softcap argument controlling the maximum value
    # of attention logits. If the argument exists, the code forces its default value to 0.0.
    if "softcap" in args:
        args["softcap"] = 0.0
    return args


def get_default_args(func):
    if inspect.isfunction(func):
        return _get_default_args(func)
    else:
        # Use the origin _init_fn in CustomOpDef
        return _get_default_args(func._init_fn)


# ── LSE shape utilities ────────────────────────────────────────────────────────
#
# Ring attention requires converting log-sum-exp (LSE) tensors between two representations:
#   Padded:    (batch_size, nheads, max_seqlen)  — one row per sequence, zero-padded
#   Flattened: (nheads, total_seqlen)            — sequences concatenated along seqlen
#
# Two implementations are provided for each direction:
#   1. Triton-accelerated kernels (_flatten_varlen_lse_triton / _unflatten_varlen_lse_triton).
#      These are alternative implementations; they are currently inactive.
#   2. torch.jit.script implementations (flatten_varlen_lse / unflatten_varlen_lse).
#      These are the active implementations used throughout the module.

# The TorchScript versions iterate over the batch dimension and process one sequence at a time, 
# relying on PyTorch kernels for memory movement. The Triton versions parallelize across batch, 
# head, and position blocks simultaneously, assigning each tile to an independent GPU thread block. 
# This exposes greater parallelism for large batches or many heads. For small batches, the 
# difference is minor and TorchScript remains simpler to debug.

# A secondary difference is control over memory layout. The Triton kernels receive explicit strides 
# for every dimension, so they can correctly handle non-contiguous tensors without first making them 
# contiguous. The TorchScript versions rely on PyTorch's indexing, which handles this transparently 
# but with less control over the resulting memory access pattern and potential intermediate allocations.


@torch.jit.script
def flatten_varlen_lse(lse, cu_seqlens):
    # lse.shape = (num_seq, num_heads, max_seqlen)
    new_lse = []
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        # Select sequence i. Using an integer index removes the sequence dimension, so the result has shape (num_heads, sequence_length_i).
        new_lse.append(lse[i, :, : end - start])  # -> (num_heads, sequence_length_i)
    return torch.cat(new_lse, dim=1)  # (4 heads, 5 tokens), (4 heads, 4 tokens) -> (4 heads, 9 tokens)


@torch.jit.script
def unflatten_varlen_lse(lse, cu_seqlens, max_seqlen: int):
    # lse.shape = (total_tokens, num_heads, 1)
    num_seq = len(cu_seqlens) - 1
    num_head = lse.shape[-2]
    new_lse = torch.empty((num_seq, max_seqlen, num_head, 1), dtype=torch.float32, device=lse.device)
    for i in range(num_seq):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        # Restore sequence i by copying its token range [start:end] from the flattened lse into the valid token region of the padded tensor
        new_lse[i, : end - start] = lse[start:end]
    return new_lse.squeeze(dim=-1).transpose(1, 2).contiguous() # -> (num_seq, num_head, max_seqlen)


@triton.jit
def _flatten_varlen_lse_kernel(
    # Device pointers to tensors in GPU memory.
    OUT,
    LSE, # the log-sum-exp values used during the numerically stable softmax computation inside attention
    # Cumulative sequence lengths, used for handling variable-length sequences efficiently. Tells the function where
    # each sequence starts and ends in a flattened tensor instead of padding all sequences to the same length.
    CU_SEQLENS, # [5, 3, 7] -> [0, 5, 8, 15]

    # Stride arguments describe how to move through the tensor memory layout. 
    # Triton kernels do not assume contiguous layout, so explicit strides are passed.
    stride_out_nheads,
    stride_out_seqlen,
    stride_lse_batch,
    stride_lse_nheads,
    stride_lse_seqlen,
    BLOCK_M: tl.constexpr,
):
    # Each Triton program is identified by coordinates in a grid.
    pid_m = tl.program_id(axis=0)      # indexes blocks along the sequence dimension.
    pid_batch = tl.program_id(axis=1)  # indexes the batch dimension.
    pid_head = tl.program_id(axis=2)   # indexes attention heads.

    # Next the kernel retrieves sequence boundaries. Example: [0, 12, 20, 37]. Sequence lengths are derived as differences:
    # sequence 0 length = 12
    # sequence 1 length = 8
    # sequence 2 length = 17
    # The kernel uses this array to determine where each sequence begins in the flattened representation.

    start_idx = tl.load(CU_SEQLENS + pid_batch)               # gives the position of the current sequence in the flattened tensor.
    seqlen = tl.load(CU_SEQLENS + pid_batch + 1) - start_idx  # gives the true length of the sequence.
    # Shift pointers to the correct batch and head so the kernel works on the relevant slice.
    LSE = LSE + pid_batch * stride_lse_batch + pid_head * stride_lse_nheads  # the offset selects the appropriate batch element and head.
    OUT = OUT + pid_head * stride_out_nheads + start_idx * stride_out_seqlen  # selects the correct head and the starting position of the flattened sequence.

    # The kernel computes the row indices handled by this program.
    # BLOCK_M elements are processed per program. For example if BLOCK_M = 4 and pid_m = 2 then row_indices becomes: [8, 9, 10, 11]
    # These are the sequence indices that the program will process.
    row_indices = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # Build LSE read addresses. row_indices[:, None] forms a column; stride_lse_seqlen converts rows to offsets.
    LSE = LSE + row_indices[:, None] * stride_lse_seqlen 

    # Load values with a mask: only positions where row_indices < seqlen are valid.
    # Padded positions return 0.0, preventing reads from invalid memory.
    x = tl.load(LSE, mask=row_indices[:, None] < seqlen, other=0.0)
    OUT = OUT + row_indices[:, None] * stride_out_seqlen  # Compute output addresses for writing flattened values.
    tl.store(OUT, x, mask=row_indices[:, None] < seqlen)  # Write results; the mask prevents writes beyond the true sequence length.


@triton.jit
def _unflatten_varlen_lse_kernel(
    OUT,
    LSE,
    CU_SEQLENS,
    stride_out_batch,
    stride_out_nheads,
    stride_out_seqlen,
    stride_lse_seqlen,
    stride_lse_nheads,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    pid_head = tl.program_id(axis=2)
    start_idx = tl.load(CU_SEQLENS + pid_batch)
    seqlen = tl.load(CU_SEQLENS + pid_batch + 1) - start_idx
    LSE = LSE + pid_head * stride_lse_nheads + start_idx * stride_lse_seqlen
    OUT = OUT + pid_batch * stride_out_batch + pid_head * stride_out_nheads
    row_indices = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    LSE = LSE + row_indices[:, None] * stride_lse_seqlen
    x = tl.load(LSE, mask=row_indices[:, None] < seqlen, other=0.0)
    OUT = OUT + row_indices[:, None] * stride_out_seqlen
    tl.store(OUT, x, mask=row_indices[:, None] < seqlen)


def _flatten_varlen_lse_triton(lse, cu_seqlens):
    """Triton-accelerated alternative to flatten_varlen_lse. Currently inactive."""
    cu_seqlens = cu_seqlens.to(lse.device)
    # lse: (batch_size, nheads, max_seqlen)
    # cu_seqlens: (batch_size + 1,)
    total_seqlen = cu_seqlens[-1]
    batch_size, nheads, max_seqlen = lse.shape
    output = torch.empty((nheads, total_seqlen), dtype=lse.dtype, device=lse.device)

    def grid(META):
        return triton.cdiv(max_seqlen, META["BLOCK_M"]), batch_size, nheads

    BLOCK_M = 4

    with torch.cuda.device(lse.device.index):
        _flatten_varlen_lse_kernel[grid](
            output,
            lse,
            cu_seqlens,
            output.stride(0),
            output.stride(1),
            lse.stride(0),
            lse.stride(1),
            lse.stride(2),
            BLOCK_M,
        )

    return output


def _unflatten_varlen_lse_triton(lse, cu_seqlens, max_seqlen: int):
    """Triton-accelerated alternative to unflatten_varlen_lse. Currently inactive."""
    cu_seqlens = cu_seqlens.to(lse.device)
    lse = lse.unsqueeze(dim=-1)
    batch_size = len(cu_seqlens) - 1
    nheads = lse.shape[1]
    output = torch.empty((batch_size, nheads, max_seqlen), dtype=lse.dtype, device=lse.device)

    def grid(META):
        return triton.cdiv(max_seqlen, META["BLOCK_M"]), batch_size, nheads

    BLOCK_M = 4

    with torch.cuda.device(lse.device.index):
        _unflatten_varlen_lse_kernel[grid](
            output,
            lse,
            cu_seqlens,
            output.stride(0),
            output.stride(1),
            output.stride(2),
            lse.stride(0),
            lse.stride(1),
            BLOCK_M,
        )
    return output


# ── Output accumulation ────────────────────────────────────────────────────────

# Conceptually this function performs the following role inside ring attention. During distributed attention computation:
# each GPU processes a subset of keys and values, each step produces partial attention outputs and then those partial
# outputs must be merged while maintaining correct softmax normalization. This function implements that merge operation
# in a numerically stable way.


@torch.jit.script
def _update_out_and_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)

    # new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))
    # torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out
    out = out - F.sigmoid(block_lse - lse) * (out - block_out)
    lse = lse - F.logsigmoid(lse - block_lse)
    return out, lse


def update_out_and_lse(
    out: Optional[torch.Tensor],   # accumulated attention output so far.
    lse: Optional[torch.Tensor],   # accumulated log-sum-exp normalization term.
    block_out: torch.Tensor,       # output computed for the current attention block.
    block_lse: torch.Tensor,       # log-sum-exp value produced for that block.
    slice_=None,                   # optional indexing specification that allows updating only a portion of the output tensors.
) -> Tuple[torch.Tensor, torch.Tensor]:

    # This condition identifies the first block processed in the accumulation process. In ring attention, attention is computed
    # in pieces because keys and values circulate between devices. During the first iteration there is no existing state, so the
    # running tensors must be initialized.
    if out is None:
        if slice_ is not None:
            raise RuntimeError("first update_out_and_lse should not pass slice_ args")
        out = block_out.to(torch.float32)
        lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)

    # This mode is used when the block result corresponds only to a subset of tokens. This occurs in two common scenarios:
    # 1. Variable-length sequences packed in a batch, 2. Ring attention updates where only a local query segment is updated.
    elif slice_ is not None:
        slice_out, slice_lse = out[slice_], lse[slice_]
        slice_out, slice_lse = _update_out_and_lse(slice_out, slice_lse, block_out, block_lse)
        out[slice_], lse[slice_] = slice_out, slice_lse

    # Handles the normal case where the entire tensor is updated. Directly merges the new block with the accumulated state.
    else:
        out, lse = _update_out_and_lse(out, lse, block_out, block_lse)
    return out, lse


# ── Communication classes ──────────────────────────────────────────────────────

# each GPU initially holds only a local shard of the key/value tensors. Instead of collecting the entire tensors on every 
# device, the tensors circulate around the GPUs. Each GPU computes partial attention using the shard currently available 
# and then forwards that shard to the next device. After world_size steps, every GPU has processed attention with all shards.

# In PyTorch distributed training there is usually a global communicator called the world group. Every process in the distributed 
# job has a unique integer identifier in this world group. Example: `Global ranks: 0 1 2 3 4 5 6 7`

# Many distributed algorithms create subgroups of processes. For example: 
# `Group A: processes {0,1,2,3}` and `Group B: processes {4,5,6,7}`
# Inside a group, ranks are reindexed locally. 
# `Group A local ranks: 0 1 2 3` and `Group B local ranks: 0 1 2 3`
# A process whose global rank is 5, inside Group B its local rank becomes 1.


class RingComm:
    def __init__(self, process_group: dist.ProcessGroup):
        self._process_group = process_group
        self._ops = []  # will accumulate pending communication operations.
        self.rank = dist.get_rank(self._process_group)  # This returns the rank inside the group, not the global rank.
        self.world_size = dist.get_world_size(self._process_group)  # number of processes in the group.
        self._reqs = None
        # class determines which processes are neighbors in the ring. Produces group-local ranks.
        self.send_rank = (self.rank + 1) % self.world_size
        self.recv_rank = (self.rank - 1) % self.world_size

        # point-to-point operations (isend, irecv) expect global ranks as the destination process identifier.
        # Therefore the local ranks must be converted.
        if process_group is not None:
            self.send_rank = dist.get_global_rank(self._process_group, self.send_rank)  # (group, local_rank) → global_rank
            self.recv_rank = dist.get_global_rank(self._process_group, self.recv_rank)

    def send_recv(self, to_send: torch.Tensor, recv_tensor: Optional[torch.Tensor] = None):

        # If `recv_tensor` is provided, the receive operation writes into it. Otherwise a buffer is created with which
        # matches the shape, dtype, and device of the sent tensor. This allows simple usage: next_k = comm.send_recv(k)

        # while still supporting buffer reuse to avoid repeated allocations:
        # buffer = torch.empty_like(k)
        # next_k = comm.send_recv(k, buffer)

        # Internally, the receive operation requires a destination buffer because `dist.irecv` writes directly 
        # into existing memory. Therefore the receive buffer must exist before the communication begins.

        if recv_tensor is None:
            res = torch.empty_like(to_send)
        else:
            res = recv_tensor
        send_op = dist.P2POp(dist.isend, to_send, self.send_rank, group=self._process_group)
        recv_op = dist.P2POp(dist.irecv, res, self.recv_rank, group=self._process_group)
        # `isend` and `irecv` are not executed immediately. Instead they are appended to `_ops` which 
        # allows multiple send/receive pairs to be batched together and launched simultaneously. 
        self._ops.append(send_op)
        self._ops.append(recv_op)
        return res

    def commit(self):
        # This ensures we do not start communication twice without finishing the previous one.
        if self._reqs is not None:
            raise RuntimeError("commit called twice")
        self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self):
        if self._reqs is None:
            raise RuntimeError("wait called before commit")
        for req in self._reqs:
            req.wait()
        self._reqs = None
        self._ops = []

    def send_recv_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        k_buffer: Optional[torch.Tensor] = None,
        v_buffer: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        next_k, next_v = self.send_recv(k, k_buffer), self.send_recv(v, v_buffer)
        self.commit()
        return next_k, next_v


class AllGatherComm:
    def __init__(self, group=None) -> None:
        self.group = group
        self.handles = []

    def all_gather(self, output_tensor: torch.Tensor, input_tensor: torch.Tensor):
        handle = dist.all_gather_into_tensor(output_tensor, input_tensor, group=self.group, async_op=True)
        self.handles.append(handle)

    def wait(self):
        for handle in self.handles:
            handle.wait()
        self.handles = []


# ── Core ring attention forward and backward ───────────────────────────────────


def ring_attn_varlen_forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens,
    max_seqlen,  # maximum sequence length in the batch, used to allocate memory efficiently.
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    comm = RingComm(process_group)
    out = None  # will store the final attention outputs after combining contributions from all GPUs.
    lse = None  # stores the log-sum-exp values from the softmax calculation.
    next_k, next_v = None, None  # placeholders for keys and values received from the next GPU in the ring.
    lse_is_padded = False  # indicating whether the lse tensor is in the old (padded 3D) shape format, used later for reshaping.

    # The idea: each GPU will sequentially receive key/value pairs from other GPUs and compute partial attention
    # contributions, updating out incrementally.
    for step in range(comm.world_size):  # this loop iterates once for each GPU in the ring. `comm.world_size` is the total number of GPUs.
        if step + 1 != comm.world_size:  # avoids sending after the last step because there's nothing left to send.
            next_k, next_v = comm.send_recv_kv(k, v)  # Sends current k, v to next GPU in the ring and receives k, v from previous GPU.

        # This condition decides whether the current GPU should compute attention for this step: If it's not 
        # causal, compute attention regardless of step. If causal, only compute attention with GPUs whose rank 
        # is <= current GPU rank. Ensures future tokens are not attended to in causal attention.
        if not causal or step <= comm.rank:
            params = get_default_args(_flash_attn_varlen_forward).copy()
            params.update(
                {
                    "q": q,
                    "k": k,
                    "v": v,
                    "cu_seqlens_q": cu_seqlens,
                    "cu_seqlens_k": cu_seqlens,
                    "max_seqlen_q": max_seqlen,
                    "max_seqlen_k": max_seqlen,
                    "dropout_p": dropout_p,
                    "softmax_scale": softmax_scale,
                    # only the first step is causal; later steps in the ring contain future tokens that must not be attended.
                    "causal": causal and step == 0,
                    "alibi_slopes": alibi_slopes,
                    "return_softmax": True and dropout_p > 0,  # needed for dropout or logging the attention probabilities.
                }
            )

            if "window_size" in params:
                params.update({"window_size": window_size})
            else:
                params.update(
                    {
                        "windows_size_left": window_size[0],
                        "window_size_right": window_size[1],
                    }
                )

            # Calls Flash Attention kernel on variable-length sequences; computes attention with q, k, v tensors 
            # and returns output plus log-sum-exp for stability.
            outputs = _flash_attn_varlen_forward(**params)

            # Handles different output formats depending on Flash Attention version. Other outputs are ignored here.
            # block_out: the attention output for this GPU/block, block_lse: the log-sum-exp from softmax.
            if len(outputs) == 8:
                block_out, _, _, _, _, block_lse, _, _ = outputs
            else:
                assert len(outputs) == 4
                block_out, block_lse, _, _ = outputs

            # Flash Attention may return 3D LSE in older versions; flatten_varlen_lse converts it into a flattened 
            # shape compatible with this implementation. Sets lse_is_padded=True for later unflattening.
            if block_lse.dim() == 3:
                lse_is_padded = True
                block_lse = flatten_varlen_lse(block_lse, cu_seqlens)


            # To use Triton version instead of torch.jit
            # if block_lse.dim() == 3:
            #     lse_is_padded = True
            #     block_lse = _flatten_varlen_lse_triton(block_lse, cu_seqlens)


            # Combines this block's attention and LSE with previous steps, accumulating across GPUs to produce the 
            # complete attention output over all sequences.
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)

        # Synchronizes GPU communication; waits for send/receive to finish and updates k, v to avoid overwriting before next step.
        if step + 1 != comm.world_size:
            comm.wait()
            k, v = next_k, next_v

    out = out.to(q.dtype)  # Flash Attention may internally use higher precision so convert the output back to the same data type as q.
    if lse_is_padded:  # Restores the LSE tensor to the expected shape.
        lse = unflatten_varlen_lse(lse, cu_seqlens, max_seqlen)  # unflatten_varlen_lse for old (padded) format.

    # Triton version
    # if lse_is_padded:
    #     lse = _unflatten_varlen_lse_triton(lse, cu_seqlens, max_seqlen)

    else:
        lse = lse.squeeze(dim=1).transpose(0, 1)  # Otherwise, squeeze dim 1 and transpose for correct alignment.
    return out, lse


# Mirrors the forward pass logic but adds gradient accumulation, distributed communication, and memory handling.
def ring_attn_varlen_backward(
    process_group,
    dout,         # gradient of the loss with respect to the output of the attention (from the upstream layers).
    q,
    k,
    v,
    out,
    softmax_lse,  # log-sum-exp values from the forward pass (needed for numerically stable backward softmax).
    cu_seqlens,
    max_seqlen,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    # Two separate RingComm objects are needed because forward and backward gradients must circulate independently.
    kv_comm = RingComm(process_group)  # handles the forward communication of k, v during backward computation (mirrors forward ring).
    d_kv_comm = RingComm(process_group)  # handles communication of the gradients dk, dv between GPUs.
    dq, dk, dv = None, None, None
    next_dk, next_dv = None, None  # placeholders for gradients received from the next GPU in the ring.

    # Allocate temporary buffers for per-block gradient computation. These buffers will hold the gradients for the current 
    # GPU before they are combined with other GPUs.
    block_dq_buffer = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    block_dk_buffer = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    block_dv_buffer = torch.empty(v.shape, dtype=v.dtype, device=v.device)

    next_k, next_v = None, None

    for step in range(kv_comm.world_size):  # Loop over each GPU in the ring.
        if step + 1 != kv_comm.world_size:
            next_k, next_v = kv_comm.send_recv_kv(k, v)  # Sends current k, v to the next GPU and receives k, v from the previous GPU.

        if step <= kv_comm.rank or not causal:
            bwd_causal = causal and step == 0
            params = get_default_args(_flash_attn_varlen_backward).copy()
            params.update(
                {
                    "dout": dout,
                    "q": q,
                    "k": k,
                    "v": v,
                    "out": out,
                    "softmax_lse": softmax_lse,
                    "dq": block_dq_buffer,
                    "dk": block_dk_buffer,
                    "dv": block_dv_buffer,
                    "cu_seqlens_q": cu_seqlens,
                    "cu_seqlens_k": cu_seqlens,
                    "max_seqlen_q": max_seqlen,
                    "max_seqlen_k": max_seqlen,
                    "dropout_p": dropout_p,
                    "softmax_scale": softmax_scale,
                    "causal": bwd_causal,
                    "alibi_slopes": alibi_slopes,
                    "deterministic": deterministic,
                }
            )

            if "window_size" in params:
                params.update({"window_size": window_size})
            else:
                params.update(
                    {
                        "window_size_left": window_size[0],
                        "window_size_right": window_size[1],
                    }
                )

            # Flash Attention backward kernel: fills block_dq/dk/dv_buffer with local gradients for this GPU and step.
            _flash_attn_varlen_backward(**params)

            if dq is None:
                dq = block_dq_buffer.to(torch.float32)
                dk = block_dk_buffer.to(torch.float32)
                dv = block_dv_buffer.to(torch.float32)
            else:
                dq += block_dq_buffer  # dq is always accumulated locally across steps.
                d_kv_comm.wait()  # ensures previous gradient communication is completed before adding new values.
                # dk and dv are combined with gradients received from the next GPU (next_dk, next_dv).
                dk = block_dk_buffer + next_dk
                dv = block_dv_buffer + next_dv

        # For steps where this GPU does not compute gradients (causal masking), it just waits for and receives dk, dv from other GPUs.
        elif step != 0:
            d_kv_comm.wait()
            dk, dv = next_dk, next_dv

        # Waits for forward key/value communication to finish and updates k, v for the next iteration.
        if step + 1 != kv_comm.world_size:
            kv_comm.wait()
            k, v = next_k, next_v

        next_dk, next_dv = d_kv_comm.send_recv_kv(dk, dv)

    d_kv_comm.wait()  # Ensures all gradient communication is complete before returning.
    return dq.to(torch.bfloat16), next_dk.to(q.dtype), next_dv.to(q.dtype)


# ── Autograd integration ───────────────────────────────────────────────────────

# Custom autograd function where we explicitly define forward and backward computations.
class RingFlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,  # context object provided by PyTorch for storing values needed in the backward pass.
        q,
        k,
        v,
        cu_seqlens,
        max_seqlen,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
        group,
    ):

        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        assert alibi_slopes is None
        k = k.contiguous()
        v = v.contiguous()
        out, softmax_lse = ring_attn_varlen_forward(
            group,
            q,
            k,
            v,
            cu_seqlens,
            max_seqlen,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=False,
        )
        ctx.save_for_backward(q, k, v, out, softmax_lse, cu_seqlens)  # Stores tensors needed for backward.
        # Stores additional non-tensor parameters for backward, needed for configuring `ring_attn_varlen_backward` correctly.
        ctx.max_seqlen = max_seqlen
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.group = group
        return out if not return_softmax else (out, softmax_lse, None)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, cu_seqlens = ctx.saved_tensors  # `ctx.saved_tensors` retrieves tensors saved during the forward pass.
        dq, dk, dv = ring_attn_varlen_backward(
            ctx.group,
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            cu_seqlens,
            ctx.max_seqlen,
            softmax_scale=ctx.softmax_scale,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
            window_size=ctx.window_size,
            alibi_slopes=ctx.alibi_slopes,
            deterministic=ctx.deterministic,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None


# Convenience wrapper for the autograd function.
def ring_flash_attn_varlen_func(
    module,
    q,
    k,
    v,
    cu_seqlens=None,
    max_seqlen=None,
    dropout=0.0,
    scaling=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window.
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    ring_pg=None,
    **kwargs,
):
    assert cu_seqlens is not None
    assert max_seqlen is not None
    return (
        RingFlashAttnVarlenFunc.apply(
            q,
            k,
            v,
            cu_seqlens,
            max_seqlen,
            dropout,
            scaling,
            causal,
            window_size,
            alibi_slopes,
            deterministic,
            return_attn_probs,
            ring_pg,
        ),
    )