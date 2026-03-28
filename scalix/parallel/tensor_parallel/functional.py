import math
from enum import Enum, auto
from typing import Optional
import torch
from torch.nn import functional as F

import scalix.distributed as dist
from scalix.parallel.tensor_parallel.collectives import reduce_scatter
from scalix.utils import MemoryBuffer


class TPLinearMode(Enum):
    # Selects how partial outputs are reduced after the local matrix multiply.
    # ALL_REDUCE: every worker receives the full summed output (replicates the result).
    # REDUCE_SCATTER: the summed output is split across workers (each worker holds a shard).
    ALL_REDUCE = auto()
    REDUCE_SCATTER = auto()

    def __format__(self, format_spec):
        return self.name

    def __str__(self):
        return self.name


# Sharded cross entropy

# Standard cross-entropy over a vocabulary that is partitioned across workers.
# Each worker holds a contiguous slice [start_index, end_index] of the vocabulary;
# together they own the full logit vector. The forward pass produces the exact same
# loss as if all logits were on a single device, using all-reduce to assemble the
# pieces. The backward pass returns per-shard gradients without additional communication.

class _ShardedCE(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        sharded_logits,  # (*, sharded_vocab_size) — logits for this worker's vocabulary shard.
        target,          # (*) — global class indices, one per position.
        group: dist.ProcessGroup,
    ):
        # Subtract the global maximum logit before exponentiation to prevent overflow.
        # Each worker finds its local max; an all-reduce broadcasts the global max.
        logits_max = torch.max(sharded_logits, dim=-1)[0]
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=group)
        sharded_logits = sharded_logits - logits_max.unsqueeze(dim=-1)

        # Vocabulary shard boundaries
        # Each rank owns a contiguous slice of the vocabulary. Rank r owns tokens
        # [r * sharded_vocab_size, (r+1) * sharded_vocab_size).
        sharded_vocab_size = sharded_logits.shape[-1]
        rank = dist.get_rank(group)
        start_index = rank * sharded_vocab_size
        end_index = start_index + sharded_vocab_size

        # target_mask is True for positions whose target token is NOT owned by this worker.
        # Those positions contribute nothing to the predicted logit on this worker.
        target_mask = (target < start_index) | (target >= end_index)

        # Convert global target indices to shard-local indices.
        # Example: shard owns tokens 500–749, target token 620 → local index 120.
        # Out-of-shard positions are clamped to 0 to avoid out-of-bounds indexing;
        # their contribution is zeroed out explicitly afterwards.
        masked_target = target.clone() - start_index
        masked_target[target_mask] = 0

        # Flatten to 2D, then gather the logit at each position's target index.
        logits_2d = sharded_logits.view(-1, sharded_vocab_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(start=0, end=logits_2d.shape[0], device=logits_2d.device)
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]

        # Ensure contiguity before reshaping.
        if predicted_logits_1d.is_contiguous():
            predicted_logits_1d = predicted_logits_1d.clone()
        else:
            predicted_logits_1d = predicted_logits_1d.contiguous()

        predicted_logits = predicted_logits_1d.view_as(target)
        # Zero positions not owned by this worker; their predicted logit lives on another rank.
        predicted_logits[target_mask] = 0.0
        # All-reduce sums contributions from every worker so that each rank holds the
        # complete predicted logit for every position after the reduction.
        dist.all_reduce(predicted_logits, op=dist.ReduceOp.SUM, group=group)

        # Softmax denominator: Exponentiate in-place, sum across the local vocabulary 
        # shard, then all-reduce to obtain the full softmax denominator on every worker.
        exp_logits = sharded_logits
        torch.exp(sharded_logits, out=exp_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)
        dist.all_reduce(sum_exp_logits, op=dist.ReduceOp.SUM, group=group)

        loss = torch.log(sum_exp_logits) - predicted_logits
        # Normalize to obtain softmax probabilities; saved for the backward pass.
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))

        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d)
        return loss.view_as(target)

    @staticmethod
    def backward(ctx, grad_output):
        softmax, target_mask, masked_target_1d = ctx.saved_tensors
        grad_input = softmax
        sharded_vocab_size = softmax.size()[-1]
        grad_2d = grad_input.view(-1, sharded_vocab_size)
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=grad_2d.device)
        # Cross-entropy gradient: dL/dlogit_i = softmax_i - 1(i == target).
        # Subtract 1 at the target index only for positions owned by this worker.
        # target_mask is True for foreign tokens, so (1 - target_mask.float()) is 0 there
        # and 1 for locally owned positions.
        grad_2d[arange_1d, masked_target_1d] -= 1.0 - target_mask.view(-1).float()
        grad_input.mul_(grad_output.unsqueeze(dim=-1))
        return grad_input, None, None


class _ShardedCEWithZLoss(torch.autograd.Function):
    # Extends _ShardedCE with an auxiliary z-loss that penalizes large log-partition values.
    # The z-loss stabilizes training by discouraging the log-normalizer log(sum_exp) from
    # growing without bound. It is computed in the same forward pass at no extra communication cost.

    @staticmethod
    def forward(ctx, sharded_logits, target, group: dist.ProcessGroup, z_loss_coef: float = 0.0):
        # Forward logic mirrors _ShardedCE exactly up to the loss computation.
        logits_max = torch.max(sharded_logits, dim=-1)[0]
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=group)
        sharded_logits = sharded_logits - logits_max.unsqueeze(dim=-1)
        sharded_vocab_size = sharded_logits.shape[-1]
        rank = dist.get_rank(group)
        start_index = rank * sharded_vocab_size
        end_index = start_index + sharded_vocab_size
        target_mask = (target < start_index) | (target >= end_index)
        masked_target = target.clone() - start_index
        masked_target[target_mask] = 0
        logits_2d = sharded_logits.view(-1, sharded_vocab_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(start=0, end=logits_2d.shape[0], device=logits_2d.device)
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        if predicted_logits_1d.is_contiguous():
            predicted_logits_1d = predicted_logits_1d.clone()
        else:
            predicted_logits_1d = predicted_logits_1d.contiguous()
        predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits[target_mask] = 0.0
        dist.all_reduce(predicted_logits, op=dist.ReduceOp.SUM, group=group)
        exp_logits = sharded_logits
        torch.exp(sharded_logits, out=exp_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)
        dist.all_reduce(sum_exp_logits, op=dist.ReduceOp.SUM, group=group)

        loss = torch.log(sum_exp_logits) - predicted_logits

        # log_z is the log-partition function log(sum_exp). The z-loss penalizes
        # large values of log_z to keep logits bounded during training.
        log_z = torch.log(sum_exp_logits)
        if z_loss_coef > 0.0:
            # Clamp log_z to [-20, 20] before squaring to avoid numerical overflow.
            z_loss = z_loss_coef * torch.square(log_z.clamp(min=-20.0, max=20.0))
            loss = loss + z_loss
        else:
            z_loss = torch.zeros_like(loss)

        # Normalize to obtain softmax probabilities; saved for the backward pass.
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d, log_z)
        ctx.z_loss_coef = z_loss_coef
        return loss.view_as(target), z_loss.view_as(target)

    @staticmethod
    def backward(ctx, grad_output, grad_z_loss=None):
        softmax, target_mask, masked_target_1d, log_z = ctx.saved_tensors
        z_loss_coef = ctx.z_loss_coef
        grad_input = softmax.clone()
        sharded_vocab_size = softmax.size()[-1]
        grad_2d = grad_input.view(-1, sharded_vocab_size)
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=grad_2d.device)
        # Standard cross-entropy gradient: subtract 1 at the target index for owned positions.
        grad_2d[arange_1d, masked_target_1d] -= 1.0 - target_mask.view(-1).float()
        if z_loss_coef > 0.0:
            # d(z_loss)/d(logit_i) = 2 * z_loss_coef * log_z * softmax_i.
            z_loss_grad_scale = 2.0 * z_loss_coef * log_z
            z_loss_grad = softmax * z_loss_grad_scale.unsqueeze(-1)
            grad_input = grad_input + z_loss_grad
        grad_input.mul_(grad_output.unsqueeze(dim=-1))
        return grad_input, None, None, None


def sharded_cross_entropy(
    sharded_logits: torch.Tensor,
    target: torch.Tensor,
    group: dist.ProcessGroup,
    dtype: torch.dtype = None,
    z_loss_coef: float = 0.0,
):
    # Cast logits if a target dtype is specified (e.g. float32 for numerical stability).
    if dtype is not None:
        sharded_logits = sharded_logits.to(dtype=dtype)
    if z_loss_coef > 0.0:
        return _ShardedCEWithZLoss.apply(sharded_logits, target, group, z_loss_coef)
    else:
        return _ShardedCE.apply(sharded_logits, target, group)


# Column linear

# A column-parallel linear layer shards the output dimension (weight rows) across workers.
# Each worker computes a partial output over its slice of the weight matrix.

#   ALL_REDUCE mode: input is replicated; each worker applies the local weight shard and
#     produces a full-height output. Gradients are all-reduced in the backward pass.

#   REDUCE_SCATTER mode: input is sharded across workers; an all-gather assembles the full
#     input before the linear, and a reduce-scatter shards the input gradient in the backward.

# The async variants (_ColumnLinearAsync) overlap the all-gather with local computation.
# ALL_REDUCE mode without async is handled inline in column_linear using identity().


class _ColumnLinearAsync(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, weight, bias, group, tp_mode, tp_recompute_allgather):
        ctx.use_bias = bias is not None
        ctx.tp_mode = tp_mode  # saved for backward dispatch.
        ctx.group = group
        ctx.tp_recompute_allgather = tp_recompute_allgather
        ctx.tensor_shape = tensor.size()

        if tp_mode is TPLinearMode.ALL_REDUCE:
            # Input is already replicated; apply the local weight shard directly.
            ctx.save_for_backward(tensor, weight)
            return F.linear(tensor, weight, bias)

        elif tp_mode is TPLinearMode.REDUCE_SCATTER:
            group_size = group.size()
            current_rank = dist.get_rank(group)

            if group_size == 1:
                # Single worker: skip the collective entirely.
                ctx.save_for_backward(tensor, weight)
                return F.linear(tensor, weight, bias)

            tensor = tensor.contiguous()
            # tensor: (sharded_batch, ..., hidden_size)
            sharded_batch_size, *intermediate_size, hidden_size = tensor.shape 

            if group is None:
                group = dist.distributed_c10d._get_default_group()
            # After all-gather, batch dimension is expanded across all ranks.
            gathered_batch_size = sharded_batch_size * group_size

            # Allocate the all-gather output buffer. When recompute is enabled, the buffer
            # is borrowed from a shared MemoryBuffer to reduce peak memory usage.
            if tp_recompute_allgather:
                gathered_tensor = MemoryBuffer().get(
                    "allgather", (gathered_batch_size, *intermediate_size, hidden_size)
                )
            else:
                gathered_tensor = torch.empty(
                    gathered_batch_size,
                    *intermediate_size,
                    hidden_size,
                    device=tensor.device,
                    dtype=tensor.dtype,
                    requires_grad=False,
                )

            # Launch the all-gather asynchronously so local shard computation can proceed in parallel.
            # Each worker contributes its local batch slice.
            # After completion: gathered_tensor = concat(tensor_0, tensor_1, ..., tensor_{N-1})
            handle = dist.all_gather_into_tensor(gathered_tensor, tensor, group=group, async_op=True)

            # Allocate output buffer for full gathered batch. 
            # Each worker computes ALL rows but only for its output feature shard.
            output_size = weight.shape[0]
            gathered_output = torch.empty(
                gathered_batch_size,
                *intermediate_size,
                output_size,
                device=tensor.device,
                dtype=tensor.dtype,
                requires_grad=tensor.requires_grad,
            )

            # Split the output buffer into three contiguous views:
            #   before_shard      — rows for ranks < current_rank (not yet gathered).
            #   same_device_shard — rows for current_rank (available immediately).
            #   after_shard       — rows for ranks > current_rank (not yet gathered).
            before_shard, same_device_shard, after_shard = torch.split(
                gathered_output,
                split_size_or_sections=[
                    sharded_batch_size * current_rank,
                    sharded_batch_size,
                    sharded_batch_size * (group_size - current_rank - 1),
                ],
                dim=0,
            )

            # Compute the output for the local shard immediately without waiting for the all-gather.
            first_dims = math.prod([sharded_batch_size, *intermediate_size])

            # We first flatten leading dims so the input becomes 2D: tensor → (first_dims, hidden_size)
            # Weight is stored as (output_size, hidden_size) so we use weight.t() → (hidden_size, output_size)
            
            # torch.mm performs pure matrix multiplication: out = X @ W^T
            # torch.addmm performs fused bias + matmul: out = b + X @ W^T
            # where bias is broadcast from (1, output_size) across all rows.

            # The result is written directly into the preallocated output buffer via `out=`, 
            # avoiding extra memory allocation.
            if bias is None:
                torch.mm(
                    input=tensor.view(first_dims, hidden_size),
                    mat2=weight.t(),
                    out=same_device_shard.view(first_dims, output_size),
                )
            else:
                torch.addmm(
                    input=bias[None, :],
                    mat1=tensor.view(first_dims, hidden_size),
                    mat2=weight.t(),
                    out=same_device_shard.view(first_dims, output_size),
                )

            # Wait for the all-gather to complete before computing the remaining shards.
            handle.wait()

            if tp_recompute_allgather:
                ctx.save_for_backward(tensor, weight)
            else:
                ctx.save_for_backward(gathered_tensor, weight)

            # Compute output for rows gathered from ranks before this one.
            if before_shard.numel() > 0:
                first_dims = math.prod(before_shard.shape[:-1])
                if bias is None:
                    torch.mm(
                        input=gathered_tensor[: sharded_batch_size * current_rank].view(first_dims, hidden_size),
                        mat2=weight.t(),
                        out=before_shard.view(first_dims, output_size),
                    )
                else:
                    torch.addmm(
                        input=bias[None, :],
                        mat1=gathered_tensor[: sharded_batch_size * current_rank].view(first_dims, hidden_size),
                        mat2=weight.t(),
                        out=before_shard.view(first_dims, output_size),
                    )

            # Compute output for rows gathered from ranks after this one.
            if after_shard.numel() > 0:
                first_dims = math.prod(after_shard.shape[:-1])
                if bias is None:
                    torch.mm(
                        input=gathered_tensor[sharded_batch_size * (current_rank + 1) :].view(
                            first_dims, hidden_size
                        ),
                        mat2=weight.t(),
                        out=after_shard.view(first_dims, output_size),
                    )
                else:
                    torch.addmm(
                        input=bias[None, :],
                        mat1=gathered_tensor[sharded_batch_size * (current_rank + 1) :].view(
                            first_dims, hidden_size
                        ),
                        mat2=weight.t(),
                        out=after_shard.view(first_dims, output_size),
                    )

            return gathered_output

        else:
            raise ValueError(f"Got unexpected mode: {tp_mode}.")

    @staticmethod
    def backward(ctx, grad_output):
        tensor, weight = ctx.saved_tensors
        group = ctx.group
        use_bias = ctx.use_bias
        tp_mode = ctx.tp_mode

        handle1: Optional[dist.Work] = None

        # In REDUCE_SCATTER mode with recompute enabled, the full gathered tensor was not saved
        # during the forward pass to reduce memory usage. Re-gather it now for grad_weight.
        if tp_mode is TPLinearMode.REDUCE_SCATTER and ctx.tp_recompute_allgather:
            sharded_batch_size, *rest_size = tensor.shape
            if group is None:
                group = dist.distributed_c10d._get_default_group()
            if group.size() == 1:
                total_tensor = tensor
            else:
                unsharded_batch_size = sharded_batch_size * group.size()
                unsharded_tensor = MemoryBuffer().get(
                    "allgather", (unsharded_batch_size, *rest_size), dtype=tensor.dtype
                )
                # Launch asynchronously to overlap with the input gradient computation below.
                handle1 = dist.all_gather_into_tensor(unsharded_tensor, tensor, group=group, async_op=True)
                total_tensor = unsharded_tensor
        else:
            total_tensor = tensor

        # Compute the input gradient immediately; this does not require the gathered tensor.
        grad_tensor = grad_output.matmul(weight)

        # Flatten the last two relevant dimensions for the weight gradient matmul.
        grad_output = grad_output.contiguous()
        grad_output_first_dims, grad_output_last_dim = grad_output.shape[:-1], grad_output.shape[-1]
        total_tensor_first_dims, total_tensor_last_dim = total_tensor.shape[:-1], total_tensor.shape[-1]
        grad_output = grad_output.view(math.prod(grad_output_first_dims), grad_output_last_dim)
        total_tensor = total_tensor.view(math.prod(total_tensor_first_dims), total_tensor_last_dim)

        handle2: Optional[dist.Work] = None

        # Launch the input gradient reduction asynchronously while grad_weight is computed.
        if tp_mode is TPLinearMode.REDUCE_SCATTER:
            if group.size() == 1:
                sub_grad_tensor = grad_tensor
            else:
                sub_grad_tensor = torch.empty(
                    ctx.tensor_shape, dtype=grad_tensor.dtype, device=grad_tensor.device, requires_grad=False
                )
                handle2 = dist.reduce_scatter_tensor(sub_grad_tensor, grad_tensor, group=group, async_op=True)
        elif tp_mode is TPLinearMode.ALL_REDUCE:
            handle2 = dist.all_reduce(grad_tensor, group=group, async_op=True)
        else:
            raise ValueError()

        grad_bias = grad_output.sum(dim=0) if use_bias else None

        # Wait for the re-gather (if applicable) before computing the weight gradient.
        if handle1 is not None:
            handle1.wait()
        grad_weight = grad_output.t().matmul(total_tensor)

        # Wait for the input gradient reduction to complete before returning.
        if handle2 is not None:
            handle2.wait()

        if tp_mode is TPLinearMode.REDUCE_SCATTER:
            return sub_grad_tensor, grad_weight, grad_bias, None, None, None
        elif tp_mode is TPLinearMode.ALL_REDUCE:
            return grad_tensor, grad_weight, grad_bias, None, None, None
        else:
            raise ValueError(f"Got unexpected mode: {tp_mode}.")


def column_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    group: dist.ProcessGroup,
    tp_mode: TPLinearMode,
    tp_recompute_allgather: bool = True,
):
    return _ColumnLinearAsync.apply(input, weight, bias, group, tp_mode, tp_recompute_allgather)


# Row linear
# A row-parallel linear layer shards the input dimension (weight columns) across workers.
# Each worker holds a partial input and computes a partial output; the partial outputs
# are combined across workers via all-reduce or reduce-scatter depending on the mode.
# The async variant (_RowLinearAsync) overlaps the output reduction with weight gradient
# computation. The synchronous path is handled inline in row_linear.


class _RowLinearAsync(torch.autograd.Function):
    # Async row-parallel linear, REDUCE_SCATTER mode only.
    # The reduce-scatter of the forward output is overlapped with the weight gradient
    # computation in the backward pass via an async all-gather.

    @staticmethod
    def forward(ctx, tensor, weight, bias, group, tp_mode):
        assert (
            tp_mode is TPLinearMode.REDUCE_SCATTER
        ), f"async communication in RowLinear only supports REDUCE_SCATTER, got {tp_mode}"

        if group is None:
            group = dist.distributed_c10d._get_default_group()

        ctx.use_bias = bias is not None
        ctx.group = group
        out = F.linear(tensor, weight, bias)
        # Reduce-scatter the output: each worker receives a sharded slice of the summed result.
        if group.size() > 1:
            out = reduce_scatter(out, group=group)
        ctx.save_for_backward(tensor, weight)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        tensor, weight = ctx.saved_tensors
        group = ctx.group
        use_bias = ctx.use_bias

        handle: Optional[dist.Work] = None

        sharded_batch_size, *rest_size = grad_output.shape

        # All-gather the upstream gradient so every worker can compute its weight gradient
        # against the full gradient tensor.
        if group.size() == 1:
            total_grad_output = grad_output
        else:
            unsharded_batch_size = sharded_batch_size * group.size()
            total_grad_output = MemoryBuffer().get(
                "allgather2", (unsharded_batch_size, *rest_size), dtype=tensor.dtype
            )
            grad_output = grad_output.contiguous()
            # Launch asynchronously to overlap with local input gradient computation.
            handle = dist.all_gather_into_tensor(total_grad_output, grad_output, group=group, async_op=True)

        sharded_batch_size, *rest_size_grad_output = grad_output.shape
        rest_size_grad_tensor = rest_size_grad_output[:-1] + [weight.shape[1]]

        if group.size() == 1:
            total_grad_tensor = grad_output.matmul(weight)
        else:
            unsharded_batch_size = sharded_batch_size * group.size()
            total_grad_tensor = torch.empty(
                unsharded_batch_size,
                *rest_size_grad_tensor,
                device=grad_output.device,
                dtype=grad_output.dtype,
                requires_grad=False,
            )

            # Split the input gradient buffer into three views mirroring the gather split.
            before_shard_grad_tensor, same_device_shard_grad_tensor, after_shard_grad_tensor = torch.split(
                total_grad_tensor,
                split_size_or_sections=[
                    sharded_batch_size * dist.get_rank(group),
                    sharded_batch_size,
                    sharded_batch_size * (group.size() - dist.get_rank(group) - 1),
                ],
                dim=0,
            )

            # Compute the input gradient for the local shard immediately without waiting.
            torch.mm(
                input=grad_output.view(-1, grad_output.shape[-1]),
                mat2=weight,
                out=same_device_shard_grad_tensor.view(-1, weight.shape[1]),
            )

            # Wait for the all-gather to complete before processing the remaining shards.
            if handle is not None:
                handle.wait()

            before_shard_grad_output, _, after_shard_grad_output = torch.split(
                total_grad_output,
                split_size_or_sections=[
                    sharded_batch_size * dist.get_rank(group),
                    sharded_batch_size,
                    sharded_batch_size * (group.size() - dist.get_rank(group) - 1),
                ],
                dim=0,
            )

            # Compute input gradients for rows gathered from ranks before and after this one.
            if before_shard_grad_tensor.numel() > 0:
                torch.mm(
                    input=before_shard_grad_output.view(-1, before_shard_grad_output.shape[-1]),
                    mat2=weight,
                    out=before_shard_grad_tensor.view(-1, weight.shape[1]),
                )

            if after_shard_grad_tensor.numel() > 0:
                torch.mm(
                    input=after_shard_grad_output.view(-1, after_shard_grad_output.shape[-1]),
                    mat2=weight,
                    out=after_shard_grad_tensor.view(-1, weight.shape[1]),
                )

        # Compute the weight gradient using the full gathered upstream gradient.
        tensor = tensor.contiguous()
        tensor_first_dims, tensor_last_dim = tensor.shape[:-1], tensor.shape[-1]
        tensor = tensor.view(math.prod(tensor_first_dims), tensor_last_dim)
        total_grad_output_first_dims, total_grad_output_last_dim = (
            total_grad_output.shape[:-1],
            total_grad_output.shape[-1],
        )
        total_grad_output = total_grad_output.view(math.prod(total_grad_output_first_dims), total_grad_output_last_dim)
        grad_weight = total_grad_output.t().matmul(tensor)
        grad_bias = total_grad_output.sum(dim=0) if use_bias else None

        return total_grad_tensor, grad_weight, grad_bias, None, None


def row_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    group: dist.ProcessGroup,
    tp_mode: TPLinearMode,
):
    return _RowLinearAsync.apply(input, weight, bias, group, tp_mode)