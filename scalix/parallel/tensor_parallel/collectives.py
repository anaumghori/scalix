from typing import Optional
import torch
from torch import distributed as torch_dist
from scalix import distributed as dist
from scalix.distributed import ProcessGroup


# PyTorch's distributed communication primitives (all_reduce, all_gather, reduce_scatter) are not
# integrated with autograd. They mutate tensors in-place and define no gradient propagation rules,
# so autograd treats them as opaque and cannot differentiate through them. In tensor parallel
# training, communication operations sit inside the computational graph, meaning gradients must
# flow correctly across workers. The classes below wrap each collective in a torch.autograd.Function 
# that explicitly defines both the forward pass (the collective itself) and the backward pass 
# (the adjoint collective that carries gradients in the reverse direction).

# The duality between forward and backward collectives reflects the mathematical structure of tensor parallelism:
#   Identity      forward  - AllReduce    backward  (column parallel: gather gradients)
#   AllReduce     forward  -  Identity     backward  (row parallel: pass gradients through)
#   AllGather     forward  -  ReduceScatter backward  (unshard activations, shard gradients)
#   ReduceScatter forward  -  AllGather    backward  (shard activations, unshard gradients)


class _Identity(torch.autograd.Function):
    # Forward pass is a no-op: the tensor passes through unchanged.
    # Backward pass performs an all-reduce so that gradients from all workers are summed
    # before flowing into the layer that produced this tensor. 
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group: Optional[ProcessGroup]):
        ctx.group = group
        return tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return _AllReduce.apply(grad_output, ctx.group), None


class _AllReduce(torch.autograd.Function):
    # Forward pass sums the tensor across all workers in the group.
    # Backward pass is a plain identity: the upstream gradient is already the correct
    # per-worker gradient because no sharding occurred in the forward direction.
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group: Optional[ProcessGroup]):
        # Skip the collective when there is only one worker; the result is identical.
        if group.size() == 1:
            return tensor

        dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
        return tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


class _AllGather(torch.autograd.Function):
    # Forward pass concatenates sharded tensors from all workers into a single unsharded tensor.
    # Backward pass applies reduce-scatter, which is the adjoint operation: it shards the
    # incoming gradient and sums contributions from all workers, reversing the gather.
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group: Optional[ProcessGroup]):
        ctx.group = group

        # Skip the collective when there is only one worker; the result is identical.
        if group.size() == 1:
            return tensor

        sharded_batch_size, *rest_size = tensor.shape

        # Fall back to the default process group if none is explicitly provided.
        if group is None:
            group = torch_dist.distributed_c10d._get_default_group()

        unsharded_batch_size = sharded_batch_size * group.size()

        # Allocate the output buffer that will receive the gathered tensor from all workers.
        unsharded_tensor = torch.empty(
            unsharded_batch_size,
            *rest_size,
            device=tensor.device,
            dtype=tensor.dtype,
            requires_grad=tensor.requires_grad,
        )

        # all_gather_into_tensor requires the input to be contiguous in memory.
        tensor = tensor.contiguous()

        dist.all_gather_into_tensor(unsharded_tensor, tensor, group=group)
        return unsharded_tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return _ReduceScatter.apply(grad_output, ctx.group), None


class _ReduceScatter(torch.autograd.Function):
    # Forward pass scatters shards of the tensor across workers while summing contributions,
    # reducing the batch dimension by a factor of group.size().
    # Backward pass applies all-gather, which is the adjoint operation: it reconstructs the
    # full tensor from the per-worker gradient shards.
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, group: Optional[ProcessGroup]):
        ctx.group = group

        # Skip the collective when there is only one worker; the result is identical.
        if group.size() == 1:
            return tensor

        unsharded_batch_size, *rest_size = tensor.shape

        # Fall back to the default process group if none is explicitly provided.
        if group is None:
            group = torch_dist.distributed_c10d._get_default_group()

        assert unsharded_batch_size % group.size() == 0, (
            f"Batch size {unsharded_batch_size} must be divisible by group size {group.size()}"
        )

        # reduce_scatter_tensor requires the input to be contiguous in memory.
        tensor = tensor.contiguous()

        # Allocate the output buffer that will receive this worker's scattered shard.
        sharded_tensor = torch.empty(
            unsharded_batch_size // group.size(),
            *rest_size,
            device=tensor.device,
            dtype=tensor.dtype,
            requires_grad=False,
        )

        dist.reduce_scatter_tensor(sharded_tensor, tensor, group=group, op=dist.ReduceOp.SUM)
        return sharded_tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return _AllGather.apply(grad_output, ctx.group), None


def identity(tensor: torch.Tensor, group: Optional[ProcessGroup] = None) -> torch.Tensor:
    return _Identity.apply(tensor, group)

def all_reduce(tensor: torch.Tensor, group: Optional[ProcessGroup] = None) -> torch.Tensor:
    return _AllReduce.apply(tensor, group)

def all_gather(tensor: torch.Tensor, group: Optional[ProcessGroup] = None) -> torch.Tensor:
    return _AllGather.apply(tensor, group)

def reduce_scatter(tensor: torch.Tensor, group: Optional[ProcessGroup] = None) -> torch.Tensor:
    return _ReduceScatter.apply(tensor, group)