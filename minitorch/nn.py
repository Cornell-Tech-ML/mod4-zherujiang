from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor

from typing import Any, List, Tuple, Optional, Union

# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    
    # Task 4.3.
    new_height = height // kh
    new_width = width // kw
    
    # Reshape the tensor to split the height dimension
    x = input.contiguous().view(batch, channel, new_height, kh, width)
    # Reshape to split the width dimension
    x = x.view(batch, channel, new_height, kh, new_width, kw)
    # Reorder dimensions to get the desired shape
    x = x.permute(0, 1, 2, 4, 3, 5)
    # Combine the kernel dimensions
    out = x.contiguous().view(batch, channel, new_height, new_width, kh * kw)
    
    return out, new_height, new_width


# Task 4.3.
def avgpool2d(a: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled average pooling 2D
    
    Args:
    ----
        a: batch x channel x height x width
        kernel: height x width of pooling
    
    Returns:
    -------
        :class:`Tensor`: Pooled tensor
        
    """
    batch, channel, height, width = a.shape
    tiled_input, new_height, new_width = tile(a, kernel)
    out = tiled_input.mean(dim=-1)
    
    return out.view(batch, channel, new_height, new_width)

# Task 4.4.

max_reduce = FastOps.reduce(operators.max, -1e9)

def argmax(a: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor.

    Args:
    ----
        a: input tensor
        dim (int): dimension to apply argmax

    Returns:
    -------
        :class:`Tensor` : tensor with 1 on highest cell in dim, 0 otherwise

    """
    out = max_reduce(a, dim)
    return out == a


class Max(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Forward pass for max should be max reduction."""
        ctx.save_for_backward(a, dim)
        return max_reduce(a, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass for max should be argmax."""
        original_tensor, dim = ctx.saved_values
        arg_max = argmax(original_tensor, int(dim.item()))
        return grad_output * arg_max, 0.0
    
    
# max = Max.apply
def max(a: Tensor, dim: Optional[int] = None) -> Tensor:
        """Computes the Max of all elements in the tensor or along a dimension."""
        if dim is None:
            return Max.apply(a.contiguous().view(a.size), tensor(0.0))
        else:
            return Max.apply(a, tensor(dim))

def softmax(a: Tensor, dim: Optional[int] = None) -> Tensor:
    """Compute the softmax as a tensor.
    math:
        softmax(a) = exp(a) / sum(exp(a))
        = exp(a - max(a)) / sum(exp(a - max(a)))
    
    Args:
    ----
        a: input tensor
        dim: dimension to apply softmax
    
    Returns:
    -------
        :class:`Tensor` : softmax tensor
        
    """
    # If dim is None, treat tensor as 1D
    if dim is None:
        a = a.contiguous().view(a.size)
        dim = 0
        
    # Subtract max for numerical stability
    a_max = max(a, dim)
    exp_a = (a - a_max).exp()
    
    # Compute sum of exponentials and divide
    sum_exp_a = exp_a.sum(dim)
    return exp_a / sum_exp_a

    
def logsoftmax(a: Tensor, dim: Optional[int] = None) -> Tensor:
    """Compute the log of the softmax as a tensor.
    math:
        log(softmax(a)) = log(exp(a - max(a)) / sum(exp(a - max(a))))
        = log(exp(a - max(a))) - log(sum(exp(a - max(a))))
        = a - max(a) - log(sum(exp(a - max(a))))
    
    Args:
    ----
        a: input tensor
        dim: dimension to apply logsoftmax
    
    Returns:
    -------
        :class:`Tensor` : logsoftmax tensor
        
    """
    if dim is None:
        a = a.contiguous().view(a.size)
        dim = 0
        
    a_max = max(a, dim)
    sum_exp_a = (a - a_max).exp().sum(dim)
    return (a - a_max) - sum_exp_a.log()


def maxpool2d(a: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled max pooling 2D
    
    Args:
    ----
        a: batch x channel x height x width
        kernel: height x width of pooling
    
    Returns:
    -------
        :class:`Tensor`: Pooled tensor
        
    """
    batch, channel, height, width = a.shape
    tiled_input, new_height, new_width = tile(a, kernel)
    out = max(tiled_input,dim=-1)
    
    return out.view(batch, channel, new_height, new_width)

def dropout(a: Tensor, p: float, ignore: bool = False) -> Tensor:
    """Dropout positions based on random noise.
    
    Args:
    ----
        a: input tensor
        p: probability [0, 1) of dropping out each position
        ignore: turn off dropout
        
    Returns:
    -------
        :class:`Tensor` : tensor with dropout applied
        
    """
    if ignore or p == 0:
        return a
    
    mask = rand(a.shape) > p
    return a * mask