
import torch
import torch.nn as nn

# A collection of all possible operations that can be placed on an edge of the network graph
OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    'skip_connect': lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    'conv_7x1_1x7': lambda C, stride, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
        nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
        nn.BatchNorm2d(C, affine=affine)
    ),
}

class ReLUConvBN(nn.Module):
    """Standard ReLU-Conv-BatchNorm block."""
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)

class DilConv(nn.Module):
    """Dilated separable convolution."""
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)

class SepConv(nn.Module):
    """Separable convolution."""
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_out, C_out, kernel_size=kernel_size, stride=1, padding=padding, groups=C_out, bias=False),
            nn.Conv2d(C_out, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)

class Identity(nn.Module):
    """Identity mapping."""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Zero(nn.Module):
    """Zero operation, effectively removing a connection."""
    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)

class FactorizedReduce(nn.Module):
    """Reduces the spatial dimensions and doubles the channel dimensions."""
    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class Resizing(nn.Module):
    """
    A utility module to resize tensors to a target channel count.
    This is used to match channel dimensions when operations with different
    channel counts are mixed.
    """
    def __init__(self, C_in, C_out, affine=True):
        super(Resizing, self).__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )
    
    def forward(self, x):
        if self.C_in == self.C_out:
            return x
        return self.op(x)


class MixedOp(nn.Module):
    """
    A differentiable mixed operation that can handle varying channel sizes.

    This module represents an edge in the network graph. It maintains a mixture
    of all possible operations, weighted by the architecture parameters alpha.
    It can now handle operations that have different output channel counts.
    """
    def __init__(self, C_in, C_out, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        self._op_channels = [] # Store output channels for each op

        # Define a set of channel options, e.g., half, same, double
        # Ensure channels are divisible by 2 for 'half'
        channel_options = {
            'half': C_in // 2 if C_in // 2 > 0 else C_in,
            'same': C_in,
            'double': C_in * 2
        }

        # First, add all channel-variant conv operations
        for primitive in OPS:
            if 'conv' in primitive:
                for size_key, C_op in channel_options.items():
                    op = OPS[primitive](C_in, C_op, stride, False)
                    self._ops.append(op)
                    self._op_channels.append(C_op)

        # Then, add all non-conv operations once
        for primitive in OPS:
            if 'conv' not in primitive:
                op = OPS[primitive](C_in, C_in, stride, False)
                if 'pool' in primitive:
                    op = nn.Sequential(op, nn.BatchNorm2d(C_in, affine=False))
                self._ops.append(op)
                self._op_channels.append(C_in)

        # Create resizers to unify output channels to C_out
        self.resizers = nn.ModuleList()
        for op_C_out in self._op_channels:
            self.resizers.append(Resizing(op_C_out, C_out, affine=False))

    def forward(self, x, weights):
        """
        Args:
            x: input tensor
            weights: a tensor of shape [num_ops], representing arch params.
        Returns:
            The weighted sum of the outputs of all operations, resized to a common
            output channel dimension C_out.
        """
        return sum(w * resizer(op(x)) 
                   for w, op, resizer in zip(weights, self._ops, self.resizers)) 