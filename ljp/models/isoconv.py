"""
        PAPER: Qian Xiang, Xiaodan Wang, Yafei Song, and Lei Lei. 2025.
        ISONet: Reforming 1DCNN for aero-engine system inter-shaft bearing
        fault diagnosis via input spatial over-parameterization,
        Expert Systems with Applications: 12724
        https://doi.org/10.1016/j.eswa.2025.127248
        Email: qianxljp@126.com
"""
from itertools import repeat
from torch._jit_internal import Optional
import torch
import math
import numpy

TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])
if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
    from torch._six import container_abcs
else:
    import collections.abc as container_abcs


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_pair = _ntuple(2)


class ISOConv2d(torch.nn.Module):
    """
        PAPER: Qian Xiang, Xiaodan Wang, Yafei Song, and Lei Lei. 2025.
        ISONet: Reforming 1DCNN for aero-engine system inter-shaft bearing
        fault diagnosis via input spatial over-parameterization,
        Expert Systems with Applications: 12724
        https://doi.org/10.1016/j.eswa.2025.127248
        Email: qianxljp@126.com

       ISOConv2d can be used as an alternative for torch.nn.Conv2d.
       The interface is similar to that of torch.nn.Conv2d.
    """
    __constants__ = ['stride', 'padding', 'dilation', 'groups',
                     'padding_mode', 'in_channels',
                     'out_channels', ]
    __annotations__ = {'bias': Optional[torch.Tensor]}

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros'):
        super(ISOConv2d, self).__init__()

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self._padding_repeated_twice = tuple(x for x in self.padding for _ in range(2))

        #################################### Initailization of F_l & G_l ###################################
        # M = self.kernel_size[0]
        # N = self.kernel_size[1]
        self.H_l = self.kernel_size[0] * self.kernel_size[1]
        ## Initailization G_l
        self.G_l = torch.nn.Parameter(torch.Tensor(out_channels, in_channels // groups, self.H_l))
        torch.nn.init.kaiming_uniform_(self.G_l, a=math.sqrt(5))

        ## Initailization of F_l
        if self.H_l > 1:
            self.F_l = torch.nn.Parameter(torch.Tensor(in_channels, self.H_l, self.H_l))
            init_zero = numpy.zeros([in_channels, self.H_l, self.H_l], dtype=numpy.float32)
            self.F_l.data = torch.from_numpy(init_zero)

            eye = torch.reshape(torch.eye(self.H_l, dtype=torch.float32), (1, self.H_l, self.H_l))
            F_l_diag = eye.repeat((in_channels, 1, self.H_l // (self.H_l)))
            self.F_l_diag = torch.nn.Parameter(F_l_diag, requires_grad=False)
        ##################################################################################################

        ## Initailization of bias
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.G_l)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(ISOConv2d, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return torch.nn.functional.conv2d(
                torch.nn.functional.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                weight, self.bias, self.stride,
                _pair(0), self.dilation, self.groups)
        return torch.nn.functional.conv2d(input, weight, self.bias, self.stride,
                                          self.padding, self.dilation, self.groups)

    def forward(self, input):
        M = self.kernel_size[0]
        N = self.kernel_size[1]
        ISO_W_shape = (self.out_channels, self.in_channels // self.groups, M, N)
        if M * N > 1:
            ######################### Compute ISOW #################
            # (input_channels, H_l, M * N)
            D = self.F_l + self.F_l_diag
            W = torch.reshape(self.G_l, (self.out_channels // self.groups, self.in_channels, self.H_l))

            # einsum outputs (out_channels // groups, in_channels, M * N),
            # which is reshaped to
            # (out_channels, in_channels // groups, M, N)
            # F_l:[in_channels, M * N, self.H_l]
            # G_l:[self.out_channels // self.groups, self.in_channels, self.H_l]

            ISO_W = torch.reshape(torch.einsum('ims,ois->oim', D, W), ISO_W_shape)
            #######################################################
        else:
            # in this case H_l == M * N
            # reshape from
            # (out_channels, in_channels // groups, H_l)
            # to
            # (out_channels, in_channels // groups, M, N)

            ISO_W = torch.reshape(self.G_l, ISO_W_shape)
        return self._conv_forward(input, ISO_W)
