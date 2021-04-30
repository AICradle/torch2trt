from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


def squeeze(ctx, input, dim):
    layer = ctx.network.add_shuffle(trt_(ctx.network, input))

    shape = input.shape[1:dim]
    layer.reshape_dims = tuple(shape)

    return layer.get_output(0)


@tensorrt_converter('torch.gather')
@tensorrt_converter('torch.Tensor.gather')
def convert_gather(ctx):
    input = get_arg(ctx, 'input', pos=0, default=None)
    dim = get_arg(ctx, 'dim', pos=1, default=None)
    indices = get_arg(ctx, 'index', pos=2, default=None)

    output = ctx.method_return
    input_trt = add_missing_trt_tensors(ctx.network, [input], check_dtypes=False)[0]
    # TODO: create function to find singleton dimensions for squeezing.
    #  For now, we know we're squeezing in the last dimeinsion for CenterNet-type models
    indices_trt = squeeze(ctx, indices, 2)

    layer = ctx.network.add_gather(input=input_trt, indices=indices_trt, axis=dim - 1)
    output._trt = layer.get_output(0)


class Gather(torch.nn.Module):
    def __init__(self, dim, index):
        super(Gather, self).__init__()
        self.dim = dim
        self.index = index

    def forward(self, x):
        return torch.gather(x, self.dim, self.index)

class TensorGather(torch.nn.Module):
    def __init__(self, dim, index):
        super(TensorGather, self).__init__()
        self.dim = dim
        self.index = index

    def forward(self, x):
        return x.gather(self.dim, self.index)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 4, 4), (1, 3, 4), (1, 17, 4)])
def test_Gather_basic():
    return Gather(1, [1])

@add_module_test(torch.float32, torch.device('cuda'), [(1, 4, 4), (1, 3, 4), (1, 17, 4)])
def test_TensorGather_basic():
    return TensorGather(1, [1])
