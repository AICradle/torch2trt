from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.gather')
@tensorrt_converter('torch.Tensor.gather')
def convert_gather(ctx):
    input = get_arg(ctx, 'input', pos=0, default=None)
    dim = get_arg(ctx, 'dim', pos=1, default=None)
    index = get_arg(ctx, 'index', pos=2, default=None)

    outputs = ctx.method_return
    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]

    start = [0] * len(input.shape[1:]) # exclude batch
    stride = [1] * len(start)
    offset = 0
    trt_dim = dim - 1

    # add slice layers
    for i, output in enumerate(outputs):
        shape = list(output.shape[1:]) # exclude batch dim
        start[trt_dim] = offset
        layer = ctx.network.add_slice(input_trt, start=start, shape=shape, stride=stride)
        output._trt = layer.get_output(0)
        offset = offset + shape[trt_dim]


class TopK(torch.nn.Module):
    def __init__(self, k):
        super(TopK, self).__init__()
        self.k = k

    def forward(self, x):
        return torch.topk(x, self.k)

@add_module_test(torch.float32, torch.device('cuda'), [(1, 4, 4), (1, 3, 4), (1, 17, 4)])
def test_TopK_basic():
    return TopK(10)
