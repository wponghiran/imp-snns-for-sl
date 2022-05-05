
import torch
import torch.nn as nn

class FakeSpikeQuantization(nn.Module):
    def __init__(self, num_bits=8, threshold=0, ema_momentum=0.1):
        super(FakeSpikeQuantization, self).__init__()

        self.num_bits = num_bits
        self.threshold = threshold

        # We track activations ranges with exponential moving average, as proposed by Jacob et al., 2017
        # https://arxiv.org/abs/1712.05877
        # We perform bias correction on the EMA, so we keep both unbiased and biased values and the iterations count
        # For a simple discussion of this see here:
        # https://www.coursera.org/lecture/deep-neural-network/bias-correction-in-exponentially-weighted-averages-XjuhD
        self.register_buffer('ema_momentum', torch.tensor(ema_momentum))
        self.register_buffer('tracked_min_biased', torch.zeros(1))
        self.register_buffer('tracked_min', torch.zeros(1))
        self.register_buffer('tracked_max_biased', torch.zeros(1))
        self.register_buffer('tracked_max', torch.zeros(1))
        self.register_buffer('iter_count', torch.zeros(1))
        self.register_buffer('scale', torch.ones(1))
        self.register_buffer('zero_point', torch.zeros(1))
#         self.max_abs = torch.tensor(2 ** (num_bits - 1) - 1)
#         self.max_abs = torch.tensor(1.)
#         self.scale.data, self.zero_point.data = symmetric_linear_quantization_params(self.num_bits, self.max_abs, restrict_qrange=True)
#         print('max_abs={}, scale={}, zero_point={}'.format(self.max_abs,self.scale.data, self.zero_point.data))

    def forward(self, inp):
        # We update the tracked stats only in training
        #
        # Due to the way DataParallel works, we perform all updates in-place so the "main" device retains
        # its updates. (see https://pytorch.org/docs/stable/nn.html#dataparallel)
        # However, as it is now, the in-place update of iter_count causes an error when doing
        # back-prop with multiple GPUs, claiming a variable required for gradient calculation has been modified
        # in-place. Not clear why, since it's not used in any calculations that keep a gradient.
        # It works fine with a single GPU. TODO: Debug...
        if self.training:
            with torch.no_grad():
                current_min, current_max = get_tensor_min_max(inp)
            self.iter_count += 1
            self.tracked_min_biased.data, self.tracked_min.data = update_ema(self.tracked_min_biased.data,
                                                                             current_min, self.ema_momentum,
                                                                             self.iter_count)
            self.tracked_max_biased.data, self.tracked_max.data = update_ema(self.tracked_max_biased.data,
                                                                             current_max, self.ema_momentum,
                                                                             self.iter_count)

        max_abs = max(abs(self.tracked_min), abs(self.tracked_max))
        actual_min, actual_max = -max_abs, max_abs
        if self.training:
            self.scale.data, self.zero_point.data = symmetric_linear_quantization_params(self.num_bits, max_abs, restrict_qrange=True)
#         actual_min, actual_max = -self.max_abs, self.max_abs

        inp = torch.clamp(inp, actual_min.item(), actual_max.item())
#         inp = SpikeQuantizeSTE.apply(inp, self.scale, self.zero_point, self.threshold, actual_min, actual_max)
        inp = SpikeQuantizeSTE.apply(inp, self.scale, self.zero_point, self.threshold)

        return inp
        
    def update_num_bits_and_threshold(self, num_bits, threshold):
        print('num_bits & threshold are updated from {} & {} to {} & {}'.format(self.num_bits, self.threshold, num_bits, threshold))
        self.num_bits = num_bits
        self.threshold = threshold

        max_abs = max(abs(self.tracked_min), abs(self.tracked_max))
        actual_min, actual_max = -max_abs, max_abs
        self.scale.data, self.zero_point.data = symmetric_linear_quantization_params(self.num_bits, max_abs, restrict_qrange=True)

    def extra_repr(self):
        return 'num_bits={}, ema_momentum={:.4f}'.format(self.num_bits, self.ema_momentum) \
                + ', threshold={}'.format(self.threshold) \
                + ', scale={}'.format(self.scale) \
                + ', tracked_min(b/u)={:.3e}/{:e}'.format(float(self.tracked_min_biased), float(self.tracked_min)) \
                + ', tracked_max(b/u)={:.3e}/{:e}'.format(float(self.tracked_max_biased), float(self.tracked_max))

def get_tensor_min_max(t, per_dim=None):
    if per_dim is None:
        return t.min(), t.max()
    if per_dim >= t.dim():
        raise ValueError('Got per_dim={0}, but tensor only has {1} dimensions', per_dim, t.dim())
    view_dims = [t.shape[i] for i in range(per_dim + 1)] + [-1]
    tv = t.view(*view_dims)
    return tv.min(dim=-1)[0], tv.max(dim=-1)[0]

def update_ema(biased_ema, value, momentum, step):
    biased_ema = biased_ema * (1 - momentum) + momentum * value
    unbiased_ema = biased_ema / (1 - (1 - momentum) ** step)  # Bias correction
    return biased_ema, unbiased_ema

def symmetric_linear_quantization_params(num_bits, saturation_val, restrict_qrange=False):
    """
    Calculate quantization parameters assuming float range of [-saturation_val, saturation_val].
    The returned zero-point is ALWAYS set to 0.

    Setting the 'restrict_qrange' parameter limits the quantized range to N-1 bins, where N = 2 ** num_bits -1.
    This matches the symmetric quantization mode in TensorFlow which uses signed integer and limits the quantized
    range to [-127, 127] (when using 8-bits), as opposed to the "vanilla" case of [-128, 127].

    See: https://arxiv.org/abs/1806.08342, section 2.2
    """
    is_scalar, sat_val = _prep_saturation_val_tensor(saturation_val)

    if any(sat_val < 0):
        raise ValueError('Saturation value must be >= 0')

    if restrict_qrange:
        # n = 2 ** (num_bits - 1) - 1
        n = 2 ** (num_bits - 1)
    else:
        n = (2 ** num_bits - 1) / 2

    # If float values are all 0, we just want the quantized values to be 0 as well. So overriding the saturation
    # value to 'n', so the scale becomes 1
    sat_val[sat_val == 0] = n
    scale = n * (1-1e-4) / sat_val
    zero_point = torch.zeros_like(scale)

    if is_scalar:
        # If input was scalar, return scalars
        return scale.item(), zero_point.item()
    return scale, zero_point

def _prep_saturation_val_tensor(sat_val):
    is_scalar = not isinstance(sat_val, torch.Tensor)
    out = torch.tensor(sat_val) if is_scalar else sat_val.clone().detach()
    if not out.is_floating_point():
        out = out.to(torch.float32)
    if out.dim() == 0:
        out = out.unsqueeze(0)
    return is_scalar, out

class SpikeQuantizeSTE(torch.autograd.Function):
    @staticmethod
#     def forward(ctx, inp, scale, zero_point, threshold, actual_min, actual_max):
#         ctx.save_for_backward(inp, actual_min, actual_max)
#         # Truncate
#         outp = torch.trunc(scale * inp - zero_point)
#         # Threshold
#         idx = (outp >= -threshold) & (outp <= threshold)
#         outp[idx] = 0.0
#         # Dequantize
#         outp = (outp + zero_point) / scale
#         return outp
    def forward(ctx, inp, scale, zero_point, threshold):
        # # Quantize
        # outp = torch.round(scale * inp - zero_point)
        outp = torch.trunc(scale * inp - zero_point)
        # Threshold
        idx = (outp >= -threshold) & (outp <= threshold)
        outp[idx] = 0.0
        # Dequantize
        outp = (outp + zero_point) / scale
        return outp

    @staticmethod
    def backward(ctx, grad_outp):
        # Setting grad to 0.0 is unnecesary as the input is already clipping between (actual_min, actual_max)       
#         inp, actual_min, actual_max, = ctx.save_tensors
#         grad_inp = grad_outp.clone()
#         idx = (inp < actual_min) | (inp > actual_max)
#         grad_inp[idx] = 0.0
#         return grad_inp, None, None, None
        # Straight-through estimator
        grad_inp = grad_outp.clone()
        return grad_inp, None, None, None
