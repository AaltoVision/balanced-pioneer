import torch

from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Variable

from math import sqrt

from pioneer import utils
from pioneer import config

args   = config.get_config()

def init_linear(linear):
    init.xavier_normal(linear.weight)
    linear.bias.data.zero_()


def init_conv(conv, glu=True):
    init.kaiming_normal(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()


class SpectralNorm:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')

        u = getattr(module, self.name + '_u')

        size = weight.size()
        weight_mat = weight.contiguous().view(size[0], -1)
        if weight_mat.is_cuda:
            u = u.to(device=args.device)
        v = weight_mat.t() @ u
        v = v / v.norm()
        u = weight_mat @ v
        u = u / u.norm()
        weight_sn = weight_mat / (u.t() @ weight_mat @ v)
        weight_sn = weight_sn.view(*size)

        return weight_sn, Variable(u.data)

    @staticmethod
    def apply(module, name):
        fn = SpectralNorm(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        input_size = weight.size(0)
        u = Variable(torch.randn(input_size, 1) * 0.1, requires_grad=False)
        setattr(module, name + '_u', u)
        setattr(module, name, fn.compute_weight(module)[0])

        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight_sn, u = self.compute_weight(module)
        setattr(module, self.name, weight_sn)
        setattr(module, self.name + '_u', u)


def spectral_norm(module, name='weight'):
    SpectralNorm.apply(module, name)

    return module


class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True)
                                  + 1e-8)


class SpectralNormConv2d(nn.Module):
    spectral_norm_layers = nn.ModuleList()
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        init.kaiming_normal(conv.weight)
        conv.bias.data.zero_()
        self.conv = spectral_norm(conv)

        SpectralNormConv2d.spectral_norm_layers.append(self.conv)

    def forward(self, input):
        return self.conv(input)


class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)


class NopLayer(nn.Module):
    def __init__(self):
        super(NopLayer, self).__init__()
    def forward(self, input):
        return input


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size,
                 padding,
                 kernel_size2=None, padding2=None,
                 pixel_norm=True, spectral_norm=False, use_last_act = True):
        super().__init__()

        if not use_last_act:
            last_act = NopLayer()
        else:
            last_act = nn.LeakyReLU(0.2)

        pad1 = padding
        pad2 = padding
        if padding2 is not None:
            pad2 = padding2

        kernel1 = kernel_size
        kernel2 = kernel_size
        if kernel_size2 is not None:
            kernel2 = kernel_size2

        if spectral_norm:
            self.conv = nn.Sequential(SpectralNormConv2d(in_channel,
                                                         out_channel, kernel1,
                                                         padding=pad1),
                                      nn.LeakyReLU(0.2),
                                      SpectralNormConv2d(out_channel,
                                                         out_channel, kernel2,
                                                         padding=pad2),
                                      last_act)

        else:
            pixel_norm = args.PN

            conv_type = EqualConv2d if args.EQLR else nn.Conv2d

            if pixel_norm:
                self.conv = nn.Sequential(conv_type(in_channel, out_channel,
                                                      kernel1, padding=pad1),
                                          PixelNorm(),
                                          nn.LeakyReLU(0.2),
                                          conv_type(out_channel, out_channel,
                                                      kernel2, padding=pad2),
                                          PixelNorm(),
                                          nn.LeakyReLU(0.2))
            else:
                self.conv = nn.Sequential(conv_type(in_channel, out_channel,
                                                      kernel1, padding=pad1),
                                          nn.LeakyReLU(0.2),
                                          conv_type(out_channel, out_channel,
                                                      kernel2, padding=pad2),
                                          nn.LeakyReLU(0.2))

    def forward(self, input):
        out = self.conv(input)

        return out

#  SN-only is used unless noSNG is True. In that case, PN and EQLR are switched  on depending on args.PN and args.EQLR 

class Generator(nn.Module):
    def __init__(self, nz, n_label=10):
        super().__init__()

        gen_spectral_norm = not args.noSNG

        self.label_embed = nn.Embedding(n_label, n_label)
        self.code_norm = PixelNorm()
        self.label_embed.weight.data.normal_()
        self.progression = nn.ModuleList([ConvBlock(nz, nz, 4, 3, 3, 1, spectral_norm=gen_spectral_norm),
                                          ConvBlock(nz, nz, 3, 1, spectral_norm=gen_spectral_norm),
                                          ConvBlock(nz, nz, 3, 1, spectral_norm=gen_spectral_norm),
                                          ConvBlock(nz, nz, 3, 1, spectral_norm=gen_spectral_norm),
                                          ConvBlock(nz, int(nz/2), 3, 1, spectral_norm=gen_spectral_norm),
                                          ConvBlock(int(nz/2), int(nz/4), 3, 1, spectral_norm=gen_spectral_norm),
                                          ConvBlock(int(nz/4), int(nz/8), 3, 1, spectral_norm=gen_spectral_norm),
                                          ConvBlock(int(nz/8), int(nz/16), 3, 1, spectral_norm=gen_spectral_norm),
                                          ConvBlock(int(nz/16), int(nz/32), 3, 1, spectral_norm=gen_spectral_norm)])

        self.to_rgb = nn.ModuleList([nn.Conv2d(nz, 3, 1),
                                     nn.Conv2d(nz, 3, 1),
                                     nn.Conv2d(nz, 3, 1),
                                     nn.Conv2d(nz, 3, 1),
                                     nn.Conv2d(int(nz/2), 3, 1),
                                     nn.Conv2d(int(nz/4), 3, 1),
                                     nn.Conv2d(int(nz/8), 3, 1),
                                     nn.Conv2d(int(nz/16), 3, 1),
                                     nn.Conv2d(int(nz/32), 3, 1)])

    def forward(self, input, label, step=0, alpha=-1):
        out_act = lambda x: x

        # Label is reserved for future use. Make None if not in use.
        if not label is None:
            input = torch.cat([input, label], 1)

        out = input.unsqueeze(2).unsqueeze(3)

        for i, (conv, to_rgb) in enumerate(zip(self.progression, self.to_rgb)):
            if i > 0 and step > 0:
                if args.upsampling == 'bilinear0':
                    upsample = F.upsample(out, scale_factor=2, mode=('bilinear' if i==step else 'nearest'))
                else:
                    upsample = F.upsample(out, scale_factor=2, mode=args.upsampling)

                out = conv(upsample)

            else:
                out = conv(out)

            if i == step: # The final layer is ALWAYS either to_rgb layer, or a mixture of 2 to-rgb_layers!
                out = out_act(to_rgb(out))

                if i > 0 and 0 <= alpha < 1:
                    skip_rgb = out_act(self.to_rgb[i - 1](upsample))
                    out = (1 - alpha) * skip_rgb + alpha * out

                break

        return out

pixelNormInDiscriminator = False
use_mean_std_layer = False
spectralNormInDiscriminator = True

class Discriminator(nn.Module):
    def __init__(self, nz, n_label=10, binary_predictor = True):
        super().__init__()
        self.binary_predictor = binary_predictor
        self.progression = nn.ModuleList([ConvBlock(int(nz/32), int(nz/16), 3, 1,
                                                    pixel_norm=pixelNormInDiscriminator,
                                                    spectral_norm=spectralNormInDiscriminator),
                                          ConvBlock(int(nz/16), int(nz/8), 3, 1,
                                                    pixel_norm=pixelNormInDiscriminator,
                                                    spectral_norm=spectralNormInDiscriminator),
                                          ConvBlock(int(nz/8), int(nz/4), 3, 1,
                                                    pixel_norm=pixelNormInDiscriminator,
                                                    spectral_norm=spectralNormInDiscriminator),
                                          ConvBlock(int(nz/4), int(nz/2), 3, 1,
                                                    pixel_norm=pixelNormInDiscriminator,
                                                    spectral_norm=spectralNormInDiscriminator),
                                          ConvBlock(int(nz/2), nz, 3, 1,
                                                    pixel_norm=pixelNormInDiscriminator,
                                                    spectral_norm=spectralNormInDiscriminator),
                                          ConvBlock(nz, nz, 3, 1,
                                                    pixel_norm=pixelNormInDiscriminator,
                                                    spectral_norm=spectralNormInDiscriminator),
                                          ConvBlock(nz, nz, 3, 1,
                                                    pixel_norm=pixelNormInDiscriminator,
                                                    spectral_norm=spectralNormInDiscriminator),
                                          ConvBlock(nz, nz, 3, 1,
                                                    pixel_norm=pixelNormInDiscriminator,
                                                    spectral_norm=spectralNormInDiscriminator),
                                          ConvBlock((nz+1 if use_mean_std_layer else nz), nz, 3, 1, 4, 0,
                                                    pixel_norm=pixelNormInDiscriminator,
                                                    spectral_norm=spectralNormInDiscriminator,
                                                    use_last_act=args.e_last_relu)])

        self.from_rgb = nn.ModuleList([nn.Conv2d(3, int(nz/32), 1),
                                       nn.Conv2d(3, int(nz/16), 1),
                                       nn.Conv2d(3, int(nz/8), 1),
                                       nn.Conv2d(3, int(nz/4), 1),
                                       nn.Conv2d(3, int(nz/2), 1),
                                       nn.Conv2d(3, nz, 1),
                                       nn.Conv2d(3, nz, 1),
                                       nn.Conv2d(3, nz, 1),
                                       nn.Conv2d(3, nz, 1)])

        self.n_layer = len(self.progression)

        if self.binary_predictor:
            self.linear = nn.Linear(nz, 1 + n_label)
    c = 0
    def forward(self, input, step, alpha, use_ALQ):
        for i in range(step, -1, -1):
            index = self.n_layer - i - 1

            if i == step:
                out = self.from_rgb[index](input)

            if i == 0 and use_mean_std_layer:
                mean_std = input.std(0).mean()
                mean_std = mean_std.expand(input.size(0), 1, 4, 4)
                out = torch.cat([out, mean_std], 1)

            out = self.progression[index](out)

            if i > 0:
                out = F.avg_pool2d(out, 2)

                if i == step and 0 <= alpha < 1:
                    skip_rgb = F.avg_pool2d(input, 2)
                    skip_rgb = self.from_rgb[index + 1](skip_rgb)
                    out = (1 - alpha) * skip_rgb + alpha * out
        z_out = out.squeeze(2).squeeze(2)
        if self.binary_predictor:
            out = self.linear(z_out)
            return out[:, 0], out[:, 1:]
        else:
            out = z_out.view(z_out.size(0), -1)
            ret = utils.normalize(out)

            return  ret
