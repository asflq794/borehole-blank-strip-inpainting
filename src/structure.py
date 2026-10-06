from __future__ import annotations

# Portions adapted from Dmitry Ulyanov's Deep Image Prior (Apache-2.0).
# Modified for this restoration method; see LICENSE_DIP.txt.
import math
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGE_SIZE = 512
MIXER_ALPHA_INIT = 0.5

def add_module(self, module):
    self.add_module(str(len(self) + 1), module)


class Concat(nn.Module):
    def __init__(self, dim, *args):
        super(Concat, self).__init__()
        self.dim = dim

        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def forward(self, input):
        inputs = []
        for module in self._modules.values():
            inputs.append(module(input))

        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]

        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(np.array(inputs_shapes3) == min(inputs_shapes3)):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)

            inputs_ = []
            for inp in inputs:
                diff2 = (inp.size(2) - target_shape2) // 2
                diff3 = (inp.size(3) - target_shape3) // 2
                inputs_.append(inp[:, :, diff2: diff2 + target_shape2, diff3:diff3 + target_shape3])

        return torch.cat(inputs_, dim=self.dim)

    def __len__(self):
        return len(self._modules)


class Swish(nn.Module):
    """
        https://arxiv.org/abs/1710.05941
        The hype was so huge that I could not help but try it
    """
    def __init__(self):
        super(Swish, self).__init__()
        self.s = nn.Sigmoid()

    def forward(self, x):
        return x * self.s(x)


def act(act_fun = 'LeakyReLU'):
    '''
        Either string defining an activation function or module (e.g. nn.ReLU)
    '''
    if isinstance(act_fun, str):
        if act_fun == 'LeakyReLU':
            return nn.LeakyReLU(0.2, inplace=True)
        elif act_fun == 'Swish':
            return Swish()
        elif act_fun == 'ELU':
            return nn.ELU()
        elif act_fun == 'none':
            return nn.Sequential()
        else:
            assert False
    else:
        return act_fun()


def bn(num_features):
    return nn.BatchNorm2d(num_features)


torch.nn.Module.add = add_module

def conv(in_f, out_f, kernel_size, stride=1, bias=True, pad="zero", downsample_mode="stride"):
    if downsample_mode != "stride":
        raise ValueError("The released StructureNet uses stride downsampling only")
    to_pad = int((kernel_size - 1) / 2)
    padder = None
    if pad == "reflection":
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0
    convolver = nn.Conv2d(in_f, out_f, kernel_size, stride, padding=to_pad, bias=bias)
    return nn.Sequential(*(layer for layer in (padder, convolver) if layer is not None))


class AsymmetricStructureEnhancer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_h = nn.Conv2d(dim, dim, (1, 11), padding=(0, 5), groups=dim, padding_mode='reflect')
        self.conv_v = nn.Conv2d(dim, dim, (11, 1), padding=(5, 0), groups=dim, padding_mode='reflect')
        self.conv_c = nn.Conv2d(dim, dim, 3, padding=1, padding_mode='reflect')

        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.fuse = nn.Conv2d(dim * 3, dim, 1)

    def forward(self, x):
        xh = self.act(self.conv_h(x))
        xv = self.act(self.conv_v(x))
        xc = self.act(self.conv_c(x))
        out = self.fuse(torch.cat([xh, xv, xc], dim=1))
        return out + x


def skip(
        num_input_channels=2, num_output_channels=3,
        num_channels_down=[16, 32, 64, 128, 128], num_channels_up=[16, 32, 64, 128, 128],
        num_channels_skip=[4, 4, 4, 4, 4],
        filter_size_down=3, filter_size_up=3, filter_skip_size=1,
        need_sigmoid=True, need_bias=True,
        pad='zero', upsample_mode='nearest', downsample_mode='stride', act_fun='LeakyReLU',
        need1x1_up=True):
    assert len(num_channels_down) == len(num_channels_up) == len(num_channels_skip)
    n_scales = len(num_channels_down)

    if not isinstance(upsample_mode, (list, tuple)):
        upsample_mode = [upsample_mode] * n_scales
    if not isinstance(downsample_mode, (list, tuple)):
        downsample_mode = [downsample_mode] * n_scales
    if not isinstance(filter_size_down, (list, tuple)):
        filter_size_down = [filter_size_down] * n_scales
    if not isinstance(filter_size_up, (list, tuple)):
        filter_size_up = [filter_size_up] * n_scales

    last_scale = n_scales - 1
    model = nn.Sequential()
    model_tmp = model
    input_depth = num_input_channels

    for i in range(len(num_channels_down)):
        deeper = nn.Sequential()
        skip_net = nn.Sequential()

        if num_channels_skip[i] != 0:
            model_tmp.add(Concat(1, skip_net, deeper))
        else:
            model_tmp.add(deeper)

        model_tmp.add(
            bn(num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i]))
        )

        if num_channels_skip[i] != 0:
            skip_net.add(conv(input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
            skip_net.add(bn(num_channels_skip[i]))
            skip_net.add(act(act_fun))

        deeper.add(conv(
            input_depth, num_channels_down[i], filter_size_down[i], 2,
            bias=need_bias, pad=pad, downsample_mode=downsample_mode[i]
        ))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        deeper.add(conv(num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))
        deeper.add(bn(num_channels_down[i]))
        deeper.add(act(act_fun))

        deeper_main = nn.Sequential()

        if i == len(num_channels_down) - 1:
            k = num_channels_down[i]
        else:
            deeper.add(deeper_main)
            k = num_channels_up[i + 1]

        deeper.add(nn.Upsample(scale_factor=2, mode=upsample_mode[i]))

        model_tmp.add(conv(num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
        model_tmp.add(bn(num_channels_up[i]))
        model_tmp.add(act(act_fun))

        if need1x1_up:
            model_tmp.add(conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
            model_tmp.add(bn(num_channels_up[i]))
            model_tmp.add(act(act_fun))

        input_depth = num_channels_down[i]
        model_tmp = deeper_main

    model.add(conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))

    if need_sigmoid:
        model.add(nn.Sigmoid())

    return model


class StructureNet(nn.Module):
    def __init__(self, input_depth, out_channels=3, depth=6, pad='reflection'):
        super().__init__()
        self.core = skip(
            input_depth, 32,
            num_channels_down=[32, 64, 128, 128, 128, 128][:depth],
            num_channels_up=[32, 64, 128, 128, 128, 128][:depth],
            num_channels_skip=[0, 0, 0, 0, 0, 0][:depth],
            filter_size_down=5, filter_size_up=5, filter_skip_size=1,
            upsample_mode='bilinear', need1x1_up=False,
            need_sigmoid=False, need_bias=True, pad=pad
        )
        self.ase = AsymmetricStructureEnhancer(dim=32)
        self.head = nn.Conv2d(32, out_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, use_modules=True):
        feat = self.core(x)
        if use_modules:
            feat = self.ase(feat)
        return self.sigmoid(self.head(feat))


def get_structured_noise(input_depth: int, spatial_size: tuple[int, int]) -> torch.Tensor:
    h, w = spatial_size
    noise = torch.randn(1, input_depth, h, w)
    x = torch.linspace(0, 1, w).repeat(h, 1)
    y = torch.linspace(0, 1, h).unsqueeze(1).repeat(1, w)
    freq = 10.0
    angle = 0.3
    direction = x * torch.cos(torch.tensor(angle)) + y * torch.sin(torch.tensor(angle))
    sinusoid = torch.sin(freq * direction).unsqueeze(0).unsqueeze(0)
    noise[:, 0:1, :, :] += sinusoid
    return noise


def scalar(x: torch.Tensor | float) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)


class HorizontalDepthwiseBranch(nn.Module):
    def __init__(self, channels: int, dilation_width: int) -> None:
        super().__init__()
        pad_w = 4 * dilation_width
        self.pad = nn.ReflectionPad2d((pad_w, pad_w, 0, 0))
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=(1, 9),
            dilation=(1, dilation_width),
            groups=channels,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pad(x))


class HorizontalCrossGapMixer(nn.Module):
    def __init__(self, channels: int = 32, alpha_init: float = MIXER_ALPHA_INIT) -> None:
        super().__init__()
        self.branch_d1 = HorizontalDepthwiseBranch(channels, dilation_width=1)
        self.branch_d2 = HorizontalDepthwiseBranch(channels, dilation_width=2)
        self.branch_d4 = HorizontalDepthwiseBranch(channels, dilation_width=4)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1),
        )
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.last_context_rms = math.nan

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        mixed = self.fuse(torch.cat([feat, self.branch_d1(feat), self.branch_d2(feat), self.branch_d4(feat)], dim=1))
        self.last_context_rms = scalar(torch.sqrt(torch.mean(mixed.detach() * mixed.detach()) + 1e-12))
        return feat + torch.clamp(self.alpha, 0.0, 2.0) * mixed


class TeleaHorizontalASEStage1Net(nn.Module):
    """StructureNet core/ASE/head with horizontal mixer inserted before RGB head."""

    def __init__(self, input_depth: int, out_channels: int = 3, depth: int = 6, pad: str = "reflection") -> None:
        super().__init__()
        base_model = StructureNet(input_depth, out_channels, depth=depth, pad=pad)
        self.core = base_model.core
        self.ase = base_model.ase
        self.mixer = HorizontalCrossGapMixer(channels=32, alpha_init=MIXER_ALPHA_INIT)
        self.head = base_model.head
        self.sigmoid = base_model.sigmoid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.core(x)
        feat = self.ase(feat)
        feat = self.mixer(feat)
        out = self.head(feat)
        return self.sigmoid(out)


def build_telea_scaffold(
    image_u8: np.ndarray,
    hole: np.ndarray,
    radius: float = 3.0,
) -> np.ndarray:
    if image_u8.dtype != np.uint8:
        raise RuntimeError(f"Telea scaffold requires uint8 image, got {image_u8.dtype}")
    if image_u8.ndim != 3 or image_u8.shape[2] != 3:
        raise RuntimeError(f"Telea scaffold requires HxWx3 image, got {image_u8.shape}")
    if image_u8.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or hole.shape != (IMAGE_SIZE, IMAGE_SIZE):
        raise RuntimeError(f"Bad Telea scaffold input shapes: image={image_u8.shape}, hole={hole.shape}")
    hole = hole.astype(bool)
    known = ~hole
    if not np.any(known):
        raise RuntimeError("Cannot build Telea scaffold: no known pixels")

    src = image_u8.copy()
    src[hole] = 0
    mask_u8 = hole.astype(np.uint8) * 255
    scaffold = cv2.inpaint(
        src,
        mask_u8,
        inpaintRadius=float(radius),
        flags=cv2.INPAINT_TELEA,
    )
    scaffold[known] = image_u8[known]
    return scaffold.astype(np.uint8)
