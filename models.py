import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from torch import Tensor

from audio import mel_spectrogram

LRELU_SLOPE = 0.1

''' ↓↓↓ HiFiGAN ↓↓↓ '''

def init_weights(m:Conv2d, mean:float=0.0, std:float=0.01):
    classname = m.__class__.__name__
    if 'Conv' in classname:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size:int, dilation:int=1):
    return int((kernel_size * dilation - dilation) / 2)


class ResBlock1(nn.Module):

    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()

        self.h = h
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=get_padding(kernel_size, dilation[1]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2], padding=get_padding(kernel_size, dilation[2]))),
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))),
        ])
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1: remove_weight_norm(l)
        for l in self.convs2: remove_weight_norm(l)


class ResBlock2(nn.Module):

    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()

        self.h = h
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0], padding=get_padding(kernel_size, dilation[0]))),
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1], padding=get_padding(kernel_size, dilation[1]))),
        ])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs: remove_weight_norm(l)


class Generator(nn.Module):

    def __init__(self, h):
        super().__init__()

        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(Conv1d(80, h.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(weight_norm(ConvTranspose1d(h.upsample_initial_channel//(2**i), h.upsample_initial_channel//(2**(i+1)), k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups: remove_weight_norm(l)
        for l in self.resblocks: l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class DiscriminatorP(nn.Module):

    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super().__init__()

        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1,    32,   (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(32,   128,  (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(128,  512,  (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(512,  1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), 'reflect')
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs = [], []
        fmap_rs, fmap_gs = [], []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(nn.Module):

    def __init__(self, use_spectral_norm=False):
        super().__init__()

        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1,    128,  15, 1, padding=7)),
            norm_f(Conv1d(128,  128,  41, 2, groups=4,  padding=20)),
            norm_f(Conv1d(128,  256,  41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256,  512,  41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512,  1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024,  5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=2),
            AvgPool1d(4, 2, padding=2),
        ])

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs = [], []
        fmap_rs, fmap_gs = [], []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))
    return loss*2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses, g_losses = [], []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())
    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l
    return loss, gen_losses


''' ↓↓↓ UnivNet ↓↓↓ '''

class DiscriminatorR(nn.Module):

    def __init__(self, resolution):
        super().__init__()

        self.resolution = resolution

        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1,  32, (3, 9),                padding=(1, 4))),
            weight_norm(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(32, 32, (3, 3),                padding=(1, 1))),
        ])
        self.conv_post = weight_norm(nn.Conv2d(32, 1, (3, 3), padding=(1, 1)))

    def forward(self, x):
        fmap = []
        x = self.spectrogram(x)
        x = x.unsqueeze(1)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.2)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return fmap, x

    def spectrogram(self, x):
        n_fft, hop_length, win_length = self.resolution
        x = F.pad(x, (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)), mode='reflect')
        x = x.squeeze(1)
        x = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=False, return_complex=True)
        return torch.abs(x)  # [B, F, TT]


class MultiResolutionDiscriminator(nn.Module):

    def __init__(self):
        super().__init__()

        # (n_fft, hop_len, win_len)
        self.resolutions = [
            (1024, 120, 600),
            (2048, 240, 1200),
            (512,   50, 240),
        ]
        self.discriminators = nn.ModuleList([
            DiscriminatorR(resolution) for resolution in self.resolutions
        ])

    def forward(self, x):
        return [disc(x) for disc in self.discriminators]


''' ↓↓↓ RefineGAN ↓↓↓ '''

class Refiner(nn.Module):

    ''' As we call the RefineGAN's generator '''

    def __init__(self, h):
        super().__init__()

        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.n_blocks = len(h.upsample_rates)
        resblock = ResBlock1
        init_ch = h.upsample_initial_channel
        
        self.conv_pre = weight_norm(Conv1d(h.num_mels, init_ch, 7, 1, padding=3))
        self.ups = nn.ModuleList([
            weight_norm(ConvTranspose1d(init_ch//(2**i), init_ch//(2**(i+1)), k, u, padding=(k-u)//2))
                for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes))
        ])
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = init_ch//(2**(i+1))
            for k, d in zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes):
                self.resblocks.append(resblock(h, ch, k, d))
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

        # RefineGAN specific
        init_ch_y = init_ch // 2**len(h.upsample_rates) // 2        # NOTE: use half depth of decoder to reduce params
        self.conv_pre_y = weight_norm(Conv1d(1, init_ch_y, 7, 1, padding=3))
        self.downs_y = nn.ModuleList([
            weight_norm(Conv1d(init_ch_y*(2**i), init_ch_y*(2**(i+1)), k, u, padding=(k-u)//2))
                for i, (u, k) in enumerate(zip(h.upsample_rates[::-1], h.upsample_kernel_sizes[::-1]))
        ])
        assert init_ch_y*(2**(i+1)) == init_ch//2
        self.fuse_y_mel = weight_norm(Conv1d(init_ch+init_ch//2, init_ch, 1))
        self.resblocks_y = nn.ModuleList([
            resblock(h, init_ch_y*(2**(i+1)), h.resblock_y_kernel_sizes, h.resblock_y_dilation_sizes)
                for i in range(self.n_blocks)
        ])
        self.fuse_y = nn.ModuleList([
            weight_norm(Conv1d(init_ch_y*(2**(self.n_blocks-(i+1)))+init_ch//(2**(i+1)), init_ch//(2**(i+1)), 1))
                for i in range(self.n_blocks)
        ])
        self.conv_pre_y.apply(init_weights)
        self.downs_y.apply(init_weights)
        self.fuse_y.apply(init_weights)

    def forward(self, x:Tensor, y_tmpl:Tensor)-> Tensor:
        mid = []
        y = self.conv_pre_y(y_tmpl)
        for i in range(self.n_blocks):
            y = F.leaky_relu(y, LRELU_SLOPE)
            mid.append(y)
            y = self.downs_y[i](y)
            y = self.resblocks_y[i](y)
        x = self.conv_pre(x)
        x = torch.cat([x, y], dim=1)
        x = self.fuse_y_mel(x)
        for i in range(self.n_blocks):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            x = torch.cat([x, mid[self.n_blocks-i-1]], dim=1)
            x = self.fuse_y[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups: remove_weight_norm(l)
        for l in self.resblocks: l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


MaxPool = nn.MaxPool1d(160)     # NOTE: this number is chosen by me :)

def envelope_loss(y, y_g):
  loss = 0
  loss += torch.mean(torch.abs(MaxPool( y) - MaxPool( y_g)))
  loss += torch.mean(torch.abs(MaxPool(-y) - MaxPool(-y_g)))
  return loss


def multi_param_melspec_loss(y, y_g, multi_stft_params):
    loss = 0
    for n_fft, win_length, hop_length in multi_stft_params:
        y_mel   = mel_spectrogram(y,   n_fft, win_length, hop_length)
        y_g_mel = mel_spectrogram(y_g, n_fft, win_length, hop_length)
        loss += F.l1_loss(y_mel, y_g_mel)
    return loss / len(multi_stft_params)


''' ↓↓↓ RetuneGAN ↓↓↓ '''

def dynamic_loss(y, y_g):
  dyn_y   = torch.abs(MaxPool(y)   + MaxPool(-y))
  dyn_y_g = torch.abs(MaxPool(y_g) + MaxPool(-y_g))
  loss    = torch.mean(torch.abs(dyn_y - dyn_y_g))
  return loss


if __name__ == '__main__':
    from utils import load_config
    h = load_config('configs/refinegan.json')
    model = Refiner(h)
    print(model)
    print('>> param_cnt:', sum([p.numel() for p in model.parameters() if p.requires_grad]))

    x = torch.rand([1, h.num_mels, h.segment_size//256])
    y = torch.rand([1, 1, h.segment_size])
    print(x.shape)
    print(y.shape)
    o = model(x, y)
    print(o.shape)
