import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
from pdb import set_trace as stx
import cv2
from basicsr.utils.registry import ARCH_REGISTRY


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class Kmeans_Feature(nn.Module):
    def __init__(self, n_fea_middle, n_fea_in=4, stride=1, padding=0):
        super(Kmeans_Feature, self).__init__()

        # 第一个卷积层
        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)

        # 深度卷积层
        self.depth_conv = nn.Conv2d(
            n_fea_middle,
            n_fea_middle,
            kernel_size=5,
            stride=stride,
            padding=padding,
            bias=True,
            groups=n_fea_in,
        )

    def forward(self, img):
        x_1 = self.conv1(img)
        kmeans_feature = self.depth_conv(x_1)
        return kmeans_feature


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=(kernel_size // 2),
        bias=bias,
        stride=stride,
    )


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2).contiguous())
        return out.permute(0, 2, 3, 1)


class IGAB(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        num_blocks=2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            # 对于 IG_MSA的 LN 来说，它的LN应该是写到里面了 即 F.normalize(
            self.blocks.append(
                nn.ModuleList(
                    [
                        IG_MSA(dim=dim, dim_head=dim_head, heads=heads),
                        PreNorm(dim, FeedForward(dim=dim)),
                    ]
                )
            )

    def forward(self, x, illu_fea):
        """
        x: [b,c,h,w]
        illu_fea: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for attn, ff in self.blocks:
            x = attn(x, illu_fea_trans=illu_fea.permute(0, 2, 3, 1)) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out


class IG_MSA(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        # 注意，这里的参数设置，恰好使得 dim = c = dim_head * heads，因此最后的v_inp.reshape(b, h, w, c)才不报错
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in, illu_fea_trans):
        # print('x',x_in.shape)
        # print(illu_fea_trans.shape)
        """
        x_in: [b,h,w,c]         # input_feature
        illu_fea: [b,h,w,c]         # mask shift? 为什么是 b, h, w, c?
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        # print("x:",x.shape)
        # print("v_inp1:",v_inp.shape)
        # q_inp: b,hw,dim_head*heads
        illu_attn = illu_fea_trans  # illu_fea: b,c,h,w -> b,h,w,c
        q, k, v, illu_attn = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads),
            (q_inp, k_inp, v_inp, illu_attn.flatten(1, 2)),
        )

        v = v * illu_attn
        # q: b,heads,hw,dim_head
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        # q: b,heads,dim_head,hw
        # k: b heads dim_head hw
        q = F.normalize(q, dim=-1, p=2)
        # 对 hw 做normailize这段代码使用了 PyTorch 的函数 normalize，用于对张量进行标准化操作。具体来说，它将张量 q 的最后一个维度（即 dim=-1）进行 L2 范数归一化，使得每个向量的模长都为 1。

        # 这种标准化操作常用于注意力机制中，可以使得不同维度的特征在计算注意力权重时具有相同的重要性，从而提高模型的泛化能力。

        # 注意，这里使用的 L2 范数归一化是一种常见的标准化方式，但在某些情况下可能会使用其他的标准化方式。
        k = F.normalize(k, dim=-1, p=2)

        # q.transpose(-2,-1): b heads hw c
        attn = k @ q.transpose(-2, -1)  # A = K^T*Q
        # attn: b heads c c (c=dim_head)
        attn = attn * self.rescale
        # 具体来说，这里的 self.rescale 是一个形状为 (heads, 1, 1) 的可学习参数，其中 heads 表示注意力头的数量。为了对注意力分数进行缩放，代码使用了广播机制，将 self.rescale 沿着第二个和第三个维度进行复制，得到一个形状与 attn 相同的张量。然后，将这两个张量相乘，就完成了对注意力分数的缩放操作。

        attn = attn.softmax(dim=-1)
        # v:b,heads,dim_head,hw
        # attn: b heads c c (c=dim_head)
        x = attn @ v  # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)  # Transpose
        # x: b,hw,head,dim_head
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        # print("v_inp2:",v_inp.shape)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)).permute(
            0, 2, 3, 1
        )
        out = out_c + out_p

        return out


@ARCH_REGISTRY.register()
class FlareFormer(nn.Module):
    def __init__(
        self, in_channels=3, output_ch=6, dim=40, level=2, num_blocks=[1, 2, 2],**kwargs
    ):
        super(FlareFormer, self).__init__()
        self.dim = dim
        self.level = level
        ## input projection
        self.embedding = nn.Conv2d(in_channels, self.dim, 3, 1, 1, bias=False)
        # encoder
        self.encoder_layers = nn.ModuleList([])
        dim_level = dim

        for i in range(level):
            self.encoder_layers.append(
                nn.ModuleList(
                    [
                        IGAB(
                            dim=dim_level,
                            num_blocks=num_blocks[i],
                            dim_head=dim,
                            heads=dim_level // dim,
                        ),
                        nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                        Kmeans_Feature(
                            n_fea_middle=dim_level, stride=pow(2, i), padding=2
                        ),
                    ]
                )
            )
            dim_level *= 2
            # Bottleneck
        self.bottleneck_kmeans = Kmeans_Feature(
            n_fea_middle=dim_level, stride=4, padding=2
        )
        self.bottleneck = IGAB(
            dim=dim_level,
            dim_head=dim,
            heads=dim_level // dim,
            num_blocks=num_blocks[-1],
        )

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(
                nn.ModuleList(
                    [
                        nn.ConvTranspose2d(
                            dim_level,
                            dim_level // 2,
                            stride=2,
                            kernel_size=2,
                            padding=0,
                            output_padding=0,
                        ),
                        nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                        IGAB(
                            dim=dim_level // 2,
                            num_blocks=num_blocks[level - 1 - i],
                            dim_head=dim,
                            heads=(dim_level // 2) // dim,
                        ),
                    ]
                )
            )
            dim_level //= 2
        # Output projection
        self.mapping = nn.Conv2d(self.dim, output_ch, 3, 1, 1, bias=False)
        self.sig = nn.Sigmoid()
        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, img, kmeans_imgs):
        B, C, H, W = img.shape
        fea = self.embedding(img)

        fea_encoder = []
        kmeans_fea_list = []

        for i, (IGAB, FeaDownSample, KmeansDown) in enumerate(self.encoder_layers):
            kmeans_fea = KmeansDown(
                torch.cat((img, kmeans_imgs[i].reshape(B, 1, H, W)), dim=1)
            )
            kmeans_fea_list.append(kmeans_fea)

            fea = IGAB(fea, kmeans_fea)
            # print('-----',i)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)

        kmeans_fea = self.bottleneck_kmeans(
            torch.cat((img, kmeans_imgs[-1].reshape(B, 1, H, W)), dim=1)
        )

        fea = self.bottleneck(fea, kmeans_fea)

        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
            kmeans_fea = kmeans_fea_list[self.level - 1 - i]
            fea = LeWinBlcok(fea, kmeans_fea)
        out = self.sig(self.mapping(fea))

        _, out_c, _, _ = out.shape

        if out_c == 3:

            out = out + img
        return out
