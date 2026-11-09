import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class CurvilinearStripConv(nn.Module):
    """
    Anisotropic Strip Convolution Block (1xk dan kx1) untuk menangkap
    topologi filamen matahari yang memanjang dan melengkung.
    """
    def __init__(self, in_channels, out_channels, k=15):
        super().__init__()
        self.conv1x1_in = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.branch_horiz = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, (1, k), padding=(0, k // 2), groups=out_channels, bias=False),
            nn.Conv2d(out_channels, out_channels, (k, 1), padding=(k // 2, 0), groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.branch_vert = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, (k, 1), padding=(k // 2, 0), groups=out_channels, bias=False),
            nn.Conv2d(out_channels, out_channels, (1, k), padding=(0, k // 2), groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.branch_local = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.conv1x1_out = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.identity = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        res = self.identity(x)
        feat = self.conv1x1_in(x)
        return self.conv1x1_out(torch.cat([self.branch_horiz(feat), self.branch_vert(feat), self.branch_local(feat)], dim=1)) + res


class DFFNBlock(nn.Module):
    """
    Dilated / Dual-Frequency Feed-Forward Network (DFFN) dengan Gated Depthwise
    Multi-Scale Context. Menggunakan residual gate terinisialisasi 0 agar 100%
    kompatibel saat Warm-Start dari checkpoint terbaik V3.
    """
    def __init__(self, channels, expansion=2):
        super().__init__()
        hidden = channels * expansion
        self.norm = nn.BatchNorm2d(channels)
        self.proj_in = nn.Conv2d(channels, hidden * 2, 1, bias=False)
        self.dw_local = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False)
        self.dw_dilated = nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2, groups=hidden, bias=False)
        self.proj_out = nn.Conv2d(hidden, channels, 1, bias=False)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        h = self.norm(x)
        u, v = self.proj_in(h).chunk(2, dim=1)
        u = F.gelu(self.dw_local(u))
        v = self.dw_dilated(v)
        out = self.proj_out(u * v)
        return x + self.gamma * out


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

    def forward(self, x, skip=None):
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SolarFilNetV4(nn.Module):
    """
    Arsitektur Utama SolarFilNet-V4:
      - Backbone: pvt_v2_b1 (Pyramid Vision Transformer v2 B1)
      - Bottleneck: CurvilinearStripConv + DFFNBlock
      - Multi-scale Feature Pyramid Decoder + Full-Resolution Shallow Branch
      - Multi-task Heads: Mask, Spine (Skeleton), Edge, dan Pixel Contrastive (PCL)
    """
    def __init__(
        self,
        backbone_name="pvt_v2_b1",
        in_channels=3,
        decoder_channels=(256, 128, 64, 32),
        use_fullres=True,
        use_dffn=True,
        pretrained=True
    ):
        super().__init__()
        self.use_fullres = use_fullres
        self.use_dffn = use_dffn
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, in_chans=in_channels, features_only=True)
        c1, c2, c3, c4 = self.backbone.feature_info.channels()[-4:]

        self.bottleneck_cscb = CurvilinearStripConv(c4, decoder_channels[0], k=15)
        self.bottleneck_dffn = DFFNBlock(decoder_channels[0]) if use_dffn else nn.Identity()

        self.dec1 = DecoderBlock(decoder_channels[0], c3, decoder_channels[1])
        self.dec2 = DecoderBlock(decoder_channels[1], c2, decoder_channels[2])
        self.dec3 = DecoderBlock(decoder_channels[2], c1, decoder_channels[3])
        self.dec3_dffn = DFFNBlock(decoder_channels[3]) if use_dffn else nn.Identity()

        if self.use_fullres:
            self.f0 = nn.Sequential(
                nn.Conv2d(in_channels, 16, 3, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.GELU(),
                nn.Conv2d(16, 16, 3, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.GELU()
            )
            self.dec0 = DecoderBlock(decoder_channels[3], 16, 24)
            head_ch = 24
        else:
            head_ch = decoder_channels[3]

        self.mask_head = nn.Sequential(CurvilinearStripConv(head_ch, head_ch, k=9), nn.Conv2d(head_ch, 1, 1))
        self.spine_head = nn.Sequential(CurvilinearStripConv(head_ch, head_ch, k=11), nn.Conv2d(head_ch, 1, 1))
        self.edge_head = nn.Sequential(
            nn.Conv2d(head_ch, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 1, 1)
        )
        self.contrast_head = nn.Sequential(
            nn.Conv2d(decoder_channels[3], 32, 1),
            nn.GELU(),
            nn.Conv2d(32, 16, 1)
        )

    def backbone_parameters(self):
        return self.backbone.parameters()

    def forward(self, x):
        input_size = x.shape[2:]
        c1, c2, c3, c4 = self.backbone(x)[-4:]
        b = self.bottleneck_dffn(self.bottleneck_cscb(c4))
        d3 = self.dec3_dffn(self.dec3(self.dec2(self.dec1(b, c3), c2), c1))
        contrast_feat = self.contrast_head(d3)

        if self.use_fullres:
            d0 = self.dec0(d3, self.f0(x))
            mask_logits = self.mask_head(d0)
            spine_logits = self.spine_head(d0)
            edge_logits = self.edge_head(d0)
        else:
            mask_logits = F.interpolate(self.mask_head(d3), size=input_size, mode="bilinear", align_corners=False)
            spine_logits = F.interpolate(self.spine_head(d3), size=input_size, mode="bilinear", align_corners=False)
            edge_logits = F.interpolate(self.edge_head(d3), size=input_size, mode="bilinear", align_corners=False)

        return {
            "mask_logits": mask_logits,
            "spine_logits": spine_logits,
            "edge_logits": edge_logits,
            "contrast_feat": contrast_feat
        }


# Alias untuk menjaga kompatibilitas penuh apabila memuat script lama
SolarFilNetV3 = SolarFilNetV4


class ModelEMA:
    def __init__(self, model, decay=0.998):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model, step):
        d = min(self.decay, (1.0 + step) / (10.0 + step))
        for ema_v, model_v in zip(self.ema.state_dict().values(), model.state_dict().values()):
            if ema_v.dtype.is_floating_point:
                ema_v.mul_(d).add_(model_v.detach(), alpha=1.0 - d)
            else:
                ema_v.copy_(model_v)
