import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class FPNFusionDecoder(nn.Module):
    """
    FPN-style 4-scale fusion:
    1) Lateral 1x1 projections
    2) Top-down refinement
    3) Multi-scale concat at highest spatial resolution
    """

    def __init__(self, in_channels, fpn_channels=128):
        super().__init__()
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(ch, fpn_channels, kernel_size=1) for ch in in_channels]
        )
        self.refine_convs = nn.ModuleList(
            [ConvBNReLU(fpn_channels, fpn_channels, kernel_size=3) for _ in in_channels]
        )
        self.fuse = ConvBNReLU(fpn_channels * len(in_channels), fpn_channels, kernel_size=3)

    def forward(self, features):
        # features should be ordered high-res -> low-res
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]

        # Top-down path: propagate strong semantics to finer levels.
        for i in range(len(laterals) - 2, -1, -1):
            up = F.interpolate(
                laterals[i + 1], size=laterals[i].shape[-2:], mode="bilinear", align_corners=False
            )
            laterals[i] = laterals[i] + up

        pyramids = [conv(feat) for conv, feat in zip(self.refine_convs, laterals)]
        target_size = pyramids[0].shape[-2:]
        upsampled = [
            feat
            if idx == 0
            else F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
            for idx, feat in enumerate(pyramids)
        ]
        return self.fuse(torch.cat(upsampled, dim=1))


class SegFormerB4FPNBoundary(nn.Module):
    """
    MiT-B4 encoder + FPN fusion decoder + auxiliary boundary head.
    During inference (eval mode), returns only segmentation logits.
    """

    def __init__(self, n_classes=1, pretrained=True, fpn_channels=128):
        super().__init__()
        self.encoder = timm.create_model(
            "mit_b4",
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        in_channels = self.encoder.feature_info.channels()
        if len(in_channels) != 4:
            raise ValueError(f"Expected 4 feature scales from MiT-B4, got {len(in_channels)}.")

        self.decoder = FPNFusionDecoder(in_channels=in_channels, fpn_channels=fpn_channels)
        head_channels = max(fpn_channels // 2, 32)

        self.seg_head = nn.Sequential(
            ConvBNReLU(fpn_channels, head_channels, kernel_size=3),
            nn.Conv2d(head_channels, n_classes, kernel_size=1),
        )
        self.edge_head = nn.Sequential(
            ConvBNReLU(fpn_channels, head_channels, kernel_size=3),
            nn.Conv2d(head_channels, 1, kernel_size=1),
        )

    def forward(self, x, return_aux=False):
        features = self.encoder(x)
        fused = self.decoder(features)

        seg_logits = self.seg_head(fused)
        edge_logits = self.edge_head(fused)

        out_size = (x.shape[2], x.shape[3])
        seg_logits = F.interpolate(seg_logits, size=out_size, mode="bilinear", align_corners=False)
        edge_logits = F.interpolate(edge_logits, size=out_size, mode="bilinear", align_corners=False)

        if self.training or return_aux:
            return {"seg": seg_logits, "edge": edge_logits}
        return seg_logits
