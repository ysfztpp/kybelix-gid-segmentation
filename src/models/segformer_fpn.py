import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class _FeatureInfo:
    def __init__(self, channels):
        self._channels = channels

    def channels(self):
        return self._channels


class HFMiTB4Features(nn.Module):
    """
    Hugging Face fallback encoder for MiT-B4 when timm model names are unavailable.
    Returns 4 feature maps in high-res -> low-res order to match FPNFusionDecoder.
    """

    def __init__(self, pretrained=True):
        super().__init__()
        try:
            from transformers import SegformerConfig, SegformerModel
        except Exception as exc:
            raise RuntimeError(
                "MiT-B4 not found in timm and transformers is unavailable. "
                "Install: pip install -U transformers"
            ) from exc

        model_id = "nvidia/mit-b4"
        if pretrained:
            self.model = SegformerModel.from_pretrained(model_id, use_safetensors=False)
        else:
            cfg = SegformerConfig.from_pretrained(model_id)
            self.model = SegformerModel(cfg)
        # Reduces activation memory for deep MiT-B4 transformer blocks.
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

        self.feature_info = _FeatureInfo(list(self.model.config.hidden_sizes))

    @staticmethod
    def _to_nchw(feat):
        if feat.dim() == 4:
            return feat
        if feat.dim() == 3:
            b, n, c = feat.shape
            side = int(n ** 0.5)
            if side * side != n:
                raise ValueError(f"Cannot reshape token sequence of length {n} to 2D map.")
            return feat.transpose(1, 2).reshape(b, c, side, side)
        raise ValueError(f"Unsupported hidden-state shape: {tuple(feat.shape)}")

    def forward(self, x):
        outputs = self.model(
            pixel_values=x,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = list(outputs.hidden_states)
        features = hidden_states[-4:]
        return [self._to_nchw(feat) for feat in features]


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
        self.encoder = self._build_encoder(pretrained=pretrained)
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

    @staticmethod
    def _resolve_encoder_name():
        """Resolve MiT-B4 naming differences across timm versions."""
        available = set(timm.list_models())
        preferred = ("mit_b4", "mit_b4.in1k", "mit_b4.in22k_ft_in1k")
        for name in preferred:
            if name in available:
                return name

        wildcard_patterns = ("*mit_b4*", "*segformer*")
        for pattern in wildcard_patterns:
            matches = sorted(timm.list_models(pattern))
            b4_matches = [m for m in matches if "b4" in m]
            if b4_matches:
                return b4_matches[0]

        raise RuntimeError(
            "No MiT-B4 encoder found in installed timm. "
            "Will try transformers fallback ('nvidia/mit-b4'). "
            "To inspect available names: python -c \"import timm; print(timm.__version__); "
            "print([m for m in timm.list_models('*mit*') if 'b4' in m])\""
        )

    def _build_encoder(self, pretrained=True):
        try:
            encoder_name = self._resolve_encoder_name()
            return timm.create_model(
                encoder_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=(0, 1, 2, 3),
            )
        except Exception:
            return HFMiTB4Features(pretrained=pretrained)

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
