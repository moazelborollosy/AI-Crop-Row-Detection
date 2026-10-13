"""RowNet: one light encoder, two heads.

- segmentation head: row-likelihood map (learned from the rendered row-pattern labels)
- regression head:   the row pattern itself as [Vx, r, d, sin(phase), cos(phase)]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

IN_W, IN_H = 384, 224  # network input size
OUT_W, OUT_H = IN_W // 2, IN_H // 2  # row-likelihood map size
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
_TAPS = (1, 3, 8, 12)  # MobileNetV3-small feature taps at stride 4, 8, 16, 32
_CH = (16, 24, 48, 576)


def _conv(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class RowNet(nn.Module):
    def __init__(self, pretrained=False, width=32):
        super().__init__()
        self.pretrained = False
        weights = None
        if pretrained:
            weights = torchvision.models.MobileNet_V3_Small_Weights.DEFAULT
        try:
            net = torchvision.models.mobilenet_v3_small(weights=weights)
            self.pretrained = weights is not None
        except Exception as e:  # no network access: train from scratch
            print(f"[RowNet] ImageNet weights unavailable ({type(e).__name__}); random init")
            net = torchvision.models.mobilenet_v3_small(weights=None)
        self.features = net.features
        self.lat = nn.ModuleList(nn.Conv2d(c, width, 1) for c in _CH)
        self.smooth = nn.ModuleList(_conv(width, width) for _ in _CH[:-1])
        self.seg = nn.Conv2d(width, 1, 1)
        self.reg = nn.Sequential(
            nn.Conv2d(_CH[-1], 32, 1, bias=False), nn.BatchNorm2d(32), nn.Hardswish(),
            nn.AdaptiveAvgPool2d((4, 6)), nn.Flatten(),  # keeps coarse position information
            nn.Linear(32 * 24, 128), nn.Hardswish(), nn.Dropout(0.1), nn.Linear(128, 5))

    def forward(self, x):
        feats = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in _TAPS:
                feats.append(x)
        y = self.lat[-1](feats[-1])
        for i in range(len(feats) - 2, -1, -1):
            y = F.interpolate(y, size=feats[i].shape[-2:], mode="bilinear", align_corners=False)
            y = self.smooth[i](y + self.lat[i](feats[i]))
        # decode at stride 4 (cheap on CPU), upsample the logits to the stride-2 map
        seg = F.interpolate(self.seg(y), scale_factor=2, mode="bilinear", align_corners=False)
        return seg, self.reg(feats[-1])


def preprocess(rgb_u8):
    """HxWx3 uint8 RGB already at (IN_H, IN_W) -> normalised 1x3xHxW tensor."""
    t = torch.from_numpy(rgb_u8).permute(2, 0, 1).float().div_(255.0)
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return ((t - mean) / std).unsqueeze(0)
