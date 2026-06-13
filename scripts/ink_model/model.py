"""Mini U-Net for per-strip ink coverage: RGB strip in, soft alpha matte out.

Fully convolutional; height is 48 in training but any (H, W) divisible by 8 works.
~100K parameters — small enough for CPU training and sub-ms mobile inference.
"""

import torch
import torch.nn as nn


def conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class InkUNet(nn.Module):
    def __init__(self, base: int = 16):
        super().__init__()
        self.enc1 = conv_block(3, base)
        self.enc2 = conv_block(base, base * 2)
        self.enc3 = conv_block(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = conv_block(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)  # logits; sigmoid at the call site


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    m = InkUNet()
    print(f"params: {param_count(m):,}")
    y = m(torch.zeros(1, 3, 48, 320))
    print("out:", tuple(y.shape))
