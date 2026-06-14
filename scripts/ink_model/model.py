"""Mini U-Net for per-strip ink coverage: RGB strip in, soft alpha matte out.

Fully convolutional; height is 48 in training. `levels=N` needs (H, W) divisible by
2**N (so levels<=4 at H=48: 48/16=3). Each extra level enlarges the receptive field
so the interiors of thick/superbold strokes (edge-starved at 2 levels) get filled, not
just outlined; depth buys reach far more cheaply (in FLOPs) than wider channels do.
Layer names are kept fixed per level so older 2-/3-level checkpoints still load.
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
    def __init__(self, base: int = 16, levels: int = 2):
        super().__init__()
        self.levels = levels
        self.enc1 = conv_block(3, base)
        self.enc2 = conv_block(base, base * 2)
        self.enc3 = conv_block(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        if levels >= 3:
            self.enc4 = conv_block(base * 4, base * 8)
            self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
            self.dec3 = conv_block(base * 8, base * 4)
        if levels >= 4:
            self.enc5 = conv_block(base * 8, base * 16)
            self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
            self.dec4 = conv_block(base * 16, base * 8)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = conv_block(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = conv_block(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        if self.levels >= 3:
            e4 = self.enc4(self.pool(e3))
            if self.levels >= 4:
                e5 = self.enc5(self.pool(e4))
                e4 = self.dec4(torch.cat([self.up4(e5), e4], dim=1))
            e3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
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
